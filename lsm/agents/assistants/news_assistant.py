"""
News assistant agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from lsm.config.models import AgentConfig, LLMRegistryConfig, RemoteProviderConfig
from lsm.logging import get_logger
from lsm.remote.base import RemoteResult
from lsm.remote.factory import create_remote_provider

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace

logger = get_logger(__name__)


@dataclass
class NewsAssistantResult:
    """
    Structured result payload for news assistant runs.
    """

    summary_path: Path
    summary_json_path: Path
    log_path: Path


class NewsAssistantAgent(BaseAgent):
    """
    Produce newsletter-style summaries from news providers.
    """

    name = "news_assistant"
    tier = "normal"
    description = "Summarize news from multiple sources with topic filtering."
    tool_allowlist = set()
    risk_posture = "network"

    _NEWS_PROVIDER_TYPES = {
        "newsapi",
        "guardian",
        "nytimes",
        "gdelt",
        "rss",
    }

    def __init__(
        self,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
        agent_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=self.name, description=self.description)
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.agent_config = agent_config
        self.agent_overrides = agent_overrides or {}

    def run(self, initial_context: AgentContext) -> Any:
        self._tokens_used = 0
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"News Assistant: {topic}"

        request = self._parse_request(initial_context)
        providers = self._resolve_providers(request)
        payload = self._build_news_summary(providers, request)

        run_dir = self._resolve_output_dir(topic, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = run_dir / "news_summary.json"
        summary_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        summary_path = run_dir / "news_summary.md"
        summary_path.write_text(
            self._format_summary_markdown(payload),
            encoding="utf-8",
        )
        self.state.add_artifact(str(summary_json_path))
        self.state.add_artifact(str(summary_path))

        log_path = self._save_log()
        self.last_result = NewsAssistantResult(
            summary_path=summary_path,
            summary_json_path=summary_json_path,
            log_path=log_path,
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def _extract_topic(self, context: AgentContext) -> str:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() == "user":
                topic = str(message.get("content", "")).strip()
                if topic:
                    return topic
        return "News Summary"

    def _resolve_output_dir(self, topic: str, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)
        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "news_assistant"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}"

    def _parse_request(self, context: AgentContext) -> Dict[str, Any]:
        payload = self._extract_payload(context)
        overrides = self.agent_overrides if isinstance(self.agent_overrides, dict) else {}
        topics = payload.get("topics")
        if isinstance(topics, str):
            topics = [item.strip() for item in topics.split(",") if item.strip()]
        if not isinstance(topics, list):
            topics = []
        query = str(payload.get("query") or "").strip()
        max_results = payload.get("max_results")
        max_results = int(max_results) if max_results is not None else None

        now = self._resolve_now(payload, overrides)
        start, end = self._resolve_window(payload, overrides, now)

        request = {
            "topics": topics,
            "query": query,
            "max_results": max_results,
            "window_start": start,
            "window_end": end,
            "providers": overrides.get("providers") or overrides.get("provider_instances"),
            "provider_names": payload.get("providers") or overrides.get("providers"),
            "raw_payload": payload,
        }
        return request

    def _build_news_summary(
        self,
        providers: Sequence[Any],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        topics = request.get("topics") or []
        if not topics and request.get("query"):
            topics = [request.get("query")]
        if not topics:
            topics = ["general"]
        max_results = request.get("max_results") or 10
        window_start = request.get("window_start")
        window_end = request.get("window_end")

        results: List[RemoteResult] = []
        for provider in providers:
            for topic in topics:
                query = "" if topic == "general" else str(topic)
                try:
                    items = provider.search(query, max_results=max_results)
                except Exception as exc:
                    self._log(f"News provider {getattr(provider, 'name', provider)} failed: {exc}")
                    continue
                for item in items or []:
                    self._tag_provider(item, provider)
                    results.append(item)

        filtered = self._filter_results(results, topics, window_start, window_end)
        deduped = self._dedupe_results(filtered)
        indexed = self._index_by_topic(deduped, topics)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "window": {
                "start": window_start.isoformat() if window_start else None,
                "end": window_end.isoformat() if window_end else None,
            },
            "topics": topics,
            "total_stories": len(deduped),
            "stories": [self._story_summary(item) for item in deduped],
            "topics_index": indexed,
            "sources": sorted({item.get("provider") for item in indexed.values() for item in item})
            if indexed
            else [],
        }

    def _filter_results(
        self,
        results: Sequence[RemoteResult],
        topics: Sequence[str],
        window_start: Optional[datetime],
        window_end: Optional[datetime],
    ) -> List[RemoteResult]:
        filtered: List[RemoteResult] = []
        topic_terms = [str(topic).lower() for topic in topics if topic and topic != "general"]
        for item in results:
            published = self._extract_published_at(item)
            if window_start and published and published < window_start:
                continue
            if window_end and published and published > window_end:
                continue
            if topic_terms:
                haystack = f"{item.title} {item.snippet}".lower()
                if not any(term in haystack for term in topic_terms):
                    continue
            filtered.append(item)
        return filtered

    def _dedupe_results(self, results: Sequence[RemoteResult]) -> List[RemoteResult]:
        seen: set[str] = set()
        deduped: List[RemoteResult] = []
        for item in results:
            key = self._dedupe_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _index_by_topic(
        self,
        results: Sequence[RemoteResult],
        topics: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        normalized_topics = [str(topic).lower() for topic in topics if topic]
        for item in results:
            assigned = "general"
            haystack = f"{item.title} {item.snippet}".lower()
            for topic in normalized_topics:
                if topic != "general" and topic in haystack:
                    assigned = topic
                    break
            index.setdefault(assigned, []).append(self._story_summary(item))
        return index

    def _story_summary(self, item: RemoteResult) -> Dict[str, Any]:
        metadata = item.metadata or {}
        return {
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
            "score": item.score,
            "provider": metadata.get("provider"),
            "published_at": self._published_str(item),
        }

    def _published_str(self, item: RemoteResult) -> Optional[str]:
        published = self._extract_published_at(item)
        return published.isoformat() if published else None

    def _tag_provider(self, item: RemoteResult, provider: Any) -> None:
        if item.metadata is None:
            item.metadata = {}
        if not item.metadata.get("provider"):
            item.metadata["provider"] = getattr(provider, "name", None) or provider.__class__.__name__

    def _extract_published_at(self, item: RemoteResult) -> Optional[datetime]:
        if not item.metadata:
            return None
        for key in (
            "published",
            "published_date",
            "published_at",
            "date",
            "pub_date",
        ):
            value = item.metadata.get(key)
            parsed = self._parse_datetime(value)
            if parsed is not None:
                return parsed
        return None

    def _dedupe_key(self, item: RemoteResult) -> str:
        metadata = item.metadata or {}
        return str(item.url or metadata.get("source_id") or item.title or "").strip()

    def _format_summary_markdown(self, payload: Dict[str, Any]) -> str:
        lines = ["# News Briefing", ""]
        lines.append(f"- Generated at: {payload.get('generated_at', '-')}")
        window = payload.get("window") or {}
        lines.append(f"- Window start: {window.get('start', '-')}")
        lines.append(f"- Window end: {window.get('end', '-')}")
        sources = payload.get("sources") or []
        lines.append(f"- Sources: {', '.join(sources) if sources else '-'}")
        lines.append("")

        lines.append("## Top Stories")
        stories = payload.get("stories") or []
        if not stories:
            lines.append("- None")
        else:
            for story in stories[:8]:
                lines.append(
                    f"- {story.get('title')} — {story.get('provider', 'unknown')}"
                )

        lines.append("")
        lines.append("## By Topic")
        topics_index = payload.get("topics_index") or {}
        if not topics_index:
            lines.append("- None")
        else:
            for topic, items in topics_index.items():
                lines.append("")
                lines.append(f"### {topic.title()}")
                if not items:
                    lines.append("- None")
                else:
                    for item in items:
                        lines.append(
                            f"- {item.get('title')} — {item.get('provider', 'unknown')}"
                        )

        return "\n".join(lines).strip() + "\n"

    def _resolve_now(self, payload: Dict[str, Any], overrides: Dict[str, Any]) -> datetime:
        now_value = payload.get("now") or overrides.get("now")
        parsed = self._parse_datetime(now_value)
        return parsed or datetime.utcnow()

    def _resolve_window(
        self,
        payload: Dict[str, Any],
        overrides: Dict[str, Any],
        now: datetime,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        start_raw = payload.get("start") or payload.get("window_start")
        end_raw = payload.get("end") or payload.get("window_end")
        start_dt = self._parse_datetime(start_raw)
        end_dt = self._parse_datetime(end_raw)
        if start_dt or end_dt:
            return start_dt, end_dt
        hours = payload.get("window_hours") or overrides.get("default_window_hours") or 24
        try:
            hours = float(hours)
        except (TypeError, ValueError):
            hours = 24.0
        return now - timedelta(hours=hours), now

    def _resolve_providers(self, request: Dict[str, Any]) -> List[Any]:
        if request.get("providers"):
            providers = request.get("providers")
            return list(providers)
        lsm_config = self._resolve_lsm_config()
        provider_names = request.get("provider_names") or []
        if isinstance(provider_names, str):
            provider_names = [provider_names]
        if lsm_config is not None:
            candidates = [
                provider
                for provider in (lsm_config.remote_providers or [])
                if provider.type in self._NEWS_PROVIDER_TYPES
            ]
            selected: List[Any] = []
            if provider_names:
                for provider in candidates:
                    if provider.name in provider_names:
                        selected.append(self._build_provider_from_config(provider, lsm_config))
            if not selected:
                selected = [self._build_provider_from_config(provider, lsm_config) for provider in candidates]
            return selected
        raise ValueError("News providers not configured")

    def _resolve_lsm_config(self) -> Optional[Any]:
        try:
            tool = self.tool_registry.lookup("query_remote")
        except KeyError:
            return None
        return getattr(tool, "config", None)

    def _build_provider_from_config(
        self,
        provider_cfg: RemoteProviderConfig,
        lsm_config: Any,
    ) -> Any:
        config = {
            "type": provider_cfg.type,
            "weight": provider_cfg.weight,
            "api_key": provider_cfg.api_key,
            "endpoint": provider_cfg.endpoint,
            "max_results": provider_cfg.max_results,
            "language": provider_cfg.language,
            "user_agent": provider_cfg.user_agent,
            "timeout": provider_cfg.timeout,
            "min_interval_seconds": provider_cfg.min_interval_seconds,
            "section_limit": provider_cfg.section_limit,
            "snippet_max_chars": provider_cfg.snippet_max_chars,
            "include_disambiguation": provider_cfg.include_disambiguation,
        }
        if provider_cfg.extra:
            config.update(provider_cfg.extra)
        if getattr(lsm_config, "global_folder", None) is not None:
            config["global_folder"] = str(lsm_config.global_folder)
        return create_remote_provider(provider_cfg.type, config)

    def _extract_payload(self, context: AgentContext) -> Dict[str, Any]:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() != "user":
                continue
            content = message.get("content")
            if isinstance(content, dict):
                return dict(content)
            if not isinstance(content, str):
                return {}
            text = content.strip()
            if not text:
                return {}
            if text.startswith("{") and text.endswith("}"):
                parsed = self._parse_json(text)
                if isinstance(parsed, dict):
                    return parsed
            return {"query": text}
        return {}

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return parsedate_to_datetime(text)
            except Exception:
                return None
