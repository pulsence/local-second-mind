"""
Curator agent implementation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig
from lsm.providers.factory import create_provider

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace


@dataclass
class CuratorResult:
    """
    Structured result payload for a curator run.
    """

    topic: str
    report_markdown: str
    output_path: Path
    log_path: Path


class CuratorAgent(BaseAgent):
    """
    Maintain corpus quality with actionable reports.
    """

    name = "curator"
    tier = "normal"
    description = "Maintain corpus quality with actionable reports."
    risk_posture = "writes_workspace"
    tool_allowlist = {
        "read_folder",
        "file_metadata",
        "hash_file",
        "query_embeddings",
        "similarity_search",
        "write_file",
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

        self.max_iterations = int(
            self.agent_overrides.get("max_iterations", self.agent_config.max_iterations)
        )
        self.max_tokens_budget = int(
            self.agent_overrides.get(
                "max_tokens_budget",
                self.agent_config.max_tokens_budget,
            )
        )
        self.last_result: Optional[CuratorResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the curator workflow.

        Args:
            initial_context: Agent context containing the user request/topic.

        Returns:
            Agent state after execution.
        """
        self._tokens_used = 0
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )
        topic_input = self._extract_topic(initial_context)
        mode, topic = self._resolve_mode(topic_input)

        if mode == "memory":
            self.state.current_task = f"Curating memory: {topic}"
            self.last_result = self._run_memory_mode(topic, initial_context)
            self.state.set_status(AgentStatus.COMPLETED)
            return self.state

        self.state.current_task = f"Curating: {topic}"
        provider = create_provider(self._resolve_llm_config(self.llm_registry))
        scope = self._select_scope(provider, topic)
        inventory = self._inventory_files(scope)
        metadata: List[Dict[str, Any]] = []
        exact_duplicates: List[Dict[str, Any]] = []
        near_duplicates: List[Dict[str, Any]] = []
        quality_signals: List[Dict[str, Any]] = []
        heuristics: Dict[str, Any] = {
            "stale_files": [],
            "tiny_files": [],
            "empty_files": [],
            "quality_hits": [],
        }
        recommendations: List[str] = []
        if not self._is_stop_requested():
            metadata = self._collect_metadata(inventory)
        if not self._is_stop_requested():
            exact_duplicates = self._detect_exact_duplicates(inventory)
        if not self._is_stop_requested():
            near_duplicates = self._detect_near_duplicates(inventory, scope)
        if not self._is_stop_requested():
            quality_signals = self._collect_quality_signals(topic)
        if not self._is_stop_requested():
            heuristics = self._apply_heuristics(metadata, quality_signals, scope)
        if not self._is_stop_requested():
            recommendations = self._generate_recommendations(
                provider,
                topic=topic,
                scope=scope,
                inventory=inventory,
                metadata=metadata,
                exact_duplicates=exact_duplicates,
                near_duplicates=near_duplicates,
                heuristics=heuristics,
            )
        if self._handle_stop_request():
            recommendations = recommendations or ["Run stopped before full completion."]

        report_markdown = self._build_report(
            topic=topic,
            scope=scope,
            inventory=inventory,
            metadata=metadata,
            exact_duplicates=exact_duplicates,
            near_duplicates=near_duplicates,
            heuristics=heuristics,
            recommendations=recommendations,
        )
        output_path = self._save_report(topic, report_markdown, initial_context)
        log_path = self._save_log()
        self.last_result = CuratorResult(
            topic=topic,
            report_markdown=report_markdown,
            output_path=output_path,
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
        return "Corpus Curation"

    def _resolve_mode(self, topic: str) -> tuple[str, str]:
        configured_mode = str(self.agent_overrides.get("mode", "default")).strip().lower()
        topic_mode_match = re.search(r"--mode\s+([a-zA-Z_]+)", str(topic))
        mode = configured_mode
        if topic_mode_match:
            mode = str(topic_mode_match.group(1)).strip().lower()
        if mode not in {"default", "memory"}:
            mode = "default"

        cleaned_topic = re.sub(r"\s*--mode\s+[a-zA-Z_]+\s*", " ", str(topic)).strip()
        if not cleaned_topic:
            cleaned_topic = "Corpus Curation"
        return mode, cleaned_topic

    def _run_memory_mode(self, topic: str, initial_context: AgentContext) -> CuratorResult:
        if self._is_stop_requested():
            self._handle_stop_request()
        summary_limit = int(self.agent_overrides.get("memory_summary_limit", 50))
        summaries = self._load_recent_run_summaries(limit=max(1, summary_limit))
        candidates = self._build_memory_candidates_from_summaries(summaries)
        markdown = self._build_memory_candidates_markdown(topic, summaries, candidates)

        run_dir = self._resolve_output_dir(topic, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = run_dir / "memory_candidates.md"
        json_path = run_dir / "memory_candidates.json"
        markdown_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        self.state.add_artifact(str(markdown_path))
        self.state.add_artifact(str(json_path))
        self._log(f"Saved memory candidates markdown to {markdown_path}")
        self._log(f"Saved memory candidates json to {json_path}")

        log_path = self._save_log()
        return CuratorResult(
            topic=topic,
            report_markdown=markdown,
            output_path=markdown_path,
            log_path=log_path,
        )

    def _load_recent_run_summaries(self, limit: int = 50) -> List[Dict[str, Any]]:
        root = Path(self.agent_config.agents_folder)
        if not root.exists():
            return []
        paths = sorted(
            root.rglob("run_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        summaries: list[Dict[str, Any]] = []
        for path in paths[: max(1, int(limit))]:
            if self._is_stop_requested():
                break
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            parsed["summary_path"] = str(path)
            summaries.append(parsed)
        self._log(f"Loaded {len(summaries)} run summaries for memory distillation.")
        return summaries

    def _build_memory_candidates_from_summaries(
        self,
        summaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._is_stop_requested():
            return []
        tool_threshold = max(2, int(self.agent_overrides.get("memory_tool_threshold", 3)))
        constraint_threshold = max(
            2, int(self.agent_overrides.get("memory_constraint_threshold", 2))
        )
        denial_threshold = max(1, int(self.agent_overrides.get("memory_denial_threshold", 2)))
        trust_threshold = max(
            3, int(self.agent_overrides.get("memory_trusted_tool_threshold", 5))
        )

        tool_usage: Dict[str, int] = {}
        constraints: Dict[str, int] = {}
        approvals_denials: Dict[str, Dict[str, int]] = {}

        for summary in summaries:
            if self._is_stop_requested():
                break
            tool_map = summary.get("tools_used")
            if isinstance(tool_map, dict):
                for raw_tool, raw_count in tool_map.items():
                    tool_name = str(raw_tool).strip()
                    if not tool_name:
                        continue
                    try:
                        count = max(0, int(raw_count))
                    except (TypeError, ValueError):
                        continue
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + count

            summary_constraints = summary.get("constraints")
            if isinstance(summary_constraints, list):
                for raw_constraint in summary_constraints:
                    constraint = str(raw_constraint).strip()
                    if not constraint:
                        continue
                    constraints[constraint] = constraints.get(constraint, 0) + 1

            decisions = summary.get("approvals_denials")
            by_tool = decisions.get("by_tool") if isinstance(decisions, dict) else None
            if not isinstance(by_tool, dict):
                continue
            for raw_tool, raw_metrics in by_tool.items():
                tool_name = str(raw_tool).strip()
                if not tool_name or not isinstance(raw_metrics, dict):
                    continue
                entry = approvals_denials.setdefault(
                    tool_name,
                    {"approvals": 0, "denials": 0},
                )
                entry["approvals"] += self._coerce_int(raw_metrics.get("approvals"), default=0)
                entry["denials"] += self._coerce_int(raw_metrics.get("denials"), default=0)

        candidates: list[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        for tool_name, count in sorted(tool_usage.items(), key=lambda item: item[1], reverse=True):
            if count < tool_threshold:
                continue
            key = f"preferred_tool_{self._slugify(tool_name)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "key": key,
                    "type": "task_state",
                    "scope": "agent",
                    "tags": ["workflow", "tooling"],
                    "confidence": min(0.95, 0.60 + (0.05 * min(count, 6))),
                    "value": {"tool": tool_name, "usage_count": count},
                    "rationale": (
                        f"Tool '{tool_name}' appears repeatedly across recent run summaries."
                    ),
                    "provenance": "curator_memory_mode",
                }
            )

        for constraint, count in sorted(constraints.items(), key=lambda item: item[1], reverse=True):
            if count < constraint_threshold:
                continue
            key = f"constraint_{self._slugify(constraint)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "key": key,
                    "type": "project_fact",
                    "scope": "project",
                    "tags": ["constraints", "workflow"],
                    "confidence": min(0.95, 0.55 + (0.08 * min(count, 5))),
                    "value": {"constraint": constraint, "mentions": count},
                    "rationale": (
                        f"Constraint '{constraint}' is repeated in multiple run summaries."
                    ),
                    "provenance": "curator_memory_mode",
                }
            )

        for tool_name, metrics in sorted(
            approvals_denials.items(),
            key=lambda item: (item[1].get("denials", 0), item[1].get("approvals", 0)),
            reverse=True,
        ):
            approvals = self._coerce_int(metrics.get("approvals"), default=0)
            denials = self._coerce_int(metrics.get("denials"), default=0)
            if denials >= denial_threshold and denials > approvals:
                key = f"permission_guardrail_{self._slugify(tool_name)}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        {
                            "key": key,
                            "type": "task_state",
                            "scope": "agent",
                            "tags": ["permissions", "safety"],
                            "confidence": min(0.95, 0.60 + (0.05 * min(denials, 6))),
                            "value": {
                                "tool": tool_name,
                                "approvals": approvals,
                                "denials": denials,
                            },
                            "rationale": (
                                f"Tool '{tool_name}' is denied more often than approved in recent runs."
                            ),
                            "provenance": "curator_memory_mode",
                        }
                    )
            elif approvals >= trust_threshold and denials == 0:
                key = f"trusted_tool_{self._slugify(tool_name)}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        {
                            "key": key,
                            "type": "task_state",
                            "scope": "agent",
                            "tags": ["permissions", "workflow"],
                            "confidence": min(0.95, 0.60 + (0.04 * min(approvals, 8))),
                            "value": {
                                "tool": tool_name,
                                "approvals": approvals,
                                "denials": denials,
                            },
                            "rationale": (
                                f"Tool '{tool_name}' is consistently approved with no denials."
                            ),
                            "provenance": "curator_memory_mode",
                        }
                    )

        self._log(f"Generated {len(candidates)} memory candidates.")
        return candidates

    def _build_memory_candidates_markdown(
        self,
        topic: str,
        summaries: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> str:
        lines = [
            f"# Memory Candidates: {topic}",
            "",
            "## Input",
            "",
            f"- Run summaries scanned: {len(summaries)}",
            f"- Candidates generated: {len(candidates)}",
            "",
            "## Candidates",
            "",
        ]
        if not candidates:
            lines.append("No memory candidates generated from available run summaries.")
            lines.append("")
            return "\n".join(lines)

        for candidate in candidates:
            lines.append(
                f"- `{candidate['key']}` "
                f"({candidate['type']} | {candidate['scope']} | confidence={candidate['confidence']:.2f})"
            )
            lines.append(f"  - tags: {', '.join(candidate.get('tags', [])) or '-'}")
            lines.append(f"  - rationale: {candidate.get('rationale', '')}")
            value_text = json.dumps(candidate.get("value", {}), ensure_ascii=True)
            lines.append(f"  - value: `{value_text}`")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
        slug = slug.strip("_")
        return (slug or "item")[:60]

    def _available_tools(self) -> set[str]:
        return {
            str(item.get("name", "")).strip()
            for item in self._get_tool_definitions(
                self.tool_registry,
                tool_allowlist=self.tool_allowlist,
            )
            if str(item.get("name", "")).strip()
        }

    def _select_scope(self, provider: Any, topic: str) -> Dict[str, Any]:
        if self._is_stop_requested():
            return {
                "scope_path": self._default_scope_path(),
                "stale_days": 365,
                "near_duplicate_threshold": 0.9,
                "top_near_duplicates": 25,
            }
        default_scope_path = str(
            self.agent_overrides.get("scope_path", self._default_scope_path())
        )
        default_stale_days = int(self.agent_overrides.get("stale_days", 365))
        default_near_duplicate_threshold = float(
            self.agent_overrides.get("near_duplicate_threshold", 0.9)
        )
        default_top_near_duplicates = int(
            self.agent_overrides.get("top_near_duplicates", 25)
        )

        prompt = (
            "Select corpus curation scope. "
            "Return strict JSON with keys: "
            '{"scope_path":"...", "stale_days": 365, '
            '"near_duplicate_threshold": 0.9, "top_near_duplicates": 25}.'
        )
        response = provider.synthesize(prompt, f"Topic:\n{topic}", mode="insight")
        self._consume_tokens(response)
        parsed = self._parse_json(response)

        scope_path = default_scope_path
        stale_days = default_stale_days
        near_duplicate_threshold = default_near_duplicate_threshold
        top_near_duplicates = default_top_near_duplicates

        if isinstance(parsed, dict):
            parsed_scope = str(parsed.get("scope_path", "")).strip()
            if parsed_scope and "scope_path" not in self.agent_overrides:
                scope_path = parsed_scope

            try:
                stale_days = int(parsed.get("stale_days", stale_days))
            except (TypeError, ValueError):
                pass
            try:
                near_duplicate_threshold = float(
                    parsed.get("near_duplicate_threshold", near_duplicate_threshold)
                )
            except (TypeError, ValueError):
                pass
            try:
                top_near_duplicates = int(
                    parsed.get("top_near_duplicates", top_near_duplicates)
                )
            except (TypeError, ValueError):
                pass

        stale_days = max(30, min(stale_days, 3650))
        near_duplicate_threshold = max(0.5, min(near_duplicate_threshold, 0.999))
        top_near_duplicates = max(5, min(top_near_duplicates, 200))

        scope = {
            "scope_path": scope_path,
            "stale_days": stale_days,
            "near_duplicate_threshold": near_duplicate_threshold,
            "top_near_duplicates": top_near_duplicates,
        }
        self._log(f"Selected curation scope: {json.dumps(scope)}")
        return scope

    def _default_scope_path(self) -> str:
        if self.sandbox.config.allowed_read_paths:
            return str(self.sandbox.config.allowed_read_paths[0])
        return "."

    def _inventory_files(self, scope: Dict[str, Any]) -> List[str]:
        if self._is_stop_requested():
            return []
        if "read_folder" not in self._available_tools():
            return []
        output = self._run_tool(
            "read_folder",
            {"path": str(scope.get("scope_path", ".")), "recursive": True},
        )
        parsed = self._parse_json(output)
        if not isinstance(parsed, list):
            return []

        allowed_exts = {".txt", ".md", ".rst", ".pdf", ".docx", ".html", ".htm"}
        files: list[str] = []
        for item in parsed:
            if self._is_stop_requested():
                break
            if not isinstance(item, dict):
                continue
            if bool(item.get("is_dir", False)):
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            ext = Path(path).suffix.lower()
            if ext and ext not in allowed_exts:
                continue
            if path not in files:
                files.append(path)

        files = files[:400]
        self._log(f"Inventory found {len(files)} files.")
        return files

    def _collect_metadata(self, paths: List[str]) -> List[Dict[str, Any]]:
        if self._is_stop_requested():
            return []
        if "file_metadata" not in self._available_tools() or not paths:
            return []

        results: list[Dict[str, Any]] = []
        batch_size = 50
        for start in range(0, len(paths), batch_size):
            if self._is_stop_requested():
                break
            batch = paths[start : start + batch_size]
            output = self._run_tool("file_metadata", {"paths": batch})
            parsed = self._parse_json(output)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        results.append(item)
        return results

    def _detect_exact_duplicates(self, paths: List[str]) -> List[Dict[str, Any]]:
        if self._is_stop_requested():
            return []
        if "hash_file" not in self._available_tools() or not paths:
            return []

        by_hash: Dict[str, List[str]] = {}
        for path in paths:
            if self._is_stop_requested():
                break
            output = self._run_tool("hash_file", {"path": path})
            parsed = self._parse_json(output)
            if not isinstance(parsed, dict):
                continue
            digest = str(parsed.get("sha256", "")).strip()
            hashed_path = str(parsed.get("path", path)).strip()
            if digest and hashed_path:
                by_hash.setdefault(digest, []).append(hashed_path)

        duplicates: list[Dict[str, Any]] = []
        for digest in sorted(by_hash.keys()):
            dup_paths = sorted(set(by_hash[digest]))
            if len(dup_paths) < 2:
                continue
            duplicates.append(
                {
                    "sha256": digest,
                    "count": len(dup_paths),
                    "paths": dup_paths,
                }
            )
        return duplicates

    def _detect_near_duplicates(
        self,
        paths: List[str],
        scope: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self._is_stop_requested():
            return []
        if "similarity_search" not in self._available_tools() or not paths:
            return []

        candidate_paths = paths[:120]
        args = {
            "paths": candidate_paths,
            "top_k": int(scope.get("top_near_duplicates", 25)),
            "threshold": float(scope.get("near_duplicate_threshold", 0.9)),
        }
        output = self._run_tool("similarity_search", args)
        parsed = self._parse_json(output)
        if not isinstance(parsed, list):
            return []

        results: list[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            left = str(item.get("source_path_a", "")).strip()
            right = str(item.get("source_path_b", "")).strip()
            if not left or not right:
                continue
            try:
                similarity = float(item.get("similarity", 0.0))
            except (TypeError, ValueError):
                similarity = 0.0
            results.append(
                {
                    "source_path_a": left,
                    "source_path_b": right,
                    "similarity": similarity,
                }
            )
        return results

    def _collect_quality_signals(self, topic: str) -> List[Dict[str, Any]]:
        if self._is_stop_requested():
            return []
        if "query_embeddings" not in self._available_tools():
            return []
        query = str(
            self.agent_overrides.get(
                "quality_query",
                f"{topic} TODO draft placeholder incomplete",
            )
        ).strip()
        if not query:
            return []
        output = self._run_tool(
            "query_embeddings",
            {"query": query, "top_k": 10, "max_chars": 500},
        )
        parsed = self._parse_json(output)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _apply_heuristics(
        self,
        metadata: List[Dict[str, Any]],
        quality_signals: List[Dict[str, Any]],
        scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._is_stop_requested():
            return {
                "stale_files": [],
                "tiny_files": [],
                "empty_files": [],
                "quality_hits": [],
            }
        stale_cutoff = datetime.now(timezone.utc) - timedelta(
            days=int(scope.get("stale_days", 365))
        )

        stale_files: list[Dict[str, Any]] = []
        tiny_files: list[Dict[str, Any]] = []
        empty_files: list[Dict[str, Any]] = []
        quality_hits: list[Dict[str, Any]] = []

        for item in metadata:
            if self._is_stop_requested():
                break
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            size = self._coerce_int(item.get("size_bytes"), default=0)
            if size == 0:
                empty_files.append({"path": path, "size_bytes": size})
            if size > 0 and size < 120:
                tiny_files.append({"path": path, "size_bytes": size})

            mtime_raw = str(item.get("mtime_iso", "")).strip()
            if mtime_raw:
                try:
                    mtime = datetime.fromisoformat(mtime_raw)
                    if mtime.tzinfo is None:
                        mtime = mtime.replace(tzinfo=timezone.utc)
                    if mtime < stale_cutoff:
                        stale_files.append({"path": path, "mtime_iso": mtime.isoformat()})
                except ValueError:
                    continue

        for hit in quality_signals:
            if self._is_stop_requested():
                break
            text = str(hit.get("text", "")).lower()
            if not text:
                continue
            if any(marker in text for marker in ("todo", "tbd", "lorem ipsum", "fixme")):
                metadata_obj = hit.get("metadata")
                source_path = ""
                if isinstance(metadata_obj, dict):
                    source_path = str(metadata_obj.get("source_path", "")).strip()
                if not source_path:
                    source_path = str(hit.get("source_path", "")).strip()
                if source_path:
                    quality_hits.append(
                        {
                            "path": source_path,
                            "relevance": float(hit.get("relevance", 0.0) or 0.0),
                        }
                    )

        return {
            "stale_files": self._dedupe_dicts(stale_files, key="path"),
            "tiny_files": self._dedupe_dicts(tiny_files, key="path"),
            "empty_files": self._dedupe_dicts(empty_files, key="path"),
            "quality_hits": self._dedupe_dicts(quality_hits, key="path"),
        }

    def _generate_recommendations(
        self,
        provider: Any,
        topic: str,
        scope: Dict[str, Any],
        inventory: List[str],
        metadata: List[Dict[str, Any]],
        exact_duplicates: List[Dict[str, Any]],
        near_duplicates: List[Dict[str, Any]],
        heuristics: Dict[str, Any],
    ) -> List[str]:
        if self._is_stop_requested():
            return []
        default_recommendations = self._default_recommendations(
            exact_duplicates,
            near_duplicates,
            heuristics,
        )
        prompt = (
            "Produce concise actionable corpus curation recommendations. "
            "Return JSON array of strings."
        )
        context = json.dumps(
            {
                "topic": topic,
                "scope": scope,
                "inventory_count": len(inventory),
                "metadata_count": len(metadata),
                "exact_duplicates": exact_duplicates[:5],
                "near_duplicates": near_duplicates[:5],
                "heuristics": heuristics,
                "defaults": default_recommendations,
            },
            indent=2,
        )
        response = provider.synthesize(prompt, context, mode="insight")
        self._consume_tokens(response)
        parsed = self._parse_json(response)
        if isinstance(parsed, list):
            normalized = [str(item).strip() for item in parsed if str(item).strip()]
            if normalized:
                return normalized[:12]
        return default_recommendations

    def _default_recommendations(
        self,
        exact_duplicates: List[Dict[str, Any]],
        near_duplicates: List[Dict[str, Any]],
        heuristics: Dict[str, Any],
    ) -> List[str]:
        recs: list[str] = []
        if exact_duplicates:
            recs.append(
                "Consolidate exact-duplicate files and retain a single canonical copy per hash group."
            )
        if near_duplicates:
            recs.append(
                "Review high-similarity file pairs and merge overlapping notes where appropriate."
            )
        stale_files = heuristics.get("stale_files") or []
        if stale_files:
            recs.append(
                "Archive or refresh stale files that have not been updated within the configured staleness window."
            )
        tiny_files = heuristics.get("tiny_files") or []
        if tiny_files:
            recs.append(
                "Expand very small files into fuller notes or merge them into related documents."
            )
        empty_files = heuristics.get("empty_files") or []
        if empty_files:
            recs.append("Delete empty files that do not serve as intentional placeholders.")
        quality_hits = heuristics.get("quality_hits") or []
        if quality_hits:
            recs.append(
                "Resolve placeholder markers (e.g., TODO/FIXME/TBD) in flagged files."
            )
        if not recs:
            recs.append("No urgent quality issues detected; continue periodic curation checks.")
        return recs

    def _build_report(
        self,
        topic: str,
        scope: Dict[str, Any],
        inventory: List[str],
        metadata: List[Dict[str, Any]],
        exact_duplicates: List[Dict[str, Any]],
        near_duplicates: List[Dict[str, Any]],
        heuristics: Dict[str, Any],
        recommendations: List[str],
    ) -> str:
        lines = [
            f"# Curation Report: {topic}",
            "",
            "## Scope",
            "",
            f"- Path: `{scope.get('scope_path', '.')}`",
            f"- Stale threshold (days): {int(scope.get('stale_days', 365))}",
            f"- Near-duplicate similarity threshold: {float(scope.get('near_duplicate_threshold', 0.9)):.3f}",
            "",
            "## Summary",
            "",
            f"- Files scanned: {len(inventory)}",
            f"- Files with metadata: {len(metadata)}",
            f"- Exact duplicate groups: {len(exact_duplicates)}",
            f"- Near-duplicate pairs: {len(near_duplicates)}",
            "",
            "## Exact Duplicates",
            "",
        ]
        if not exact_duplicates:
            lines.append("No exact duplicates detected.")
            lines.append("")
        else:
            for group in exact_duplicates[:20]:
                lines.append(f"- Hash `{group['sha256'][:12]}...` ({group['count']} files):")
                for path in group["paths"][:8]:
                    lines.append(f"  - `{path}`")
            lines.append("")

        lines.extend(["## Near Duplicates", ""])
        if not near_duplicates:
            lines.append("No near-duplicate pairs above threshold detected.")
            lines.append("")
        else:
            for pair in near_duplicates[:30]:
                lines.append(
                    f"- {pair['similarity']:.3f}: `{pair['source_path_a']}` <-> `{pair['source_path_b']}`"
                )
            lines.append("")

        stale_files = heuristics.get("stale_files") or []
        tiny_files = heuristics.get("tiny_files") or []
        empty_files = heuristics.get("empty_files") or []
        quality_hits = heuristics.get("quality_hits") or []

        lines.extend(["## Heuristics", ""])
        lines.append(f"- Stale files: {len(stale_files)}")
        for item in stale_files[:20]:
            lines.append(f"  - `{item['path']}` (mtime: {item['mtime_iso']})")
        lines.append(f"- Tiny files (<120 bytes): {len(tiny_files)}")
        for item in tiny_files[:20]:
            lines.append(f"  - `{item['path']}` ({item['size_bytes']} bytes)")
        lines.append(f"- Empty files: {len(empty_files)}")
        for item in empty_files[:20]:
            lines.append(f"  - `{item['path']}`")
        lines.append(f"- Placeholder-quality hits: {len(quality_hits)}")
        for item in quality_hits[:20]:
            lines.append(f"  - `{item['path']}`")
        lines.append("")

        lines.extend(["## Recommendations", ""])
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
        lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _save_report(
        self,
        topic: str,
        report_markdown: str,
        initial_context: AgentContext,
    ) -> Path:
        run_dir = self._resolve_output_dir(topic, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)

        report_path = run_dir / "curation_report.md"
        report_path.write_text(report_markdown, encoding="utf-8")
        self.state.add_artifact(str(report_path))
        self._log(f"Saved curation report to {report_path}")
        return report_path

    def _resolve_output_dir(self, topic: str, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)

        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "curator"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}"

    def _run_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if self._handle_stop_request():
            return ""
        try:
            tool = self.tool_registry.lookup(tool_name)
        except KeyError:
            self._log(f"Skipping unknown tool '{tool_name}'.")
            return ""
        try:
            output = self.sandbox.execute(tool, args)
            self._consume_tokens(output)
            self._log(
                output,
                actor="tool",
                action=tool_name,
                action_arguments=args,
            )
            return output
        except Exception as exc:
            self._log(f"Tool '{tool_name}' failed: {exc}")
            return ""

    def _dedupe_dicts(self, rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        deduped: list[Dict[str, Any]] = []
        for row in rows:
            marker = str(row.get(key, "")).strip()
            if not marker or marker in seen:
                continue
            seen.add(marker)
            deduped.append(row)
        return deduped

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
