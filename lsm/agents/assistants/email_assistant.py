"""
Email assistant agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from lsm.config.models import AgentConfig, LLMRegistryConfig, LSMConfig, RemoteProviderConfig
from lsm.logging import get_logger
from lsm.remote.factory import create_remote_provider
from lsm.remote.providers.communication.models import EmailDraft, EmailMessage

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox

logger = get_logger(__name__)


@dataclass
class EmailAssistantResult:
    """
    Structured result payload for email assistant runs.
    """

    summary_path: Path
    summary_json_path: Path
    log_path: Path


class EmailAssistantAgent(BaseAgent):
    """
    Summarize inbox activity and draft emails with approval gating.
    """

    name = "email_assistant"
    tier = "normal"
    description = "Summarize emails, extract tasks, and draft replies with approval."
    tool_allowlist = {"ask_user"}
    risk_posture = "network"

    _ACTION_KEYWORDS = (
        "action",
        "please",
        "need",
        "review",
        "approve",
        "reply",
        "respond",
        "follow up",
        "follow-up",
        "schedule",
        "confirm",
        "request",
        "due",
        "asap",
        "urgent",
    )

    _IMPORTANCE_KEYWORDS = ("urgent", "asap", "important", "action required", "review")

    _EMAIL_PROVIDER_TYPES = {
        "gmail",
        "microsoft_graph_mail",
        "imap",
    }

    def __init__(
        self,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
        agent_overrides: Optional[Dict[str, Any]] = None,
        lsm_config: Optional[LSMConfig] = None,
    ) -> None:
        super().__init__(name=self.name, description=self.description)
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.agent_config = agent_config
        self.agent_overrides = agent_overrides or {}
        self.lsm_config = lsm_config
        self._bind_interaction_tools()

    def run(self, initial_context: AgentContext) -> Any:
        self._reset_harness()
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        self._workspace_root()
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"Email Assistant: {topic}"

        request = self._parse_request(initial_context)
        provider = self._resolve_provider(request)

        run_dir = self._artifacts_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        action = request.get("action", "summary")
        if action in {"draft", "compose", "reply", "send"}:
            payload = self._handle_draft_flow(provider, request)
        else:
            payload = self._handle_summary_flow(provider, request)

        summary_json_path = run_dir / "email_summary.json"
        summary_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        summary_path = run_dir / "email_summary.md"
        summary_path.write_text(
            self._format_summary_markdown(payload),
            encoding="utf-8",
        )
        self.state.add_artifact(str(summary_json_path))
        self.state.add_artifact(str(summary_path))

        log_path = self._save_log()
        self.last_result = EmailAssistantResult(
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
        return "Email Summary"

    def _parse_request(self, context: AgentContext) -> Dict[str, Any]:
        payload: Dict[str, Any] = self._extract_payload(context)
        overrides = self.agent_overrides if isinstance(self.agent_overrides, dict) else {}
        action = str(payload.get("action") or payload.get("mode") or "summary").strip().lower()
        filters_value = payload.get("filters")
        if isinstance(filters_value, dict):
            filters: Dict[str, Any] = {
                str(key): value for key, value in filters_value.items()
            }
        else:
            filters = {}
        query = payload.get("query") or filters.get("query") or ""
        from_address = payload.get("from") or filters.get("from") or filters.get("from_address")
        to_address = payload.get("to") or filters.get("to") or filters.get("to_address")
        unread_only = payload.get("unread_only")
        if unread_only is None:
            unread_only = filters.get("unread_only", False)
        folder = payload.get("folder") or filters.get("folder")
        max_results = payload.get("max_results") or filters.get("max_results")
        max_results = int(max_results) if max_results is not None else None

        now = self._resolve_now(payload, overrides)
        after, before = self._resolve_window(payload, overrides, now)

        request = {
            "action": action,
            "query": str(query or "").strip(),
            "from_address": str(from_address or "").strip() or None,
            "to_address": str(to_address or "").strip() or None,
            "unread_only": bool(unread_only),
            "folder": str(folder or "").strip() or None,
            "max_results": max_results,
            "after": after,
            "before": before,
            "draft": payload.get("draft") or {},
            "send": bool(payload.get("send", False)),
            "provider": payload.get("provider") or overrides.get("provider") or overrides.get("provider_name"),
            "provider_config": payload.get("provider_config") or overrides.get("provider_config"),
            "provider_instance": overrides.get("provider_instance"),
            "priority_senders": payload.get("priority_senders") or overrides.get("priority_senders") or [],
            "raw_payload": payload,
        }
        return request

    def _handle_summary_flow(self, provider: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._fetch_messages(provider, request)
        importance = self._bucket_by_importance(messages, request)
        topics = self._group_by_topic(messages)
        tasks = self._extract_tasks(messages)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "provider": getattr(provider, "name", None) or provider.__class__.__name__,
            "window": {
                "after": request["after"].isoformat() if request.get("after") else None,
                "before": request["before"].isoformat() if request.get("before") else None,
            },
            "filters": {
                "query": request.get("query"),
                "from": request.get("from_address"),
                "to": request.get("to_address"),
                "unread_only": request.get("unread_only"),
                "folder": request.get("folder"),
            },
            "total_messages": len(messages),
            "importance": importance,
            "topics": topics,
            "tasks": tasks,
        }

    def _handle_draft_flow(self, provider: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        draft_payload = request.get("draft") or {}
        recipients = draft_payload.get("recipients") or draft_payload.get("to") or []
        if isinstance(recipients, str):
            recipients = [item.strip() for item in recipients.split(",") if item.strip()]
        subject = str(draft_payload.get("subject") or "").strip()
        body = str(draft_payload.get("body") or "").strip()
        thread_id = draft_payload.get("thread_id")

        draft: Optional[EmailDraft] = None
        if recipients and subject and body:
            draft = provider.create_draft(
                recipients=list(recipients),
                subject=subject,
                body=body,
                thread_id=thread_id,
            )

        send_requested = bool(request.get("send")) or request.get("action") == "send"
        send_status = "not_requested"
        if send_requested:
            approved = self._request_approval(
                f"Send email draft '{subject}' to {', '.join(recipients) or 'recipients'}?",
                context="Email assistant send operation",
            )
            if not approved:
                send_status = "denied"
            elif draft is not None:
                self._send_draft(provider, draft, recipients, subject, body)
                send_status = "sent"
            else:
                send_status = "no_draft"

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "provider": getattr(provider, "name", None) or provider.__class__.__name__,
            "action": request.get("action"),
            "draft": {
                "draft_id": getattr(draft, "draft_id", None),
                "subject": subject,
                "recipients": recipients,
                "body_preview": body[:200],
                "thread_id": thread_id,
            },
            "send_status": send_status,
        }

    def _fetch_messages(self, provider: Any, request: Dict[str, Any]) -> List[EmailMessage]:
        max_results = request.get("max_results")
        if max_results is None:
            max_results = int(self.agent_overrides.get("max_results", 10))
        try:
            messages = provider.search_messages(
                request.get("query", ""),
                max_results=max_results,
                unread_only=bool(request.get("unread_only")),
                from_address=request.get("from_address"),
                to_address=request.get("to_address"),
                after=request.get("after"),
                before=request.get("before"),
                folder=request.get("folder"),
            )
        except Exception as exc:
            self._log(f"Email provider search failed: {exc}")
            return []
        return list(messages or [])

    def _bucket_by_importance(
        self,
        messages: Sequence[EmailMessage],
        request: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        priority_senders = {
            str(item).strip().lower()
            for item in (request.get("priority_senders") or [])
            if str(item).strip()
        }
        buckets = {"high": [], "normal": [], "low": []}
        for message in messages:
            score = 0
            if message.is_unread:
                score += 2
            subject_snippet = f"{message.subject} {message.snippet}".lower()
            if any(keyword in subject_snippet for keyword in self._IMPORTANCE_KEYWORDS):
                score += 1
            if message.sender and message.sender.strip().lower() in priority_senders:
                score += 1
            if score >= 3:
                bucket = "high"
            elif score >= 1:
                bucket = "normal"
            else:
                bucket = "low"
            buckets[bucket].append(self._message_summary(message))
        return buckets

    def _group_by_topic(self, messages: Sequence[EmailMessage]) -> List[Dict[str, Any]]:
        topics: Dict[str, List[str]] = {}
        for message in messages:
            subject = message.subject or "(no subject)"
            topic = self._normalize_subject(subject)
            topics.setdefault(topic, []).append(message.message_id)
        summary = []
        for topic, message_ids in sorted(topics.items()):
            summary.append({"topic": topic, "count": len(message_ids), "message_ids": message_ids})
        return summary

    def _extract_tasks(self, messages: Sequence[EmailMessage]) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for message in messages:
            subject_snippet = f"{message.subject} {message.snippet}".lower()
            if any(keyword in subject_snippet for keyword in self._ACTION_KEYWORDS):
                tasks.append(
                    {
                        "message_id": message.message_id,
                        "subject": message.subject,
                        "from": message.sender,
                        "received_at": message.received_at.isoformat()
                        if message.received_at
                        else None,
                        "reason": "action_keyword",
                    }
                )
        return tasks

    def _message_summary(self, message: EmailMessage) -> Dict[str, Any]:
        return {
            "message_id": message.message_id,
            "subject": message.subject,
            "from": message.sender,
            "received_at": message.received_at.isoformat() if message.received_at else None,
            "snippet": message.snippet,
            "unread": message.is_unread,
        }

    def _normalize_subject(self, subject: str) -> str:
        normalized = str(subject or "").strip()
        lowered = normalized.lower()
        for prefix in ("re:", "fwd:", "fw:"):
            if lowered.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                lowered = normalized.lower()
        return normalized or "(no subject)"

    def _format_summary_markdown(self, payload: Dict[str, Any]) -> str:
        lines = ["# Email Summary", ""]
        window = payload.get("window") or {}
        lines.append(f"- Generated at: {payload.get('generated_at', '-')}")
        lines.append(f"- Provider: {payload.get('provider', '-')}")
        lines.append(f"- Window start: {window.get('after', '-')}")
        lines.append(f"- Window end: {window.get('before', '-')}")
        lines.append("")

        if "importance" in payload:
            lines.append("## By Importance")
            for label in ("high", "normal", "low"):
                lines.append("")
                lines.append(f"### {label.title()}")
                messages = payload.get("importance", {}).get(label, []) or []
                if not messages:
                    lines.append("- None")
                    continue
                for msg in messages:
                    subject = msg.get("subject") or "(no subject)"
                    sender = msg.get("from") or "unknown"
                    received = msg.get("received_at") or "unknown"
                    unread = "unread" if msg.get("unread") else "read"
                    lines.append(f"- [{unread}] {subject} — {sender} — {received}")

            lines.append("")
            lines.append("## Topics")
            topics = payload.get("topics") or []
            if not topics:
                lines.append("- None")
            else:
                for topic in topics:
                    lines.append(
                        f"- {topic.get('topic')}: {topic.get('count', 0)} messages"
                    )

            lines.append("")
            lines.append("## Action Items")
            tasks = payload.get("tasks") or []
            if not tasks:
                lines.append("- None")
            else:
                for task in tasks:
                    subject = task.get("subject") or "(no subject)"
                    sender = task.get("from") or "unknown"
                    lines.append(f"- Follow up: {subject} — {sender}")
        else:
            lines.append("## Draft Status")
            draft = payload.get("draft") or {}
            lines.append(f"- Draft ID: {draft.get('draft_id', '-')}")
            lines.append(f"- Subject: {draft.get('subject', '-')}")
            recipients = draft.get("recipients") or []
            lines.append(f"- Recipients: {', '.join(recipients) if recipients else '-'}")
            lines.append(f"- Send status: {payload.get('send_status', '-')}")

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
        start_raw = payload.get("start") or payload.get("after") or payload.get("time_min")
        end_raw = payload.get("end") or payload.get("before") or payload.get("time_max")
        start_dt = self._parse_datetime(start_raw)
        end_dt = self._parse_datetime(end_raw)
        if start_dt or end_dt:
            return start_dt, end_dt

        window = payload.get("time_window") or payload.get("window") or overrides.get("time_window")
        hours = payload.get("last_hours") or overrides.get("default_window_hours")
        if hours is not None:
            try:
                hours = float(hours)
            except (TypeError, ValueError):
                hours = None
        if not hours:
            hours = 24.0
        if window:
            normalized = str(window).lower().replace(" ", "")
            if normalized in {"1h", "1hour", "lasthour", "past1hour"}:
                hours = 1.0
            elif normalized in {"24h", "24hour", "lastday", "past24hours", "1day"}:
                hours = 24.0
        return now - timedelta(hours=float(hours)), now

    def _resolve_provider(self, request: Dict[str, Any]) -> Any:
        if request.get("provider_instance") is not None:
            return request["provider_instance"]
        provider_config = request.get("provider_config")
        if isinstance(provider_config, dict):
            return create_remote_provider(provider_config.get("type", ""), provider_config)

        lsm_config = self._resolve_lsm_config()
        if lsm_config is not None:
            provider_name = request.get("provider")
            candidates = [
                provider
                for provider in (lsm_config.remote_providers or [])
                if provider.type in self._EMAIL_PROVIDER_TYPES
            ]
            if provider_name:
                for provider in candidates:
                    if provider.name.lower() == str(provider_name).lower():
                        return self._build_provider_from_config(provider, lsm_config)
            if candidates:
                return self._build_provider_from_config(candidates[0], lsm_config)

        raise ValueError("Email provider configuration not available")

    def _resolve_lsm_config(self) -> Optional[LSMConfig]:
        return self.lsm_config

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
        if provider_cfg.oauth is not None:
            config["oauth"] = {
                "client_id": provider_cfg.oauth.client_id,
                "client_secret": provider_cfg.oauth.client_secret,
                "scopes": list(provider_cfg.oauth.scopes or []),
                "redirect_uri": provider_cfg.oauth.redirect_uri,
                "refresh_buffer_seconds": provider_cfg.oauth.refresh_buffer_seconds,
            }
        if provider_cfg.extra:
            config.update(provider_cfg.extra)
        if getattr(lsm_config, "global_folder", None) is not None:
            config["global_folder"] = str(lsm_config.global_folder)
        return create_remote_provider(provider_cfg.type, config)

    def _send_draft(
        self,
        provider: Any,
        draft: EmailDraft,
        recipients: Sequence[str],
        subject: str,
        body: str,
    ) -> None:
        if hasattr(provider, "send_draft"):
            provider.send_draft(draft.draft_id)
        elif hasattr(provider, "send_message"):
            provider.send_message(list(recipients), subject, body)
        else:
            raise ValueError("Provider does not support sending drafts")

    def _request_approval(self, prompt: str, context: str) -> bool:
        try:
            result = self._run_phase(
                direct_tool_calls=[
                    {"name": "ask_user", "arguments": {"prompt": prompt, "context": context}}
                ]
            )
            for call in result.tool_calls:
                if call.get("name") == "ask_user" and "result" in call:
                    return self._is_affirmative(call["result"])
        except Exception as exc:
            self._log(f"Approval request failed: {exc}")
        return False

    @staticmethod
    def _is_affirmative(response: str) -> bool:
        text = str(response or "").strip().lower()
        if not text:
            return False
        if "continue" in text:
            return True
        return text.startswith("y") or "approve" in text or "yes" in text

    def _bind_interaction_tools(self) -> None:
        try:
            tool = self.tool_registry.lookup("ask_user")
        except KeyError:
            return
        binder = getattr(tool, "bind_harness", None)
        if not callable(binder):
            return

        class _Harness:
            def __init__(self, agent_config: AgentConfig, interaction_channel: Any) -> None:
                self.agent_config = agent_config
                self.interaction_channel = interaction_channel

            def request_interaction(self, request):
                if self.interaction_channel is None:
                    raise RuntimeError("Interaction channel is not configured")
                return self.interaction_channel.post_request(request)

        binder(_Harness(self.agent_config, self.sandbox.interaction_channel))

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
            return None
