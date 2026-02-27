"""
Calendar assistant agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lsm.config.models import AgentConfig, LLMRegistryConfig, RemoteProviderConfig
from lsm.logging import get_logger
from lsm.remote.factory import create_remote_provider
from lsm.remote.providers.communication.models import CalendarEvent

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace

logger = get_logger(__name__)


@dataclass
class CalendarAssistantResult:
    """
    Structured result payload for calendar assistant runs.
    """

    summary_path: Path
    summary_json_path: Path
    log_path: Path


class CalendarAssistantAgent(BaseAgent):
    """
    Summarize calendars, suggest slots, and manage events with approval.
    """

    name = "calendar_assistant"
    tier = "normal"
    description = "Summarize calendars, suggest availability, and manage events with approval."
    tool_allowlist = {"ask_user"}
    risk_posture = "network"

    _CALENDAR_PROVIDER_TYPES = {
        "google_calendar",
        "microsoft_graph_calendar",
        "caldav",
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
        self._bind_interaction_tools()

    def run(self, initial_context: AgentContext) -> Any:
        self._reset_harness()
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"Calendar Assistant: {topic}"

        request = self._parse_request(initial_context)
        provider = self._resolve_provider(request)

        run_dir = self._artifacts_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        action = request.get("action", "summary")
        if action == "suggest":
            payload = self._handle_suggest_flow(provider, request)
        elif action in {"create", "update", "delete"}:
            payload = self._handle_mutation_flow(provider, request)
        else:
            payload = self._handle_summary_flow(provider, request)

        summary_json_path = run_dir / "calendar_summary.json"
        summary_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        summary_path = run_dir / "calendar_summary.md"
        summary_path.write_text(
            self._format_summary_markdown(payload),
            encoding="utf-8",
        )
        self.state.add_artifact(str(summary_json_path))
        self.state.add_artifact(str(summary_path))

        log_path = self._save_log()
        self.last_result = CalendarAssistantResult(
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
        return "Calendar Summary"

    def _parse_request(self, context: AgentContext) -> Dict[str, Any]:
        payload = self._extract_payload(context)
        overrides = self.agent_overrides if isinstance(self.agent_overrides, dict) else {}
        action = str(payload.get("action") or payload.get("mode") or "summary").strip().lower()
        query = str(payload.get("query") or "").strip()
        max_results = payload.get("max_results")
        max_results = int(max_results) if max_results is not None else None

        now = self._resolve_now(payload, overrides)
        time_min, time_max = self._resolve_window(payload, overrides, now)

        request = {
            "action": action,
            "query": query,
            "max_results": max_results,
            "time_min": time_min,
            "time_max": time_max,
            "event": payload.get("event") or {},
            "event_id": payload.get("event_id"),
            "updates": payload.get("updates") or {},
            "duration_minutes": payload.get("duration_minutes") or payload.get("duration"),
            "window_start": payload.get("window_start") or payload.get("start"),
            "window_end": payload.get("window_end") or payload.get("end"),
            "workday_start": payload.get("workday_start") or overrides.get("workday_start"),
            "workday_end": payload.get("workday_end") or overrides.get("workday_end"),
            "max_suggestions": payload.get("max_suggestions") or overrides.get("max_suggestions"),
            "provider": payload.get("provider") or overrides.get("provider") or overrides.get("provider_name"),
            "provider_config": payload.get("provider_config") or overrides.get("provider_config"),
            "provider_instance": overrides.get("provider_instance"),
            "raw_payload": payload,
        }
        return request

    def _handle_summary_flow(self, provider: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        events = self._fetch_events(provider, request)
        by_day = self._group_events_by_day(events)
        by_week = self._group_events_by_week(events)
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "provider": getattr(provider, "name", None) or provider.__class__.__name__,
            "window": {
                "start": request["time_min"].isoformat() if request.get("time_min") else None,
                "end": request["time_max"].isoformat() if request.get("time_max") else None,
            },
            "total_events": len(events),
            "events_by_day": by_day,
            "events_by_week": by_week,
        }

    def _handle_suggest_flow(self, provider: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        duration_minutes = int(request.get("duration_minutes") or 30)
        window_start = self._parse_datetime(request.get("window_start"))
        window_end = self._parse_datetime(request.get("window_end"))
        if window_start is None or window_end is None:
            now = datetime.utcnow()
            window_start = now
            window_end = now + timedelta(days=7)
        workday_start = self._parse_time(request.get("workday_start")) or time(9, 0)
        workday_end = self._parse_time(request.get("workday_end")) or time(17, 0)
        max_suggestions = int(request.get("max_suggestions") or 5)

        events = provider.list_events(
            query=request.get("query"),
            time_min=window_start,
            time_max=window_end,
            max_results=request.get("max_results") or 50,
        )
        suggestions = self._find_available_slots(
            events,
            window_start,
            window_end,
            duration_minutes=duration_minutes,
            workday_start=workday_start,
            workday_end=workday_end,
            max_suggestions=max_suggestions,
        )
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "provider": getattr(provider, "name", None) or provider.__class__.__name__,
            "window": {
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
            },
            "duration_minutes": duration_minutes,
            "suggestions": [
                {"start": slot[0].isoformat(), "end": slot[1].isoformat()}
                for slot in suggestions
            ],
        }

    def _handle_mutation_flow(self, provider: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        approval_prompt = f"Approve calendar {action} operation?"
        approved = self._request_approval(approval_prompt, context="Calendar assistant mutation")
        status = "denied"
        event_id: Optional[str] = None
        if approved:
            if action == "create":
                event = self._event_from_payload(request.get("event") or {})
                created = provider.create_event(event)
                event_id = getattr(created, "event_id", None)
            elif action == "update":
                event_id = str(request.get("event_id") or "").strip() or None
                if not event_id:
                    raise ValueError("event_id is required for update")
                provider.update_event(event_id, request.get("updates") or {})
            elif action == "delete":
                event_id = str(request.get("event_id") or "").strip() or None
                if not event_id:
                    raise ValueError("event_id is required for delete")
                provider.delete_event(event_id)
            status = "completed"
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "provider": getattr(provider, "name", None) or provider.__class__.__name__,
            "mutation": {
                "action": action,
                "event_id": event_id,
                "status": status,
            },
        }

    def _fetch_events(self, provider: Any, request: Dict[str, Any]) -> List[CalendarEvent]:
        max_results = request.get("max_results")
        if max_results is None:
            max_results = int(self.agent_overrides.get("max_results", 25))
        try:
            return list(
                provider.list_events(
                    query=request.get("query") or None,
                    time_min=request.get("time_min"),
                    time_max=request.get("time_max"),
                    max_results=max_results,
                )
                or []
            )
        except Exception as exc:
            self._log(f"Calendar provider failed: {exc}")
            return []

    def _group_events_by_day(self, events: Sequence[CalendarEvent]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[CalendarEvent]] = {}
        for event in events:
            key = self._event_day_key(event)
            grouped.setdefault(key, []).append(event)
        summary: List[Dict[str, Any]] = []
        for day in sorted(grouped.keys()):
            items = sorted(grouped[day], key=lambda item: item.start or datetime.min)
            summary.append(
                {
                    "date": day,
                    "events": [self._event_summary(item) for item in items],
                }
            )
        return summary

    def _group_events_by_week(self, events: Sequence[CalendarEvent]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[CalendarEvent]] = {}
        for event in events:
            key = self._event_week_key(event)
            grouped.setdefault(key, []).append(event)
        summary: List[Dict[str, Any]] = []
        for week in sorted(grouped.keys()):
            items = sorted(grouped[week], key=lambda item: item.start or datetime.min)
            summary.append(
                {
                    "week": week,
                    "events": [self._event_summary(item) for item in items],
                }
            )
        return summary

    def _event_day_key(self, event: CalendarEvent) -> str:
        moment = event.start or event.end
        if moment is None:
            return "unknown"
        return moment.date().isoformat()

    def _event_week_key(self, event: CalendarEvent) -> str:
        moment = event.start or event.end
        if moment is None:
            return "unknown"
        year, week, _ = moment.isocalendar()
        return f"{year}-W{week:02d}"

    def _event_summary(self, event: CalendarEvent) -> Dict[str, Any]:
        return {
            "event_id": event.event_id,
            "title": event.title,
            "start": event.start.isoformat() if event.start else None,
            "end": event.end.isoformat() if event.end else None,
            "location": event.location,
        }

    def _find_available_slots(
        self,
        events: Sequence[CalendarEvent],
        window_start: datetime,
        window_end: datetime,
        *,
        duration_minutes: int,
        workday_start: time,
        workday_end: time,
        max_suggestions: int,
    ) -> List[tuple[datetime, datetime]]:
        slots: List[tuple[datetime, datetime]] = []
        if duration_minutes <= 0:
            return slots
        duration = timedelta(minutes=duration_minutes)
        tzinfo = window_start.tzinfo
        current_date = window_start.date()
        end_date = window_end.date()

        while current_date <= end_date and len(slots) < max_suggestions:
            day_start = datetime.combine(current_date, workday_start, tzinfo=tzinfo)
            day_end = datetime.combine(current_date, workday_end, tzinfo=tzinfo)
            if day_end <= window_start or day_start >= window_end:
                current_date += timedelta(days=1)
                continue
            day_start = max(day_start, window_start)
            day_end = min(day_end, window_end)

            day_events = [
                (event.start, event.end)
                for event in events
                if event.start and event.end and event.start.date() == current_date
            ]
            day_events.sort(key=lambda pair: pair[0])
            cursor = day_start
            for start, end in day_events:
                if start is None or end is None:
                    continue
                if start - cursor >= duration:
                    slots.append((cursor, cursor + duration))
                    if len(slots) >= max_suggestions:
                        return slots
                cursor = max(cursor, end)
            if day_end - cursor >= duration:
                slots.append((cursor, cursor + duration))
            current_date += timedelta(days=1)

        return slots[:max_suggestions]

    def _event_from_payload(self, payload: Dict[str, Any]) -> CalendarEvent:
        return CalendarEvent(
            event_id=str(payload.get("event_id") or ""),
            title=str(payload.get("title") or payload.get("summary") or "").strip(),
            start=self._parse_datetime(payload.get("start")),
            end=self._parse_datetime(payload.get("end")),
            location=payload.get("location"),
            description=payload.get("description"),
            attendees=list(payload.get("attendees") or []),
            status=payload.get("status"),
        )

    def _format_summary_markdown(self, payload: Dict[str, Any]) -> str:
        lines = ["# Calendar Summary", ""]
        lines.append(f"- Generated at: {payload.get('generated_at', '-')}")
        lines.append(f"- Provider: {payload.get('provider', '-')}")
        if "window" in payload:
            window = payload.get("window") or {}
            lines.append(f"- Window start: {window.get('start', '-')}")
            lines.append(f"- Window end: {window.get('end', '-')}")
        lines.append("")

        if "events_by_day" in payload:
            lines.append("## By Day")
            for entry in payload.get("events_by_day") or []:
                date = entry.get("date")
                events = entry.get("events") or []
                lines.append("")
                lines.append(f"### {date}")
                if not events:
                    lines.append("- None")
                else:
                    for event in events:
                        title = event.get("title") or "(untitled)"
                        start = event.get("start") or "unknown"
                        lines.append(f"- {title} — {start}")

            lines.append("")
            lines.append("## By Week")
            for entry in payload.get("events_by_week") or []:
                week = entry.get("week")
                events = entry.get("events") or []
                lines.append("")
                lines.append(f"### {week}")
                if not events:
                    lines.append("- None")
                else:
                    for event in events:
                        title = event.get("title") or "(untitled)"
                        start = event.get("start") or "unknown"
                        lines.append(f"- {title} — {start}")
        elif "suggestions" in payload:
            lines.append("## Suggested Slots")
            suggestions = payload.get("suggestions") or []
            if not suggestions:
                lines.append("- None")
            else:
                for slot in suggestions:
                    lines.append(
                        f"- {slot.get('start', '-') } to {slot.get('end', '-') }"
                    )
        else:
            mutation = payload.get("mutation") or {}
            lines.append("## Mutation Status")
            lines.append(f"- Action: {mutation.get('action', '-')}")
            lines.append(f"- Event ID: {mutation.get('event_id', '-')}")
            lines.append(f"- Status: {mutation.get('status', '-')}")

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
        start_raw = payload.get("start") or payload.get("time_min") or payload.get("window_start")
        end_raw = payload.get("end") or payload.get("time_max") or payload.get("window_end")
        start_dt = self._parse_datetime(start_raw)
        end_dt = self._parse_datetime(end_raw)
        if start_dt or end_dt:
            return start_dt, end_dt
        days = payload.get("window_days") or overrides.get("default_window_days") or 7
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 7
        return now, now + timedelta(days=days)

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
                if provider.type in self._CALENDAR_PROVIDER_TYPES
            ]
            if provider_name:
                for provider in candidates:
                    if provider.name.lower() == str(provider_name).lower():
                        return self._build_provider_from_config(provider, lsm_config)
            if candidates:
                return self._build_provider_from_config(candidates[0], lsm_config)

        raise ValueError("Calendar provider configuration not available")

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

    def _parse_time(self, value: Any) -> Optional[time]:
        if not value:
            return None
        if isinstance(value, time):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            parts = text.split(":")
            if len(parts) < 2:
                return None
            return time(int(parts[0]), int(parts[1]))
        except Exception:
            return None

