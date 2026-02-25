from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.agents.assistants.calendar_assistant import CalendarAssistantAgent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw
from lsm.remote.providers.communication.models import CalendarEvent


class AskUserStub(BaseTool):
    name = "ask_user"
    description = "Stub approval tool."
    input_schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}, "context": {"type": "string"}},
        "required": ["prompt"],
    }

    def __init__(self, response: str = "yes") -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def execute(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.response


@dataclass
class StubCalendarProvider:
    events: List[CalendarEvent]

    def __post_init__(self) -> None:
        self.created: List[CalendarEvent] = []
        self.updated: List[str] = []
        self.deleted: List[str] = []

    @property
    def name(self) -> str:
        return "stub_calendar"

    def list_events(
        self,
        *,
        query: Optional[str] = None,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10,
    ) -> List[CalendarEvent]:
        _ = query, time_min, time_max
        return self.events[:max_results]

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        if not event.event_id:
            event.event_id = f"evt-{len(self.created)+1}"
        self.created.append(event)
        return event

    def update_event(self, event_id: str, updates: Dict[str, Any]) -> CalendarEvent:
        self.updated.append(event_id)
        return CalendarEvent(event_id=event_id, title=str(updates.get("title") or ""), start=None, end=None)

    def delete_event(self, event_id: str) -> None:
        self.deleted.append(event_id)


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 8000,
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_calendar_assistant_summarizes_events(tmp_path: Path) -> None:
    base = datetime(2024, 2, 1, 9, 0, 0)
    events = [
        CalendarEvent(event_id="e1", title="Standup", start=base, end=base + timedelta(minutes=30)),
        CalendarEvent(
            event_id="e2",
            title="Review",
            start=base + timedelta(hours=2),
            end=base + timedelta(hours=3),
        ),
        CalendarEvent(
            event_id="e3",
            title="Planning",
            start=base + timedelta(days=1),
            end=base + timedelta(days=1, hours=1),
        ),
    ]
    provider = StubCalendarProvider(events=events)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(AskUserStub())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = CalendarAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        agent_overrides={"provider_instance": provider, "now": base.isoformat()},
    )

    agent.run(AgentContext(messages=[{"role": "user", "content": "Summary"}]))
    payload = json.loads(agent.last_result.summary_json_path.read_text(encoding="utf-8"))
    assert payload["total_events"] == 3
    by_day = payload["events_by_day"]
    assert len(by_day) == 2
    assert by_day[0]["events"]


def test_calendar_assistant_suggests_slots(tmp_path: Path) -> None:
    base = datetime(2024, 2, 2, 9, 0, 0)
    events = [
        CalendarEvent(
            event_id="e1",
            title="Busy",
            start=base + timedelta(hours=1),
            end=base + timedelta(hours=2),
        )
    ]
    provider = StubCalendarProvider(events=events)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(AskUserStub())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = CalendarAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        agent_overrides={"provider_instance": provider},
    )

    payload = {
        "action": "suggest",
        "duration_minutes": 30,
        "window_start": base.isoformat(),
        "window_end": (base + timedelta(hours=3)).isoformat(),
        "workday_start": "09:00",
        "workday_end": "12:00",
        "max_suggestions": 2,
    }
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    summary = json.loads(agent.last_result.summary_json_path.read_text(encoding="utf-8"))
    suggestions = summary["suggestions"]
    assert suggestions
    assert suggestions[0]["start"].startswith("2024-02-02T09:00")


def test_calendar_assistant_requires_approval_for_mutation(tmp_path: Path) -> None:
    provider = StubCalendarProvider(events=[])
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")

    registry = ToolRegistry()
    registry.register(AskUserStub(response="no"))
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = CalendarAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        agent_overrides={"provider_instance": provider},
    )
    payload = {
        "action": "create",
        "event": {
            "title": "New Event",
            "start": "2024-02-03T10:00:00",
            "end": "2024-02-03T11:00:00",
        },
    }
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    assert not provider.created

    registry = ToolRegistry()
    registry.register(AskUserStub(response="yes"))
    sandbox = ToolSandbox(config.agents.sandbox)
    provider = StubCalendarProvider(events=[])
    agent = CalendarAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        agent_overrides={"provider_instance": provider},
    )
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    assert provider.created
