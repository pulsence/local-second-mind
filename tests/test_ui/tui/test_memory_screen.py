from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from lsm.ui.tui.screens.agents import AgentsScreen


class _Input:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id

    def focus(self) -> None:
        return None


class _Select:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.options = []

    def set_options(self, options) -> None:
        self.options = options


class _Static:
    def __init__(self) -> None:
        self.last = ""

    def update(self, message: str) -> None:
        self.last = message


class _RichLog:
    def __init__(self) -> None:
        self.lines = []

    def write(self, message: str) -> None:
        self.lines.append(message)


class _Manager:
    def __init__(self) -> None:
        self.promoted = None
        self.rejected = None
        self.ttl = None

    def get_memory_candidates(self, app, status="pending", limit=200):
        _ = app, status, limit
        memory = SimpleNamespace(
            id="mem-1",
            key="preferred_mode",
            type="task_state",
            scope="agent",
            confidence=0.8,
            tags=["workflow"],
            created_at=datetime.now(timezone.utc),
            last_used_at=datetime.now(timezone.utc),
            expires_at=None,
            source_run_id="run-1",
            value={"mode": "grounded"},
        )
        candidate = SimpleNamespace(
            id="cand-1",
            memory=memory,
            rationale="Often selected",
            status="pending",
        )
        return [candidate]

    def promote_memory_candidate(self, app, candidate_id: str) -> str:
        _ = app
        self.promoted = candidate_id
        return f"Promoted memory candidate '{candidate_id}'.\n"

    def reject_memory_candidate(self, app, candidate_id: str) -> str:
        _ = app
        self.rejected = candidate_id
        return f"Rejected memory candidate '{candidate_id}'.\n"

    def edit_memory_candidate_ttl(self, app, candidate_id: str, ttl_days: int) -> str:
        _ = app
        self.ttl = (candidate_id, ttl_days)
        return (
            f"Updated TTL for memory candidate '{candidate_id}' "
            f"to {ttl_days} day(s).\n"
        )


class _TestableAgentsScreen(AgentsScreen):
    def __init__(self, app) -> None:
        super().__init__()
        self._test_app = app
        self.widgets = {}

    @property
    def app(self):  # type: ignore[override]
        return self._test_app

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        if selector in self.widgets:
            return self.widgets[selector]
        raise KeyError(selector)

    def call_after_refresh(self, fn):  # type: ignore[override]
        fn()


def _screen():
    app = SimpleNamespace(current_context="agents", config=SimpleNamespace())
    screen = _TestableAgentsScreen(app)
    screen.widgets["#agents-select"] = _Select("", "agents-select")
    screen.widgets["#agents-topic-input"] = _Input("", "agents-topic-input")
    screen.widgets["#agents-status-output"] = _Static()
    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()
    return screen


def test_memory_panel_refresh_approve_reject_and_ttl(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    memory_select = screen.widgets["#agents-memory-select"]
    assert memory_select.value == "cand-1"
    assert len(memory_select.options) == 1

    screen._approve_memory_candidate()
    assert manager.promoted == "cand-1"
    assert "Promoted memory candidate" in screen.widgets["#agents-memory-output"].last

    screen._reject_memory_candidate()
    assert manager.rejected == "cand-1"
    assert "Rejected memory candidate" in screen.widgets["#agents-memory-output"].last

    screen.widgets["#agents-memory-ttl-input"].value = "invalid"
    screen._edit_memory_candidate_ttl()
    assert "TTL days must be an integer." in screen.widgets["#agents-memory-output"].last

    screen.widgets["#agents-memory-ttl-input"].value = "14"
    screen._edit_memory_candidate_ttl()
    assert manager.ttl == ("cand-1", 14)
    assert "Updated TTL for memory candidate" in screen.widgets["#agents-memory-output"].last


def test_memory_panel_refresh_handles_error(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()

    def _boom(app, status="pending", limit=200):
        _ = app, status, limit
        raise RuntimeError("Agent memory is disabled.")

    manager.get_memory_candidates = _boom  # type: ignore[assignment]
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    assert "Agent memory is disabled." in screen.widgets["#agents-memory-output"].last
