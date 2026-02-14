from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from lsm.ui.tui.screens.agents import AgentsScreen


class _Input:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _Select:
    def __init__(self, value: str = "", widget_id: str = "agents-select") -> None:
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
        self.is_vertical_scroll_end = True
        self.scroll_end_calls = []

    def write(self, message: str, scroll_end=None) -> None:
        self.lines.append(message)
        self.scroll_end_calls.append(scroll_end)


class _Manager:
    def __init__(self) -> None:
        self.started = None
        self.promoted = None
        self.rejected = None
        self.ttl_updated = None

    def start(self, app, name: str, topic: str) -> str:
        self.started = (name, topic)
        return f"Started agent '{name}' with topic: {topic}\n"

    def status(self) -> str:
        return "Agent: research\nStatus: running\n"

    def pause(self) -> str:
        return "Paused agent 'research'.\n"

    def resume(self) -> str:
        return "Resumed agent 'research'.\n"

    def stop(self) -> str:
        return "Stop requested for agent 'research'.\n"

    def log(self) -> str:
        return "Agent log line\n"

    def get_memory_candidates(self, app, status="pending", limit=200):
        _ = app, status, limit
        memory = SimpleNamespace(
            id="mem-1",
            key="writing_style",
            type="task_state",
            scope="agent",
            confidence=0.9,
            tags=["style"],
            created_at=datetime.now(timezone.utc),
            last_used_at=datetime.now(timezone.utc),
            expires_at=None,
            source_run_id="run-1",
            value={"tone": "concise"},
        )
        return [
            SimpleNamespace(
                id="cand-1",
                memory=memory,
                status="pending",
                rationale="Common preference",
            )
        ]

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
        self.ttl_updated = (candidate_id, ttl_days)
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


def _screen(context: str = "agents"):
    app = SimpleNamespace(current_context=context, config=SimpleNamespace())
    screen = _TestableAgentsScreen(app)
    screen.widgets["#agents-select"] = _Select("")
    screen.widgets["#agents-topic-input"] = _Input("", "agents-topic-input")
    screen.widgets["#agents-status-output"] = _Static()
    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()
    return screen


def test_on_mount_populates_options_and_focus(monkeypatch) -> None:
    screen = _screen()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: _Manager())
    monkeypatch.setattr("lsm.ui.tui.screens.agents.AgentRegistry", lambda: SimpleNamespace(list_agents=lambda: ["research"]))
    screen.on_mount()
    select = screen.widgets["#agents-select"]
    assert ("research", "research") in select.options
    assert select.value == "research"
    assert screen.widgets["#agents-topic-input"].focused is True
    memory_select = screen.widgets["#agents-memory-select"]
    assert memory_select.value == "cand-1"


def test_start_status_and_controls(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.AgentRegistry", lambda: SimpleNamespace(list_agents=lambda: ["research"]))
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)
    screen.on_mount()

    screen.widgets["#agents-topic-input"].value = ""
    screen._start_agent()
    assert "Enter a topic" in screen.widgets["#agents-status-output"].last

    screen.widgets["#agents-topic-input"].value = "topic"
    screen._start_agent()
    assert manager.started == ("research", "topic")
    assert "Started agent 'research'" in screen.widgets["#agents-status-output"].last

    screen._show_status()
    assert "Status: running" in screen.widgets["#agents-status-output"].last

    screen._run_control_action("pause")
    assert "Paused agent" in screen.widgets["#agents-status-output"].last
    screen._run_control_action("resume")
    assert "Resumed agent" in screen.widgets["#agents-status-output"].last
    screen._run_control_action("stop")
    assert "Stop requested" in screen.widgets["#agents-status-output"].last

    screen._show_log()
    assert "Agent log line" in "".join(screen.widgets["#agents-log"].lines)

    screen._approve_memory_candidate()
    assert manager.promoted == "cand-1"
    assert "Promoted memory candidate" in screen.widgets["#agents-memory-output"].last

    screen._reject_memory_candidate()
    assert manager.rejected == "cand-1"
    assert "Rejected memory candidate" in screen.widgets["#agents-memory-output"].last

    screen.widgets["#agents-memory-ttl-input"].value = "30"
    screen._edit_memory_candidate_ttl()
    assert manager.ttl_updated == ("cand-1", 30)
    assert "Updated TTL for memory candidate" in screen.widgets["#agents-memory-output"].last


def test_button_press_routes_to_actions(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.AgentRegistry", lambda: SimpleNamespace(list_agents=lambda: ["research"]))
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)
    screen.on_mount()
    screen.widgets["#agents-topic-input"].value = "topic"
    screen.on_input_submitted(
        SimpleNamespace(input=SimpleNamespace(id="agents-topic-input"))
    )

    for button_id in (
        "agents-status-button",
        "agents-pause-button",
        "agents-resume-button",
        "agents-stop-button",
        "agents-log-button",
        "agents-memory-refresh-button",
        "agents-memory-approve-button",
        "agents-memory-reject-button",
    ):
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id=button_id)))

    assert manager.started == ("research", "topic")


def test_log_append_autoscrolls_only_when_at_bottom() -> None:
    screen = _screen()
    log_widget = screen.widgets["#agents-log"]

    log_widget.is_vertical_scroll_end = True
    screen._append_log("first")
    assert log_widget.scroll_end_calls[-1] is True

    log_widget.is_vertical_scroll_end = False
    screen._append_log("second")
    assert log_widget.scroll_end_calls[-1] is False
