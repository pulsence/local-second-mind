from __future__ import annotations

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
        self.last_toggle = None

    def list_schedules(self, app):
        _ = app
        return [
            {
                "id": "0:research",
                "agent_name": "research",
                "interval": "daily",
                "enabled": True,
            }
        ]

    def set_schedule_enabled(self, app, schedule_id: str, *, enabled: bool) -> str:
        _ = app
        self.last_toggle = (schedule_id, enabled)
        action = "Enabled" if enabled else "Disabled"
        return f"{action} schedule '{schedule_id}'.\n"

    def format_schedule_status(self, app) -> str:
        _ = app
        return "Scheduler status (1 schedule(s)):\n- 0:research | status=idle\n"

    def get_memory_candidates(self, app, status="pending", limit=200):
        _ = app, status, limit
        return []


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
    screen.widgets["#agents-schedule-select"] = _Select("", "agents-schedule-select")
    screen.widgets["#agents-schedule-output"] = _Static()
    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()
    return screen


def test_schedule_panel_refresh_toggle_and_status(monkeypatch) -> None:
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
    schedule_select = screen.widgets["#agents-schedule-select"]
    assert schedule_select.value == "0:research"
    assert len(schedule_select.options) == 1
    assert "Loaded 1 schedule(s)." in screen.widgets["#agents-schedule-output"].last

    screen._disable_selected_schedule()
    assert manager.last_toggle == ("0:research", False)
    assert "Disabled schedule '0:research'" in screen.widgets["#agents-schedule-output"].last

    screen._enable_selected_schedule()
    assert manager.last_toggle == ("0:research", True)
    assert "Enabled schedule '0:research'" in screen.widgets["#agents-schedule-output"].last

    screen._show_schedule_status()
    assert "Scheduler status (1 schedule(s))" in screen.widgets["#agents-schedule-output"].last


def test_schedule_panel_refresh_handles_error(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()

    def _boom(app):
        _ = app
        raise RuntimeError("Scheduler unavailable.")

    manager.list_schedules = _boom  # type: ignore[assignment]
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    assert "Scheduler unavailable." in screen.widgets["#agents-schedule-output"].last
