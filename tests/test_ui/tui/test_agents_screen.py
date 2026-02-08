from __future__ import annotations

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

    def write(self, message: str) -> None:
        self.lines.append(message)


class _Manager:
    def __init__(self) -> None:
        self.started = None

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
    app = SimpleNamespace(current_context=context)
    screen = _TestableAgentsScreen(app)
    screen.widgets["#agents-select"] = _Select("")
    screen.widgets["#agents-topic-input"] = _Input("", "agents-topic-input")
    screen.widgets["#agents-status-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()
    return screen


def test_on_mount_populates_options_and_focus(monkeypatch) -> None:
    screen = _screen()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.AgentRegistry", lambda: SimpleNamespace(list_agents=lambda: ["research"]))
    screen.on_mount()
    select = screen.widgets["#agents-select"]
    assert ("research", "research") in select.options
    assert select.value == "research"
    assert screen.widgets["#agents-topic-input"].focused is True


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


def test_button_press_routes_to_actions(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.AgentRegistry", lambda: SimpleNamespace(list_agents=lambda: ["research"]))
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)
    screen.on_mount()
    screen.widgets["#agents-topic-input"].value = "topic"

    for button_id in (
        "agents-start-button",
        "agents-status-button",
        "agents-pause-button",
        "agents-resume-button",
        "agents-stop-button",
        "agents-log-button",
    ):
        screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id=button_id)))

    assert manager.started == ("research", "topic")

