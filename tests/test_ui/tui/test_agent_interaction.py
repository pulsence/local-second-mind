from __future__ import annotations

from types import SimpleNamespace

from lsm.ui.tui.screens.agents import AgentsScreen


class _Input:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.disabled = False
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _Select:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.options = []

    def set_options(self, options) -> None:
        self.options = list(options)


class _DataTable:
    def __init__(self, widget_id: str = "") -> None:
        self.id = widget_id
        self.columns = []
        self.rows = []
        self.cursor_row = 0
        self.cursor_type = None
        self.zebra_stripes = False

    def add_columns(self, *columns) -> None:
        self.columns.extend(columns)

    def clear(self, *, columns: bool = False) -> None:
        self.rows = []
        if columns:
            self.columns = []

    def add_row(self, *row_values) -> None:
        self.rows.append(tuple(row_values))

    def move_cursor(self, *, row: int, column: int = 0) -> None:
        _ = column
        self.cursor_row = row

    @property
    def cursor_coordinate(self):
        return SimpleNamespace(row=self.cursor_row, column=0)


class _Static:
    def __init__(self, text: str = "") -> None:
        self.last = text
        self.display = True
        self.classes: set[str] = set()

    def update(self, message: str) -> None:
        self.last = message

    def add_class(self, class_name: str) -> None:
        self.classes.add(class_name)

    def remove_class(self, class_name: str) -> None:
        self.classes.discard(class_name)


class _Container:
    def __init__(self) -> None:
        self.display = True
        self.classes: set[str] = set()

    def add_class(self, class_name: str) -> None:
        self.classes.add(class_name)

    def remove_class(self, class_name: str) -> None:
        self.classes.discard(class_name)


class _Button:
    def __init__(self, widget_id: str = "") -> None:
        self.id = widget_id
        self.disabled = False
        self.label = ""


class _RichLog:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.scroll_end_calls: list[bool | None] = []
        self.is_vertical_scroll_end = True

    def write(self, message: str, scroll_end=None) -> None:
        self.lines.append(message)
        self.scroll_end_calls.append(scroll_end)

    def clear(self) -> None:
        self.lines = []
        self.scroll_end_calls = []


class _Manager:
    def __init__(self) -> None:
        self.running_rows = [
            {
                "agent_id": "aaaaaaaa11111111",
                "agent_name": "research",
                "topic": "alpha topic",
                "status": "running",
                "duration_seconds": 12.0,
            },
            {
                "agent_id": "bbbbbbbb22222222",
                "agent_name": "writing",
                "topic": "beta topic",
                "status": "waiting_user",
                "duration_seconds": 75.0,
            },
        ]
        self.pending_rows: list[dict] = []
        self.log_requests: list[str | None] = []
        self.responses: list[tuple[str, dict]] = []
        self.stream_payloads: dict[str, list[dict]] = {}
        self.stream_dropped: dict[str, int] = {}
        self.cleared_stream_ids: list[str] = []

    def list_running(self):
        return list(self.running_rows)

    def log(self, agent_id=None):
        self.log_requests.append(agent_id)
        if agent_id:
            return f"log for {agent_id}\n"
        return "combined log\n"

    def clear_log_stream(self, agent_id: str) -> None:
        self.cleared_stream_ids.append(str(agent_id))
        self.stream_payloads[str(agent_id)] = []
        self.stream_dropped[str(agent_id)] = 0

    def drain_log_stream(self, agent_id: str, max_entries: int = 200) -> dict:
        normalized = str(agent_id)
        rows = list(self.stream_payloads.get(normalized, []))
        limit = max(1, int(max_entries))
        drained = rows[:limit]
        self.stream_payloads[normalized] = rows[limit:]
        dropped = int(self.stream_dropped.get(normalized, 0))
        self.stream_dropped[normalized] = 0
        return {
            "agent_id": normalized,
            "entries": drained,
            "dropped_count": dropped,
            "has_more": bool(self.stream_payloads.get(normalized)),
        }

    def get_pending_interactions(self):
        return list(self.pending_rows)

    def respond_to_interaction(self, agent_id: str, response: dict) -> str:
        self.responses.append((agent_id, dict(response)))
        self.pending_rows = [
            row
            for row in self.pending_rows
            if str(row.get("request_id", "")) != str(response.get("request_id", ""))
        ]
        return f"Posted interaction response to agent '{agent_id}'.\n"

    def list_schedules(self, app):
        _ = app
        return []

    def get_memory_candidates(self, app, status="pending", limit=200):
        _ = app, status, limit
        return []

    def start(self, app, name: str, topic: str) -> str:
        _ = app
        return f"Started agent '{name}' with topic: {topic}\n"

    def status(self, agent_id=None) -> str:
        if agent_id:
            return f"Agent ID: {agent_id}\nStatus: running\n"
        return "Agents: 2 active, 0 recent completed\n"

    def pause(self, agent_id=None) -> str:
        _ = agent_id
        return "Paused.\n"

    def resume(self, agent_id=None) -> str:
        _ = agent_id
        return "Resumed.\n"

    def stop(self, agent_id=None) -> str:
        _ = agent_id
        return "Stopped.\n"


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
    screen.widgets["#agents-log"] = _RichLog()

    screen.widgets["#agents-running-table"] = _DataTable("agents-running-table")
    screen.widgets["#agents-running-output"] = _Static()
    screen.widgets["#agents-running-refresh-toggle-button"] = _Button(
        "agents-running-refresh-toggle-button"
    )
    screen.widgets["#agents-running-refresh-interval-select"] = _Select(
        "2.0",
        "agents-running-refresh-interval-select",
    )
    screen.widgets["#agents-running-refresh-now-button"] = _Button(
        "agents-running-refresh-now-button"
    )
    screen.widgets["#agents-clear-unread-button"] = _Button("agents-clear-unread-button")

    screen.widgets["#agents-interaction-panel"] = _Container()
    screen.widgets["#agents-interaction-indicator"] = _Static()
    screen.widgets["#agents-interaction-refresh-toggle-button"] = _Button(
        "agents-interaction-refresh-toggle-button"
    )
    screen.widgets["#agents-interaction-refresh-interval-select"] = _Select(
        "1.0",
        "agents-interaction-refresh-interval-select",
    )
    screen.widgets["#agents-interaction-refresh-now-button"] = _Button(
        "agents-interaction-refresh-now-button"
    )
    screen.widgets["#agents-interaction-status-output"] = _Static()
    screen.widgets["#agents-interaction-type"] = _Static()
    screen.widgets["#agents-interaction-agent"] = _Static()
    screen.widgets["#agents-interaction-tool"] = _Static()
    screen.widgets["#agents-interaction-risk"] = _Static()
    screen.widgets["#agents-interaction-reason"] = _Static()
    screen.widgets["#agents-interaction-args"] = _Static()
    screen.widgets["#agents-interaction-prompt"] = _Static()
    screen.widgets["#agents-interaction-deny-input"] = _Input("", "agents-interaction-deny-input")
    screen.widgets["#agents-interaction-reply-input"] = _Input("", "agents-interaction-reply-input")
    screen.widgets["#agents-interaction-approve-button"] = _Button("agents-interaction-approve-button")
    screen.widgets["#agents-interaction-approve-session-button"] = _Button(
        "agents-interaction-approve-session-button"
    )
    screen.widgets["#agents-interaction-deny-button"] = _Button("agents-interaction-deny-button")
    screen.widgets["#agents-interaction-reply-button"] = _Button("agents-interaction-reply-button")

    screen.widgets["#agents-meta-task-table"] = _DataTable("agents-meta-task-table")
    screen.widgets["#agents-meta-runs-table"] = _DataTable("agents-meta-runs-table")
    screen.widgets["#agents-meta-output"] = _Static()
    screen.widgets["#agents-meta-artifacts-output"] = _Static()

    screen.widgets["#agents-schedule-table"] = _DataTable("agents-schedule-table")
    screen.widgets["#agents-schedule-output"] = _Static()
    screen.widgets["#agents-schedule-agent-input"] = _Input("", "agents-schedule-agent-input")
    screen.widgets["#agents-schedule-interval-input"] = _Input("", "agents-schedule-interval-input")
    screen.widgets["#agents-schedule-params-input"] = _Input("", "agents-schedule-params-input")
    screen.widgets["#agents-schedule-concurrency-select"] = _Select("skip", "agents-schedule-concurrency-select")
    screen.widgets["#agents-schedule-confirmation-select"] = _Select("auto", "agents-schedule-confirmation-select")

    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    screen.widgets["#agents-log-follow-toggle-button"] = _Button("agents-log-follow-toggle-button")
    return screen


def test_running_agents_selection_routes_log(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    running_table = screen.widgets["#agents-running-table"]
    assert len(running_table.rows) == 2
    assert screen._selected_agent_id == "aaaaaaaa11111111"
    assert manager.log_requests[-1] == "aaaaaaaa11111111"
    assert manager.cleared_stream_ids[-1] == "aaaaaaaa11111111"

    screen.on_data_table_row_selected(
        SimpleNamespace(data_table=running_table, cursor_row=1)
    )
    assert screen._selected_agent_id == "bbbbbbbb22222222"
    assert manager.log_requests[-1] == "bbbbbbbb22222222"
    assert manager.cleared_stream_ids[-1] == "bbbbbbbb22222222"


def test_permission_interaction_panel_and_deny(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    manager.pending_rows = [
        {
            "agent_id": "bbbbbbbb22222222",
            "agent_name": "writing",
            "request_id": "perm-1",
            "request_type": "permission",
            "tool_name": "write_file",
            "risk_level": "writes_workspace",
            "reason": "Tool requires confirmation",
            "args_summary": '{"path":"./notes.md"}',
            "prompt": "Allow write?",
        }
    ]
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    indicator = screen.widgets["#agents-interaction-indicator"]
    panel = screen.widgets["#agents-interaction-panel"]
    assert "pending (1)" in indicator.last.lower()
    assert panel.display is True
    assert "agents-interaction-panel-active" in panel.classes
    assert screen.widgets["#agents-interaction-approve-button"].disabled is False
    assert screen.widgets["#agents-interaction-reply-button"].disabled is True

    screen.widgets["#agents-interaction-deny-input"].value = "No writes."
    screen._deny_interaction()
    assert manager.responses[-1][0] == "bbbbbbbb22222222"
    assert manager.responses[-1][1]["decision"] == "deny"
    assert manager.responses[-1][1]["user_message"] == "No writes."
    assert screen.widgets["#agents-interaction-panel"].display is False


def test_clarification_interaction_reply(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    manager.pending_rows = [
        {
            "agent_id": "aaaaaaaa11111111",
            "agent_name": "research",
            "request_id": "clarify-1",
            "request_type": "clarification",
            "tool_name": "ask_user",
            "risk_level": "read_only",
            "reason": "Need user detail",
            "args_summary": '{"prompt":"Which source?"}',
            "prompt": "Which source should I use?",
        }
    ]
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    assert screen.widgets["#agents-interaction-approve-button"].disabled is True
    assert screen.widgets["#agents-interaction-reply-button"].disabled is False
    assert screen.widgets["#agents-interaction-reply-input"].disabled is False

    screen.widgets["#agents-interaction-reply-input"].value = "Use the meeting notes."
    screen.action_interaction_reply()
    assert manager.responses[-1][1]["decision"] == "reply"
    assert manager.responses[-1][1]["user_message"] == "Use the meeting notes."
    assert screen.widgets["#agents-interaction-panel"].display is False


def test_running_navigation_actions(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    assert screen._selected_agent_id == "aaaaaaaa11111111"

    screen.action_running_next()
    assert screen._selected_agent_id == "bbbbbbbb22222222"

    screen.action_running_prev()
    assert screen._selected_agent_id == "aaaaaaaa11111111"


def test_log_stream_drain_appends_formatted_entries_and_drop_notice(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    selected = str(screen._selected_agent_id or "")
    manager.stream_payloads[selected] = [
        {
            "timestamp": "2026-02-14T12:00:00",
            "actor": "tool",
            "action": "write_file",
            "action_arguments": {"path": "./notes.md", "content": "hello"},
            "content": "Wrote file successfully",
        },
        {
            "timestamp": "2026-02-14T12:00:01",
            "actor": "llm",
            "action": "DONE",
            "action_arguments": {},
            "content": "Finished task",
        },
    ]
    manager.stream_dropped[selected] = 2

    screen._drain_log_streams()
    rendered_lines = [str(line) for line in screen.widgets["#agents-log"].lines]
    assert any("Dropped 2 log entries" in line for line in rendered_lines)
    assert any("[TOOL]" in line and "write_file(" in line and "->" in line for line in rendered_lines)
    assert any("[LLM]" in line and "action=DONE" in line for line in rendered_lines)


def test_refresh_controls_toggle_and_interval_updates(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    class _Timer:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    def _fake_start_timer(*, interval_seconds, callback):
        _ = interval_seconds, callback
        return _Timer()

    screen._start_timer = _fake_start_timer  # type: ignore[method-assign]
    screen.on_mount()

    running_toggle = screen.widgets["#agents-running-refresh-toggle-button"]
    interaction_toggle = screen.widgets["#agents-interaction-refresh-toggle-button"]
    assert running_toggle.label == "Auto: On"
    assert interaction_toggle.label == "Auto: On"

    screen.on_button_pressed(
        SimpleNamespace(button=SimpleNamespace(id="agents-running-refresh-toggle-button"))
    )
    assert screen._running_refresh_enabled is False
    assert running_toggle.label == "Auto: Off"

    screen.on_select_changed(
        SimpleNamespace(
            select=screen.widgets["#agents-running-refresh-interval-select"],
            value="5.0",
        )
    )
    assert screen._running_refresh_interval_seconds == 5.0

    screen.on_button_pressed(
        SimpleNamespace(button=SimpleNamespace(id="agents-interaction-refresh-toggle-button"))
    )
    assert screen._interaction_poll_enabled is False
    assert interaction_toggle.label == "Auto: Off"

    screen.on_select_changed(
        SimpleNamespace(
            select=screen.widgets["#agents-interaction-refresh-interval-select"],
            value="2.0",
        )
    )
    assert screen._interaction_poll_interval_seconds == 2.0


def test_follow_toggle_controls_log_autoscroll(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    log_widget = screen.widgets["#agents-log"]
    log_widget.is_vertical_scroll_end = False

    screen.on_button_pressed(
        SimpleNamespace(button=SimpleNamespace(id="agents-log-follow-toggle-button"))
    )
    assert screen._log_follow_selected is False

    screen._append_log("first line")
    assert log_widget.scroll_end_calls[-1] is False

    screen.on_button_pressed(
        SimpleNamespace(button=SimpleNamespace(id="agents-log-follow-toggle-button"))
    )
    assert screen._log_follow_selected is True
    screen._append_log("second line")
    assert log_widget.scroll_end_calls[-1] is True


def test_unread_log_counters_increment_and_clear(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["research", "writing"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    assert screen._selected_agent_id == "aaaaaaaa11111111"

    manager.stream_payloads["bbbbbbbb22222222"] = [
        {
            "timestamp": "2026-02-14T12:01:00",
            "actor": "agent",
            "content": "first",
            "action": "",
            "action_arguments": {},
        },
        {
            "timestamp": "2026-02-14T12:01:01",
            "actor": "agent",
            "content": "second",
            "action": "",
            "action_arguments": {},
        },
    ]
    screen._drain_log_streams()

    assert screen._unread_log_counts["bbbbbbbb22222222"] == 2
    screen._refresh_running_agents()
    running_table = screen.widgets["#agents-running-table"]
    assert running_table.rows[1][-1] == "2"

    screen.on_data_table_row_selected(
        SimpleNamespace(data_table=running_table, cursor_row=1)
    )
    assert screen._selected_agent_id == "bbbbbbbb22222222"
    assert screen._unread_log_counts["bbbbbbbb22222222"] == 0

    screen._unread_log_counts["aaaaaaaa11111111"] = 3
    screen.on_button_pressed(
        SimpleNamespace(button=SimpleNamespace(id="agents-clear-unread-button"))
    )
    assert all(count == 0 for count in screen._unread_log_counts.values())
