from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
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


class _Panel:
    def __init__(self) -> None:
        self.display = True
        self.classes = set()

    def add_class(self, class_name: str) -> None:
        self.classes.add(class_name)

    def remove_class(self, class_name: str) -> None:
        self.classes.discard(class_name)


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


class _RichLog:
    def __init__(self) -> None:
        self.lines = []
        self.is_vertical_scroll_end = True
        self.scroll_end_calls = []
        self.forced_scroll_calls = []

    def write(self, message: str, scroll_end=None) -> None:
        self.lines.append(message)
        self.scroll_end_calls.append(scroll_end)

    def scroll_end(self, **kwargs) -> None:
        self.forced_scroll_calls.append(dict(kwargs))


class _Manager:
    def __init__(self) -> None:
        self.started = None
        self.promoted = None
        self.rejected = None
        self.ttl_updated = None
        self.running_rows = []
        self.pending_rows = []
        self.schedule_rows = []
        self.status_by_id = {}

    def start(self, app, name: str, topic: str) -> str:
        self.started = (name, topic)
        return f"Started agent '{name}' with topic: {topic}\n"

    def status(self, agent_id=None) -> str:
        if agent_id is not None and agent_id in self.status_by_id:
            return self.status_by_id[agent_id]
        return "Agent: research\nStatus: running\n"

    def pause(self, agent_id=None) -> str:
        _ = agent_id
        return "Paused agent 'research'.\n"

    def resume(self, agent_id=None) -> str:
        _ = agent_id
        return "Resumed agent 'research'.\n"

    def stop(self, agent_id=None) -> str:
        _ = agent_id
        return "Stop requested for agent 'research'.\n"

    def log(self, agent_id=None) -> str:
        _ = agent_id
        return "Agent log line\n"

    def list_running(self):
        return list(self.running_rows)

    def clear_log_stream(self, agent_id: str) -> None:
        _ = agent_id

    def get_pending_interactions(self):
        return list(self.pending_rows)

    def list_schedules(self, app):
        _ = app
        return list(self.schedule_rows)

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


class _SlowStopManager(_Manager):
    def __init__(self, delay_s: float = 0.25) -> None:
        super().__init__()
        self.delay_s = max(0.0, float(delay_s))

    def stop(self, agent_id=None) -> str:
        _ = agent_id
        time.sleep(self.delay_s)
        return super().stop()


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
    notifications = []
    app = SimpleNamespace(
        current_context=context,
        config=SimpleNamespace(),
        notifications=notifications,
        notify_event=lambda message, **kwargs: notifications.append((message, kwargs)),
    )
    screen = _TestableAgentsScreen(app)
    screen.widgets["#agents-select"] = _Select("")
    screen.widgets["#agents-topic-input"] = _Input("", "agents-topic-input")
    screen.widgets["#agents-status-output"] = _Static()
    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()
    return screen


def _threaded_screen(context: str = "agents"):
    notifications = []
    app = SimpleNamespace(
        current_context=context,
        config=SimpleNamespace(),
        notifications=notifications,
        notify_event=lambda message, **kwargs: notifications.append((message, kwargs)),
    )

    def _call_from_thread(callback, *args, **kwargs):
        callback(*args, **kwargs)

    app.call_from_thread = _call_from_thread
    screen = _TestableAgentsScreen(app)
    screen.widgets["#agents-select"] = _Select("")
    screen.widgets["#agents-topic-input"] = _Input("", "agents-topic-input")
    screen.widgets["#agents-status-output"] = _Static()
    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()
    return screen


def _screen_with_runtime_widgets(context: str = "agents"):
    screen = _screen(context=context)
    screen.widgets["#agents-running-table"] = _DataTable("agents-running-table")
    screen.widgets["#agents-running-output"] = _Static()
    screen.widgets["#agents-interaction-panel"] = _Panel()
    screen.widgets["#agents-interaction-indicator"] = _Static()
    screen.widgets["#agents-interaction-status-output"] = _Static()
    screen.widgets["#agents-schedule-table"] = _DataTable("agents-schedule-table")
    screen.widgets["#agents-schedule-output"] = _Static()
    return screen


def test_on_mount_populates_options_and_focus(monkeypatch) -> None:
    screen = _screen()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: _Manager())
    monkeypatch.setattr("lsm.ui.tui.screens.agents.AgentRegistry", lambda: SimpleNamespace(list_agents=lambda: ["research"]))
    screen.on_mount()
    screen._ensure_deferred_init()
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
    screen._ensure_deferred_init()

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
    screen._ensure_deferred_init()
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
    screen._log_follow_selected = False
    log_widget = screen.widgets["#agents-log"]

    log_widget.is_vertical_scroll_end = True
    screen._append_log("first")
    assert log_widget.scroll_end_calls[-1] is True

    log_widget.is_vertical_scroll_end = False
    screen._append_log("second")
    assert log_widget.scroll_end_calls[-1] is False


def test_show_log_forces_scroll_to_bottom(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    log_widget = screen.widgets["#agents-log"]
    log_widget.is_vertical_scroll_end = False
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    screen._show_log()

    assert log_widget.scroll_end_calls[-1] is True
    assert len(log_widget.forced_scroll_calls) == 1


def test_stop_action_runs_async_when_call_from_thread_available(monkeypatch) -> None:
    screen = _threaded_screen()
    manager = _SlowStopManager(delay_s=0.25)
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    start = time.monotonic()
    screen._run_control_action("stop")
    elapsed = time.monotonic() - start

    assert elapsed < 0.1
    assert "Stopping agent" in screen.widgets["#agents-status-output"].last
    worker = screen._stop_worker
    assert worker is not None
    worker.join(timeout=1.0)


def test_on_unmount_cancels_managed_workers_via_app() -> None:
    screen = _screen()
    seen = {}

    def _cancel_owner(*, owner, reason, timeout_s=None):
        seen["owner"] = owner
        seen["reason"] = reason
        seen["timeout_s"] = timeout_s
        return {"stop-action": True}

    screen.app.cancel_managed_workers_for_owner = _cancel_owner
    screen.on_unmount()

    assert seen["reason"] == "agents-unmount"


def test_on_unmount_stops_managed_timers_via_app() -> None:
    screen = _screen()
    seen = {}

    def _stop_owner(*, owner, reason):
        seen["owner"] = owner
        seen["reason"] = reason
        return {
            "running-refresh": True,
            "interaction-poll": True,
            "log-stream": True,
        }

    screen.app.stop_managed_timers_for_owner = _stop_owner
    screen.on_unmount()

    assert seen["reason"] == "agents-unmount"


def test_on_unmount_sets_status_when_stop_worker_times_out() -> None:
    screen = _screen()
    screen._STOP_WORKER_TIMEOUT_SECONDS = 0.01  # type: ignore[attr-defined]

    def _slow() -> None:
        time.sleep(0.2)

    worker = threading.Thread(target=_slow, daemon=True)
    worker.start()
    screen._stop_worker = worker

    screen.on_unmount()

    assert "did not exit before timeout" in screen.widgets["#agents-status-output"].last
    worker.join(timeout=1.0)


def test_running_agent_notifications_for_start_and_completion(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    agent_id = "agent-12345678"
    manager.running_rows = []
    screen._refresh_running_agents()

    manager.running_rows = [
        {
            "agent_id": agent_id,
            "agent_name": "research",
            "topic": "topic",
            "status": "running",
            "duration_seconds": 1.0,
        }
    ]
    screen._refresh_running_agents()

    manager.running_rows = []
    manager.status_by_id[agent_id] = (
        f"Agent ID: {agent_id}\n"
        "Agent: research\n"
        "Status: completed\n"
    )
    screen._refresh_running_agents()

    assert screen.app.notifications == [
        ("Agent 'research' started.", {"severity": "info"}),
        ("Agent 'research' completed.", {"severity": "info"}),
    ]


def test_running_agent_failed_notification(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    agent_id = "agent-failed-1234"
    manager.running_rows = []
    screen._refresh_running_agents()

    manager.running_rows = [
        {
            "agent_id": agent_id,
            "agent_name": "research",
            "topic": "topic",
            "status": "running",
            "duration_seconds": 1.0,
        }
    ]
    screen._refresh_running_agents()

    manager.running_rows = []
    manager.status_by_id[agent_id] = (
        f"Agent ID: {agent_id}\n"
        "Agent: research\n"
        "Status: failed\n"
    )
    screen._refresh_running_agents()

    assert screen.app.notifications[-1] == (
        "Agent 'research' failed.",
        {"severity": "error"},
    )


def test_interaction_pending_notification_uses_high_priority(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    manager.pending_rows = []
    screen._refresh_interaction_panel()

    manager.pending_rows = [
        {
            "request_id": "req-1",
            "request_type": "permission",
            "agent_name": "research",
            "agent_id": "agent-1",
            "tool_name": "write_file",
            "risk_level": "writes_workspace",
            "reason": "Need approval",
            "args_summary": "{}",
            "prompt": "Allow?",
        }
    ]
    screen._refresh_interaction_panel()
    screen._refresh_interaction_panel()

    assert screen.app.notifications == [
        (
            "Agent 'research' is waiting for user interaction (permission).",
            {"severity": "warning", "timeout": 10},
        )
    ]


def test_schedule_trigger_notification_emitted_on_last_run_change(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    manager.schedule_rows = [
        {
            "id": "0:research",
            "agent_name": "research",
            "interval": "hourly",
            "enabled": True,
            "last_status": "idle",
            "next_run_at": "2026-02-14T12:00:00+00:00",
            "last_run_at": None,
        }
    ]
    screen._refresh_schedule_entries()

    manager.schedule_rows = [
        {
            "id": "0:research",
            "agent_name": "research",
            "interval": "hourly",
            "enabled": True,
            "last_status": "running",
            "next_run_at": "2026-02-14T13:00:00+00:00",
            "last_run_at": "2026-02-14T12:00:00+00:00",
        }
    ]
    screen._refresh_schedule_entries()

    assert screen.app.notifications == [
        (
            "Schedule '0:research' triggered for agent 'research'.",
            {"severity": "info"},
        )
    ]


# -------------------------------------------------------------------------
# 5.7: Keyboard-first interaction parity
# -------------------------------------------------------------------------


def test_agents_bindings_include_refresh_log_status() -> None:
    """New keybindings for Ctrl+Shift+R, Ctrl+L, Ctrl+I exist on AgentsScreen."""
    binding_keys = {b.key for b in AgentsScreen.BINDINGS}
    assert "ctrl+shift+r" in binding_keys
    assert "ctrl+l" in binding_keys
    assert "ctrl+i" in binding_keys


def test_agents_bindings_no_conflict_with_app() -> None:
    """Agents screen bindings must not conflict with app-level bindings."""
    from lsm.ui.tui.app import LSMApp

    agents_keys = {b.key for b in AgentsScreen.BINDINGS}
    app_keys = {b.key for b in LSMApp.BINDINGS}
    # Tab/shift+tab are shared by design; focus navigation is expected overlap
    shared_ok = {"tab", "shift+tab"}
    conflicts = (agents_keys & app_keys) - shared_ok
    assert not conflicts, f"Key conflicts: {conflicts}"


def test_action_refresh_running_calls_refresh(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    manager.running_rows = []
    manager.pending_rows = []

    screen.action_refresh_running()

    # Should have attempted to populate the running table and interaction panel
    assert "No running agents" in screen.widgets["#agents-running-output"].last


def test_action_show_agent_log_calls_log(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    screen._selected_agent_id = "agent-1"
    screen.action_show_agent_log()

    # Log should have been written
    assert screen.widgets["#agents-log"].lines


def test_action_show_agent_status_calls_status(monkeypatch) -> None:
    screen = _screen_with_runtime_widgets()
    manager = _Manager()
    monkeypatch.setattr("lsm.ui.tui.screens.agents.get_agent_runtime_manager", lambda: manager)

    screen._selected_agent_id = "agent-1"
    screen.action_show_agent_status()

    # Status output should have been set
    assert "Status" in screen.widgets["#agents-status-output"].last or "running" in screen.widgets["#agents-status-output"].last
