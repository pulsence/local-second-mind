"""
Agents screen for launching and monitoring agent runs.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static, Input, Button, Select, RichLog, DataTable
from textual.widget import Widget

from lsm.logging import get_logger

logger = get_logger(__name__)


class AgentStopWorkerCompleted(Message):
    """Stop-worker completion payload queued back to UI thread."""

    def __init__(self, output: str) -> None:
        self.output = output
        super().__init__()


class AgentRuntimeEventMessage(Message):
    """Runtime event forwarded from app-level manager sink."""

    def __init__(self, event: Any) -> None:
        self.event = event
        super().__init__()


def AgentRegistry():
    """
    Lazily construct the agent registry for TUI option loading.
    """
    from lsm.agents.factory import AgentRegistry as _AgentRegistry

    return _AgentRegistry()


def get_agent_runtime_manager():
    """
    Lazily import the shell runtime manager to keep TUI module import fast.
    """
    from lsm.ui.shell.commands.agents import get_agent_runtime_manager as _get_runtime_manager

    return _get_runtime_manager()


class AgentsScreen(Widget):
    """
    UI surface for starting and controlling agents.
    """

    BINDINGS = [
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
        Binding("f6", "running_prev", "Prev Agent", show=False),
        Binding("f7", "running_next", "Next Agent", show=False),
        Binding("f8", "interaction_approve", "Approve", show=False),
        Binding("f9", "interaction_approve_session", "Approve Session", show=False),
        Binding("f10", "interaction_deny", "Deny", show=False),
        Binding("f11", "interaction_reply", "Reply", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._schedule_ids: list[str] = []
        self._schedule_row_index: int = 0
        self._schedule_last_run_by_id: dict[str, str] = {}
        self._schedule_agent_by_id: dict[str, str] = {}
        self._schedule_notifications_initialized: bool = False
        self._running_agent_ids: list[str] = []
        self._known_running_agent_ids: set[str] = set()
        self._known_running_agent_names: dict[str, str] = {}
        self._running_notifications_initialized: bool = False
        self._running_row_index: int = 0
        self._selected_agent_id: Optional[str] = None
        self._pending_interaction: Optional[dict[str, Any]] = None
        self._known_pending_interaction_ids: set[str] = set()
        self._interaction_notifications_initialized: bool = False
        self._running_refresh_timer: Optional[Timer] = None
        self._interaction_poll_timer: Optional[Timer] = None
        self._log_stream_timer: Optional[Timer] = None
        self._stop_worker: Optional[threading.Thread] = None
        self._running_refresh_enabled: bool = True
        self._interaction_poll_enabled: bool = True
        self._running_refresh_interval_seconds: float = 2.0
        self._interaction_poll_interval_seconds: float = 1.0
        self._log_follow_selected: bool = True
        self._unread_log_counts: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="agents-layout"):
            with ScrollableContainer(id="agents-top"):
                with ScrollableContainer(id="agents-left"):
                    with Container(id="agents-control-panel"):
                        yield Static("Agents", classes="agents-section-title")
                        yield Static("Agent", classes="agents-label")
                        yield Select([], id="agents-select")
                        yield Static("Topic", classes="agents-label")
                        yield Input(
                            placeholder="Research topic",
                            id="agents-topic-input",
                        )
                    with Container(id="agents-running-panel"):
                        with Horizontal(classes="agents-panel-header"):
                            yield Static("Running Agents", classes="agents-section-title")
                            yield Static(
                                "No pending interaction",
                                id="agents-interaction-indicator",
                                markup=False,
                            )
                        with Horizontal(id="agents-running-refresh-controls"):
                            yield Button(
                                "Auto: On",
                                id="agents-running-refresh-toggle-button",
                            )
                            yield Button(
                                "Refresh",
                                id="agents-running-refresh-now-button",
                            )
                            yield Button(
                                "Clear",
                                id="agents-clear-unread-button",
                            )
                        with Vertical(id="agents-running-refresh-interval-row"):
                            yield Static(
                                "Refresh every",
                                classes="agents-label agents-refresh-interval-label",
                            )
                            yield Select(
                                [],
                                id="agents-running-refresh-interval-select",
                            )
                        yield DataTable(id="agents-running-table")
                        yield Static(
                            "No running agents.",
                            id="agents-running-output",
                            markup=False,
                        )
                    with Container(id="agents-status-panel"):
                        yield Static("Status", classes="agents-section-title")
                        yield Static("No active agent.", id="agents-status-output", markup=False)
                        with Vertical(id="agents-buttons"):
                            with Horizontal(classes="agents-button-row"):
                                yield Button("Status", id="agents-status-button")
                                yield Button("Pause", id="agents-pause-button")
                                yield Button("Resume", id="agents-resume-button")
                            with Horizontal(classes="agents-button-row"):
                                yield Button("Stop", id="agents-stop-button", variant="error")
                                yield Button("Log", id="agents-log-button")
                    with Container(id="agents-interaction-panel"):
                        yield Static("Interaction Request", classes="agents-section-title")
                        with Horizontal(id="agents-interaction-refresh-controls"):
                            yield Button(
                                "Auto: On",
                                id="agents-interaction-refresh-toggle-button",
                            )
                            yield Button(
                                "Refresh",
                                id="agents-interaction-refresh-now-button",
                            )
                        with Vertical(id="agents-interaction-refresh-interval-row"):
                            yield Static(
                                "Check every",
                                classes="agents-label agents-refresh-interval-label",
                            )
                            yield Select(
                                [],
                                id="agents-interaction-refresh-interval-select",
                            )
                        yield Static(
                            "No pending interaction requests.",
                            id="agents-interaction-status-output",
                            markup=False,
                        )
                        yield Static("Type: -", id="agents-interaction-type", markup=False)
                        yield Static("Agent: -", id="agents-interaction-agent", markup=False)
                        yield Static("Tool: -", id="agents-interaction-tool", markup=False)
                        yield Static("Risk: -", id="agents-interaction-risk", markup=False)
                        yield Static("Reason: -", id="agents-interaction-reason", markup=False)
                        yield Static("Args: -", id="agents-interaction-args", markup=False)
                        yield Static("Prompt: -", id="agents-interaction-prompt", markup=False)
                        yield Input(
                            placeholder="Optional deny reason",
                            id="agents-interaction-deny-input",
                        )
                        yield Input(
                            placeholder="Reply text",
                            id="agents-interaction-reply-input",
                        )
                        with Vertical(id="agents-interaction-buttons"):
                            with Horizontal(classes="agents-button-row"):
                                yield Button(
                                    "Approve",
                                    id="agents-interaction-approve-button",
                                    variant="success",
                                )
                                yield Button(
                                    "Approve Session",
                                    id="agents-interaction-approve-session-button",
                                )
                                yield Button(
                                    "Deny",
                                    id="agents-interaction-deny-button",
                                    variant="error",
                                )
                            with Horizontal(classes="agents-button-row"):
                                yield Button(
                                    "Send Reply",
                                    id="agents-interaction-reply-button",
                                    variant="primary",
                                )
                    with Container(id="agents-meta-panel"):
                        yield Static("Meta Agent", classes="agents-section-title")
                        yield DataTable(id="agents-meta-task-table")
                        yield DataTable(id="agents-meta-runs-table")
                        with Horizontal(id="agents-meta-buttons"):
                            yield Button("Refresh Meta", id="agents-meta-refresh-button")
                            yield Button("Meta Log", id="agents-meta-log-button")
                        yield Static("No active meta-agent.", id="agents-meta-output", markup=False)
                        yield Static("No meta artifacts.", id="agents-meta-artifacts-output", markup=False)
                    with Container(id="agents-schedule-panel"):
                        yield Static("Schedules", classes="agents-section-title")
                        yield DataTable(id="agents-schedule-table")
                        yield Static("Add Schedule", classes="agents-label")
                        yield Input(
                            placeholder="Agent name (e.g., research)",
                            id="agents-schedule-agent-input",
                        )
                        yield Input(
                            placeholder="Interval (daily|hourly|weekly|3600s|cron)",
                            id="agents-schedule-interval-input",
                        )
                        yield Input(
                            placeholder='Params JSON (optional, e.g. {"topic":"daily"})',
                            id="agents-schedule-params-input",
                        )
                        yield Static("Concurrency", classes="agents-label")
                        yield Select(
                            [
                                ("skip", "skip"),
                                ("queue", "queue"),
                                ("cancel", "cancel"),
                            ],
                            id="agents-schedule-concurrency-select",
                        )
                        yield Static("Confirmation", classes="agents-label")
                        yield Select(
                            [
                                ("auto", "auto"),
                                ("confirm", "confirm"),
                                ("deny", "deny"),
                            ],
                            id="agents-schedule-confirmation-select",
                        )
                        with Vertical(id="agents-schedule-buttons"):
                            with Horizontal(classes="agents-button-row"):
                                yield Button("Add", id="agents-schedule-add-button", variant="primary")
                                yield Button("Remove", id="agents-schedule-remove-button", variant="error")
                                yield Button("Refresh", id="agents-schedule-refresh-button")
                            with Horizontal(classes="agents-button-row"):
                                yield Button("Enable", id="agents-schedule-enable-button")
                                yield Button("Disable", id="agents-schedule-disable-button")
                                yield Button("Status", id="agents-schedule-status-button")
                        yield Static(
                            "No schedules loaded.",
                            id="agents-schedule-output",
                            markup=False,
                        )
                    with Container(id="agents-memory-panel"):
                        yield Static("Memory Candidates", classes="agents-section-title")
                        yield Select([], id="agents-memory-select")
                        yield Input(
                            placeholder="TTL days (e.g., 30)",
                            id="agents-memory-ttl-input",
                        )
                        with Vertical(id="agents-memory-buttons"):
                            with Horizontal(classes="agents-button-row"):
                                yield Button("Refresh", id="agents-memory-refresh-button")
                                yield Button("Approve", id="agents-memory-approve-button")
                            with Horizontal(classes="agents-button-row"):
                                yield Button("Reject", id="agents-memory-reject-button")
                                yield Button("Edit TTL", id="agents-memory-ttl-button")
                        yield Static(
                            "No memory candidates loaded.",
                            id="agents-memory-output",
                            markup=False,
                        )

                with Container(id="agents-log-panel"):
                    with Horizontal(id="agents-log-header"):
                        yield Static("Agent Log", classes="agents-section-title")
                        yield Button(
                            "Follow: On",
                            id="agents-log-follow-toggle-button",
                        )
                    with ScrollableContainer(id="agents-log-scroll"):
                        yield RichLog(id="agents-log", auto_scroll=False, wrap=True)

    def on_mount(self) -> None:
        """Initialize agent select options and focus."""
        self._refresh_agent_options()
        self._initialize_running_controls()
        self._initialize_refresh_controls()
        self._refresh_running_agents()
        self._refresh_interaction_panel()
        self._initialize_meta_controls()
        self._refresh_meta_panel()
        self._initialize_schedule_controls()
        self._refresh_schedule_entries()
        self._refresh_memory_candidates()
        self._restart_running_refresh_timer()
        self._restart_interaction_poll_timer()
        self._log_stream_timer = self._start_timer(
            interval_seconds=0.5,
            callback=self._drain_log_streams,
        )
        self._focus_default_input()

    def on_unmount(self) -> None:
        for timer in (
            self._running_refresh_timer,
            self._interaction_poll_timer,
            self._log_stream_timer,
        ):
            self._stop_timer(timer)
        self._running_refresh_timer = None
        self._interaction_poll_timer = None
        self._log_stream_timer = None

    def on_agent_stop_worker_completed(self, message: AgentStopWorkerCompleted) -> None:
        """Apply stop action output on the UI thread after worker completion."""
        self._stop_worker = None
        self._apply_control_action_output(message.output)

    def on_agent_runtime_event_message(self, message: AgentRuntimeEventMessage) -> None:
        """Handle manager runtime events routed through app-level event queue."""
        event_type = str(getattr(message.event, "event_type", "")).strip().lower()
        if event_type in {"run_started", "run_completed"}:
            self._refresh_running_agents()
            self._refresh_interaction_panel()
            self._refresh_meta_panel()
            return
        if event_type == "interaction_response":
            self._refresh_interaction_panel()
            self._refresh_running_agents()

    def action_running_prev(self) -> None:
        self._move_running_selection(-1)

    def action_running_next(self) -> None:
        self._move_running_selection(1)

    def action_interaction_approve(self) -> None:
        self._approve_interaction()

    def action_interaction_approve_session(self) -> None:
        self._approve_interaction_session()

    def action_interaction_deny(self) -> None:
        self._deny_interaction()

    def action_interaction_reply(self) -> None:
        self._reply_to_interaction()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle control button presses."""
        button_id = event.button.id or ""
        if button_id == "agents-running-refresh-toggle-button":
            self._toggle_running_refresh()
            return
        if button_id == "agents-running-refresh-now-button":
            self._refresh_running_agents()
            return
        if button_id == "agents-clear-unread-button":
            self._clear_unread_counts(manual=True)
            return
        if button_id == "agents-interaction-refresh-toggle-button":
            self._toggle_interaction_refresh()
            return
        if button_id == "agents-interaction-refresh-now-button":
            self._refresh_interaction_panel()
            return
        if button_id == "agents-log-follow-toggle-button":
            self._toggle_log_follow_mode()
            return
        if button_id == "agents-status-button":
            self._show_status()
            return
        if button_id == "agents-pause-button":
            self._run_control_action("pause")
            return
        if button_id == "agents-resume-button":
            self._run_control_action("resume")
            return
        if button_id == "agents-stop-button":
            self._run_control_action("stop")
            return
        if button_id == "agents-log-button":
            self._show_log()
            return
        if button_id == "agents-interaction-approve-button":
            self._approve_interaction()
            return
        if button_id == "agents-interaction-approve-session-button":
            self._approve_interaction_session()
            return
        if button_id == "agents-interaction-deny-button":
            self._deny_interaction()
            return
        if button_id == "agents-interaction-reply-button":
            self._reply_to_interaction()
            return
        if button_id == "agents-meta-refresh-button":
            self._refresh_meta_panel()
            return
        if button_id == "agents-meta-log-button":
            self._show_meta_log()
            return
        if button_id == "agents-schedule-add-button":
            self._add_schedule()
            return
        if button_id == "agents-schedule-remove-button":
            self._remove_selected_schedule()
            return
        if button_id == "agents-schedule-refresh-button":
            self._refresh_schedule_entries()
            return
        if button_id == "agents-schedule-enable-button":
            self._enable_selected_schedule()
            return
        if button_id == "agents-schedule-disable-button":
            self._disable_selected_schedule()
            return
        if button_id == "agents-schedule-status-button":
            self._show_schedule_status()
            return
        if button_id == "agents-memory-refresh-button":
            self._refresh_memory_candidates()
            return
        if button_id == "agents-memory-approve-button":
            self._approve_memory_candidate()
            return
        if button_id == "agents-memory-reject-button":
            self._reject_memory_candidate()
            return
        if button_id == "agents-memory-ttl-button":
            self._edit_memory_candidate_ttl()
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "agents-topic-input":
            self._start_agent()
            return
        if event.input.id == "agents-interaction-reply-input":
            self._reply_to_interaction()
            return

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = getattr(event.select, "id", "")
        value = getattr(event, "value", None)
        if select_id == "agents-running-refresh-interval-select":
            parsed = self._parse_refresh_interval(value)
            if parsed is None:
                return
            self._running_refresh_interval_seconds = parsed
            self._restart_running_refresh_timer()
            self._update_refresh_controls()
            self._set_status(
                (
                    "Running-agents refresh interval set to "
                    f"{self._format_interval_label(self._running_refresh_interval_seconds)}."
                )
            )
            return
        if select_id == "agents-interaction-refresh-interval-select":
            parsed = self._parse_refresh_interval(value)
            if parsed is None:
                return
            self._interaction_poll_interval_seconds = parsed
            self._restart_interaction_poll_timer()
            self._update_refresh_controls()
            self._set_interaction_status(
                (
                    "Interaction polling interval set to "
                    f"{self._format_interval_label(self._interaction_poll_interval_seconds)}."
                )
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = getattr(event, "data_table", None)
        if table is None:
            return
        row_index = getattr(event, "cursor_row", None)
        if not isinstance(row_index, int) or row_index < 0:
            return
        if table.id == "agents-running-table":
            self._running_row_index = row_index
            self._select_running_agent_by_row(row_index)
            return
        if table.id == "agents-schedule-table":
            self._schedule_row_index = row_index
            return

    def _refresh_agent_options(self) -> None:
        agent_select = self.query_one("#agents-select", Select)
        options = [(name, name) for name in AgentRegistry().list_agents()]
        agent_select.set_options(options)
        if options:
            agent_select.value = options[0][1]

    def _focus_default_input(self) -> None:
        if getattr(self.app, "current_context", None) != "agents":
            return
        topic_input = self.query_one("#agents-topic-input", Input)
        self.call_after_refresh(topic_input.focus)

    def _start_agent(self) -> None:
        agent_name = self.query_one("#agents-select", Select).value
        if not isinstance(agent_name, str) or not agent_name:
            self._set_status("Select an agent before starting.")
            return
        topic = self.query_one("#agents-topic-input", Input).value.strip()
        if not topic:
            self._set_status("Enter a topic to start an agent.")
            return
        manager = get_agent_runtime_manager()
        output = manager.start(self.app, agent_name, topic)
        self._set_status(output.strip())
        self._append_log(output)
        self._refresh_running_agents()
        self._refresh_interaction_panel()
        self._refresh_meta_panel()

    def _show_status(self) -> None:
        manager = get_agent_runtime_manager()
        output = manager.status()
        self._set_status(output.strip())
        self._refresh_meta_panel()

    def _run_control_action(self, action: str) -> None:
        manager = get_agent_runtime_manager()
        target_id = self._selected_agent_id
        if action == "pause":
            output = self._call_manager_with_agent_id(manager.pause, target_id)
        elif action == "resume":
            output = self._call_manager_with_agent_id(manager.resume, target_id)
        elif action == "stop":
            if self._can_run_async_stop():
                self._start_stop_worker(manager=manager, target_id=target_id)
                return
            output = self._call_manager_with_agent_id(manager.stop, target_id)
        else:
            output = "Unsupported control action."
        self._apply_control_action_output(output)

    def _apply_control_action_output(self, output: str) -> None:
        self._set_status(output.strip())
        self._append_log(output)
        self._refresh_running_agents()
        self._refresh_interaction_panel()
        self._show_selected_agent_log()
        self._refresh_meta_panel()

    def _can_run_async_stop(self) -> bool:
        app_obj = getattr(self, "app", None)
        callback = getattr(app_obj, "call_from_thread", None)
        return callable(callback)

    def _start_stop_worker(self, *, manager: Any, target_id: Optional[str]) -> None:
        if self._stop_worker is not None and self._stop_worker.is_alive():
            self._set_status("Stop already in progress.")
            return
        self._set_status("Stopping agent...")

        def _worker() -> None:
            output = self._call_manager_with_agent_id(manager.stop, target_id)
            self._post_ui_message(
                AgentStopWorkerCompleted(output),
                source="agents-stop-worker",
            )

        self._stop_worker = threading.Thread(
            target=_worker,
            daemon=True,
            name="AgentsScreenStopAction",
        )
        self._stop_worker.start()

    def _post_ui_message(self, message: Message, *, source: str) -> None:
        """Queue UI updates through app/main-thread message flow."""
        app_obj = getattr(self, "app", None)
        post_message = getattr(app_obj, "post_ui_message", None)
        if callable(post_message):
            try:
                post_message(message, source=source)
                return
            except Exception:
                pass

        callback = getattr(app_obj, "call_from_thread", None)
        if callable(callback):
            try:
                callback(lambda: self.post_message(message))
                return
            except Exception:
                pass
        try:
            self.post_message(message)
        except Exception:
            return

    def _show_log(self) -> None:
        manager = get_agent_runtime_manager()
        output = self._call_manager_with_agent_id(manager.log, self._selected_agent_id)
        self._replace_log(output, force_scroll_end=True)
        self._refresh_meta_panel()

    def _show_meta_log(self) -> None:
        manager = get_agent_runtime_manager()
        if not hasattr(manager, "meta_log"):
            self._set_meta_status("Meta log is unavailable in this runtime.")
            return
        output = manager.meta_log()
        self._append_log(output)
        self._refresh_meta_panel()

    def _start_timer(
        self,
        *,
        interval_seconds: float,
        callback,
    ) -> Optional[Timer]:
        app_obj = getattr(self, "app", None)
        if app_obj is None or not hasattr(app_obj, "set_interval"):
            return None
        try:
            return self.set_interval(interval_seconds, callback)
        except Exception:
            return None

    def _stop_timer(self, timer: Optional[Timer]) -> None:
        if timer is None:
            return
        try:
            timer.stop()
        except Exception:
            return

    def _initialize_refresh_controls(self) -> None:
        running_options = [
            ("0.5s", "0.5"),
            ("1.0s", "1.0"),
            ("2.0s", "2.0"),
            ("5.0s", "5.0"),
        ]
        interaction_options = [
            ("0.5s", "0.5"),
            ("1.0s", "1.0"),
            ("2.0s", "2.0"),
            ("3.0s", "3.0"),
        ]
        self._set_select_options(
            "#agents-running-refresh-interval-select",
            options=running_options,
            value=f"{self._running_refresh_interval_seconds:.1f}",
        )
        self._set_select_options(
            "#agents-interaction-refresh-interval-select",
            options=interaction_options,
            value=f"{self._interaction_poll_interval_seconds:.1f}",
        )
        self._update_refresh_controls()

    def _set_select_options(
        self,
        selector: str,
        *,
        options: list[tuple[str, str]],
        value: str,
    ) -> None:
        self._assert_ui_thread("_set_select_options")
        try:
            select_widget = self.query_one(selector, Select)
        except Exception:
            return
        try:
            select_widget.set_options(options)
            select_widget.value = value
        except Exception:
            return

    def _set_button_label(self, selector: str, label: str) -> None:
        self._assert_ui_thread("_set_button_label")
        try:
            button = self.query_one(selector, Button)
        except Exception:
            return
        try:
            button.label = label
        except Exception:
            return

    def _update_refresh_controls(self) -> None:
        self._set_button_label(
            "#agents-running-refresh-toggle-button",
            "Auto: On" if self._running_refresh_enabled else "Auto: Off",
        )
        self._set_button_label(
            "#agents-interaction-refresh-toggle-button",
            "Auto: On" if self._interaction_poll_enabled else "Auto: Off",
        )
        self._set_button_label(
            "#agents-log-follow-toggle-button",
            "Follow: On" if self._log_follow_selected else "Follow: Off",
        )

    def _restart_running_refresh_timer(self) -> None:
        self._stop_timer(self._running_refresh_timer)
        self._running_refresh_timer = None
        if not self._running_refresh_enabled:
            return
        self._running_refresh_timer = self._start_timer(
            interval_seconds=self._running_refresh_interval_seconds,
            callback=self._refresh_running_agents,
        )

    def _restart_interaction_poll_timer(self) -> None:
        self._stop_timer(self._interaction_poll_timer)
        self._interaction_poll_timer = None
        if not self._interaction_poll_enabled:
            return
        self._interaction_poll_timer = self._start_timer(
            interval_seconds=self._interaction_poll_interval_seconds,
            callback=self._refresh_interaction_panel,
        )

    def _toggle_running_refresh(self) -> None:
        self._running_refresh_enabled = not self._running_refresh_enabled
        self._restart_running_refresh_timer()
        self._update_refresh_controls()
        state = "enabled" if self._running_refresh_enabled else "disabled"
        self._set_status(
            (
                f"Running-agents refresh {state} "
                f"({self._format_interval_label(self._running_refresh_interval_seconds)})."
            )
        )

    def _toggle_interaction_refresh(self) -> None:
        self._interaction_poll_enabled = not self._interaction_poll_enabled
        self._restart_interaction_poll_timer()
        self._update_refresh_controls()
        state = "enabled" if self._interaction_poll_enabled else "disabled"
        self._set_interaction_status(
            (
                f"Interaction polling {state} "
                f"({self._format_interval_label(self._interaction_poll_interval_seconds)})."
            )
        )

    def _toggle_log_follow_mode(self) -> None:
        self._log_follow_selected = not self._log_follow_selected
        self._update_refresh_controls()
        if self._log_follow_selected:
            try:
                log_widget = self.query_one("#agents-log", RichLog)
                self._scroll_log_to_end(log_widget)
            except Exception:
                pass
        self._set_status(
            f"Follow selected agent log {'enabled' if self._log_follow_selected else 'disabled'}."
        )

    def _parse_refresh_interval(self, value: Any) -> Optional[float]:
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    def _format_interval_label(self, interval_seconds: float) -> str:
        return f"{float(interval_seconds):.1f}s"

    def _initialize_running_controls(self) -> None:
        try:
            running_table = self.query_one("#agents-running-table", DataTable)
        except Exception:
            return
        if len(getattr(running_table, "columns", {})) == 0:
            running_table.add_columns("ID", "Agent", "Topic", "Status", "Duration", "Unread")
        running_table.cursor_type = "row"
        running_table.zebra_stripes = True

    def _refresh_running_agents(self) -> None:
        try:
            running_table = self.query_one("#agents-running-table", DataTable)
            output_widget = self.query_one("#agents-running-output", Static)
        except Exception:
            return

        previous_selected = str(self._selected_agent_id or "").strip()
        manager = get_agent_runtime_manager()
        try:
            running = list(manager.list_running())
        except Exception as exc:
            running_table.clear(columns=False)
            self._running_agent_ids = []
            self._running_row_index = 0
            self._selected_agent_id = None
            self._unread_log_counts = {}
            output_widget.update(str(exc))
            return

        row_data: list[dict[str, Any]] = []
        current_running_agent_names: dict[str, str] = {}
        for row in running:
            agent_id = str(row.get("agent_id", "")).strip()
            if not agent_id:
                continue
            agent_name = str(row.get("agent_name", "")).strip() or "-"
            current_running_agent_names[agent_id] = agent_name
            topic = str(row.get("topic", "")).strip() or "-"
            status = str(row.get("status", "")).strip() or "-"
            duration_seconds = float(row.get("duration_seconds", 0.0) or 0.0)
            row_data.append(
                {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "topic": topic,
                    "status": status,
                    "duration_seconds": duration_seconds,
                }
            )

        self._running_agent_ids = [
            str(item.get("agent_id", "")).strip()
            for item in row_data
            if str(item.get("agent_id", "")).strip()
        ]

        current_running_agent_ids = set(self._running_agent_ids)
        self._emit_running_agent_notifications(
            manager=manager,
            current_running_agent_ids=current_running_agent_ids,
            current_running_agent_names=current_running_agent_names,
        )
        self._known_running_agent_ids = current_running_agent_ids
        self._known_running_agent_names = current_running_agent_names
        self._running_notifications_initialized = True
        self._unread_log_counts = {
            agent_id: max(0, self._coerce_int(count) or 0)
            for agent_id, count in self._unread_log_counts.items()
            if agent_id in current_running_agent_ids
        }

        if not self._running_agent_ids:
            running_table.clear(columns=False)
            self._running_row_index = 0
            self._selected_agent_id = None
            self._unread_log_counts = {}
            output_widget.update("No running agents.")
            return

        selected_index = 0
        if self._selected_agent_id and self._selected_agent_id in self._running_agent_ids:
            selected_index = self._running_agent_ids.index(self._selected_agent_id)
        else:
            selected_index = min(self._running_row_index, len(self._running_agent_ids) - 1)
            self._selected_agent_id = self._running_agent_ids[selected_index]

        selected_agent_id = str(self._selected_agent_id or "").strip()
        if selected_agent_id:
            self._clear_unread_for_agent(selected_agent_id)

        running_table.clear(columns=False)
        for item in row_data:
            agent_id = str(item.get("agent_id", "")).strip()
            if not agent_id:
                continue
            unread_count = max(0, self._coerce_int(self._unread_log_counts.get(agent_id)) or 0)
            if agent_id == selected_agent_id:
                unread_count = 0
            running_table.add_row(
                agent_id[:8],
                str(item.get("agent_name", "")),
                str(item.get("topic", "")),
                str(item.get("status", "")),
                self._format_duration(float(item.get("duration_seconds", 0.0) or 0.0)),
                str(unread_count),
            )

        self._running_row_index = selected_index
        try:
            running_table.move_cursor(row=selected_index, column=0)
        except Exception:
            pass

        output_widget.update(f"{len(self._running_agent_ids)} running agent(s).")
        if str(self._selected_agent_id or "").strip() != previous_selected:
            self._show_selected_agent_log()

    def _format_duration(self, duration_seconds: float) -> str:
        seconds = max(0, int(duration_seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _move_running_selection(self, delta: int) -> None:
        if not self._running_agent_ids:
            return
        row_index = min(
            max(0, self._running_row_index + int(delta)),
            len(self._running_agent_ids) - 1,
        )
        self._running_row_index = row_index
        try:
            running_table = self.query_one("#agents-running-table", DataTable)
            running_table.move_cursor(row=row_index, column=0)
        except Exception:
            pass
        self._select_running_agent_by_row(row_index)

    def _select_running_agent_by_row(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._running_agent_ids):
            return
        self._running_row_index = row_index
        self._selected_agent_id = self._running_agent_ids[row_index]
        self._clear_unread_for_agent(self._selected_agent_id)
        self._show_selected_agent_log()

    def _show_selected_agent_log(self) -> None:
        agent_id = self._selected_agent_id
        if not agent_id:
            return
        self._clear_unread_for_agent(agent_id)
        manager = get_agent_runtime_manager()
        output = self._call_manager_with_agent_id(manager.log, agent_id)
        self._replace_log(output)
        clear_stream = getattr(manager, "clear_log_stream", None)
        if callable(clear_stream):
            try:
                clear_stream(agent_id)
            except Exception:
                pass

    def _clear_unread_for_agent(self, agent_id: Optional[str]) -> None:
        normalized = str(agent_id or "").strip()
        if not normalized:
            return
        self._unread_log_counts[normalized] = 0

    def _clear_unread_counts(self, *, manual: bool = False) -> None:
        if not self._unread_log_counts:
            if manual:
                self._set_status("No unread log counters to clear.")
            return
        self._unread_log_counts = {agent_id: 0 for agent_id in self._unread_log_counts}
        if manual:
            self._set_status("Cleared unread log counters.")
        self._refresh_running_agents()

    def _call_manager_with_agent_id(self, method, agent_id: Optional[str]) -> str:
        if agent_id:
            return method(agent_id=agent_id)
        return method(agent_id=None)

    def _drain_log_streams(self) -> None:
        manager = get_agent_runtime_manager()
        drain = getattr(manager, "drain_log_stream", None)
        if not callable(drain):
            return

        selected_agent_id = str(self._selected_agent_id or "").strip()
        target_agent_ids: list[str] = []
        for agent_id in self._running_agent_ids:
            normalized = str(agent_id or "").strip()
            if normalized and normalized not in target_agent_ids:
                target_agent_ids.append(normalized)
        if selected_agent_id and selected_agent_id not in target_agent_ids:
            target_agent_ids.append(selected_agent_id)

        for agent_id in target_agent_ids:
            try:
                payload = drain(agent_id, max_entries=200)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            drained_agent_id = str(payload.get("agent_id", "")).strip()
            if drained_agent_id and drained_agent_id != agent_id:
                continue

            dropped_count = max(0, self._coerce_int(payload.get("dropped_count")) or 0)
            entries = payload.get("entries")
            if not isinstance(entries, list):
                entries = []

            if agent_id == selected_agent_id:
                if dropped_count > 0:
                    self._append_log(self._format_stream_drop_notice(dropped_count))
                for row in entries:
                    if not isinstance(row, dict):
                        continue
                    self._append_log(self._format_stream_log_entry(row))
                continue

            increment = dropped_count
            for row in entries:
                if isinstance(row, dict):
                    increment += 1
            if increment <= 0:
                continue
            self._unread_log_counts[agent_id] = (
                max(0, self._coerce_int(self._unread_log_counts.get(agent_id)) or 0) + increment
            )

    def _refresh_interaction_panel(self) -> None:
        try:
            panel = self.query_one("#agents-interaction-panel", Container)
            indicator = self.query_one("#agents-interaction-indicator", Static)
        except Exception:
            return

        manager = get_agent_runtime_manager()
        try:
            pending_rows = list(manager.get_pending_interactions())
        except Exception as exc:
            self._pending_interaction = None
            self._set_interaction_status(str(exc))
            indicator.update("Interaction unavailable")
            self._set_interaction_visual_state(panel=panel, indicator=indicator, pending_count=0)
            return

        pending_ids = {
            self._pending_interaction_key(row)
            for row in pending_rows
            if self._pending_interaction_key(row)
        }
        self._emit_interaction_notifications(pending_rows, pending_ids)
        self._known_pending_interaction_ids = pending_ids
        self._interaction_notifications_initialized = True

        pending_count = len(pending_rows)
        if pending_count == 0:
            self._pending_interaction = None
            self._set_interaction_status("No pending interaction requests.")
            self._set_interaction_request_details(None)
            self._set_interaction_controls(mode="none", has_pending=False)
            self._set_interaction_visual_state(panel=panel, indicator=indicator, pending_count=0)
            return

        selected = self._pick_pending_interaction(pending_rows)
        self._pending_interaction = selected
        self._set_interaction_status(
            (
                f"Pending {str(selected.get('request_type') or 'interaction')} request "
                f"for agent '{str(selected.get('agent_name') or '-').strip()}' "
                f"({str(selected.get('agent_id') or '')[:8]})."
            )
        )
        self._set_interaction_request_details(selected)
        self._set_interaction_controls(
            mode=self._interaction_mode(str(selected.get("request_type", "")).strip().lower()),
            has_pending=True,
        )
        self._set_interaction_visual_state(panel=panel, indicator=indicator, pending_count=pending_count)

        selected_id = str(selected.get("agent_id", "")).strip()
        if selected_id and selected_id in self._running_agent_ids and selected_id != self._selected_agent_id:
            self._selected_agent_id = selected_id
            self._running_row_index = self._running_agent_ids.index(selected_id)
            try:
                running_table = self.query_one("#agents-running-table", DataTable)
                running_table.move_cursor(row=self._running_row_index, column=0)
            except Exception:
                pass
            self._show_selected_agent_log()

    def _pick_pending_interaction(
        self,
        pending_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected_id = str(self._selected_agent_id or "").strip()
        if selected_id:
            for row in pending_rows:
                if str(row.get("agent_id", "")).strip() == selected_id:
                    return row
        return pending_rows[0]

    def _set_interaction_visual_state(
        self,
        *,
        panel: Container,
        indicator: Static,
        pending_count: int,
    ) -> None:
        if pending_count <= 0:
            indicator.update("No pending interaction")
            self._set_display(panel, visible=False)
            self._toggle_class(panel, "agents-interaction-panel-active", enabled=False)
            self._toggle_class(indicator, "agents-interaction-indicator-active", enabled=False)
            return
        indicator.update(f"Interaction pending ({pending_count})")
        self._set_display(panel, visible=True)
        self._toggle_class(panel, "agents-interaction-panel-active", enabled=True)
        self._toggle_class(indicator, "agents-interaction-indicator-active", enabled=True)

    def _set_interaction_request_details(
        self,
        interaction: Optional[dict[str, Any]],
    ) -> None:
        details = {
            "#agents-interaction-type": "Type",
            "#agents-interaction-agent": "Agent",
            "#agents-interaction-tool": "Tool",
            "#agents-interaction-risk": "Risk",
            "#agents-interaction-reason": "Reason",
            "#agents-interaction-args": "Args",
            "#agents-interaction-prompt": "Prompt",
        }
        if interaction is None:
            for selector, label in details.items():
                self._update_static(selector, f"{label}: -")
            return

        self._update_static(
            "#agents-interaction-type",
            f"Type: {str(interaction.get('request_type') or '-').strip()}",
        )
        self._update_static(
            "#agents-interaction-agent",
            (
                f"Agent: {str(interaction.get('agent_name') or '-').strip()} "
                f"({str(interaction.get('agent_id') or '')[:8]})"
            ),
        )
        self._update_static(
            "#agents-interaction-tool",
            f"Tool: {str(interaction.get('tool_name') or '-').strip()}",
        )
        self._update_static(
            "#agents-interaction-risk",
            f"Risk: {str(interaction.get('risk_level') or '-').strip()}",
        )
        self._update_static(
            "#agents-interaction-reason",
            f"Reason: {str(interaction.get('reason') or '-').strip()}",
        )
        self._update_static(
            "#agents-interaction-args",
            f"Args: {str(interaction.get('args_summary') or '-').strip()}",
        )
        self._update_static(
            "#agents-interaction-prompt",
            f"Prompt: {str(interaction.get('prompt') or '-').strip()}",
        )

    def _interaction_mode(self, request_type: str) -> str:
        if request_type in {"clarification", "feedback"}:
            return "reply"
        return "permission"

    def _set_interaction_controls(self, *, mode: str, has_pending: bool) -> None:
        is_permission = has_pending and mode == "permission"
        is_reply = has_pending and mode == "reply"
        self._set_widget_disabled("#agents-interaction-deny-input", disabled=not is_permission)
        self._set_widget_disabled("#agents-interaction-reply-input", disabled=not is_reply)
        self._set_widget_disabled("#agents-interaction-approve-button", disabled=not is_permission)
        self._set_widget_disabled("#agents-interaction-approve-session-button", disabled=not is_permission)
        self._set_widget_disabled("#agents-interaction-deny-button", disabled=not is_permission)
        self._set_widget_disabled("#agents-interaction-reply-button", disabled=not is_reply)

    def _approve_interaction(self) -> None:
        self._respond_to_interaction(decision="approve")

    def _approve_interaction_session(self) -> None:
        self._respond_to_interaction(decision="approve_session")

    def _deny_interaction(self) -> None:
        deny_reason = self._input_value("#agents-interaction-deny-input")
        self._respond_to_interaction(
            decision="deny",
            user_message=deny_reason,
        )

    def _reply_to_interaction(self) -> None:
        reply_message = self._input_value("#agents-interaction-reply-input")
        if not reply_message:
            self._set_interaction_status("Enter reply text before sending.")
            return
        self._respond_to_interaction(
            decision="reply",
            user_message=reply_message,
        )

    def _respond_to_interaction(self, *, decision: str, user_message: str = "") -> None:
        pending = self._pending_interaction
        if not pending:
            self._set_interaction_status("No pending interaction request.")
            return

        agent_id = str(pending.get("agent_id", "")).strip()
        if not agent_id:
            self._set_interaction_status("Pending interaction does not include an agent id.")
            return

        payload: dict[str, Any] = {"decision": decision}
        request_id = str(pending.get("request_id", "")).strip()
        if request_id:
            payload["request_id"] = request_id
        if user_message:
            payload["user_message"] = user_message

        manager = get_agent_runtime_manager()
        output = manager.respond_to_interaction(agent_id, payload)
        self._set_interaction_status(output.strip())
        self._append_log(output)
        if output.lower().startswith("posted interaction response"):
            self._clear_input("#agents-interaction-deny-input")
            self._clear_input("#agents-interaction-reply-input")

        self._refresh_interaction_panel()
        self._refresh_running_agents()

    def _set_interaction_status(self, message: str) -> None:
        self._update_static("#agents-interaction-status-output", message)

    def _update_static(self, selector: str, message: str) -> None:
        self._assert_ui_thread("_update_static")
        try:
            self.query_one(selector, Static).update(message)
        except Exception:
            return

    def _set_widget_disabled(self, selector: str, *, disabled: bool) -> None:
        self._assert_ui_thread("_set_widget_disabled")
        try:
            widget = self.query_one(selector)
        except Exception:
            return
        try:
            widget.disabled = bool(disabled)
        except Exception:
            return

    def _input_value(self, selector: str) -> str:
        try:
            widget = self.query_one(selector, Input)
        except Exception:
            return ""
        return str(widget.value or "").strip()

    def _clear_input(self, selector: str) -> None:
        try:
            widget = self.query_one(selector, Input)
        except Exception:
            return
        widget.value = ""

    def _set_display(self, widget, *, visible: bool) -> None:
        self._assert_ui_thread("_set_display")
        try:
            widget.display = bool(visible)
        except Exception:
            return

    def _toggle_class(self, widget, class_name: str, *, enabled: bool) -> None:
        self._assert_ui_thread("_toggle_class")
        try:
            if enabled:
                widget.add_class(class_name)
            else:
                widget.remove_class(class_name)
        except Exception:
            return

    def _initialize_meta_controls(self) -> None:
        try:
            task_table = self.query_one("#agents-meta-task-table", DataTable)
            run_table = self.query_one("#agents-meta-runs-table", DataTable)
        except Exception:
            return
        if len(getattr(task_table, "columns", {})) == 0:
            task_table.add_columns("Task", "Agent", "Status", "Depends On")
        if len(getattr(run_table, "columns", {})) == 0:
            run_table.add_columns("Task", "Sub-Agent", "Status", "Workspace", "Artifacts")
        task_table.cursor_type = "row"
        task_table.zebra_stripes = True
        run_table.cursor_type = "row"
        run_table.zebra_stripes = True

    def _refresh_meta_panel(self) -> None:
        try:
            task_table = self.query_one("#agents-meta-task-table", DataTable)
            run_table = self.query_one("#agents-meta-runs-table", DataTable)
            status_widget = self.query_one("#agents-meta-output", Static)
            artifacts_widget = self.query_one("#agents-meta-artifacts-output", Static)
        except Exception:
            return

        manager = get_agent_runtime_manager()
        snapshot_fn = getattr(manager, "get_meta_snapshot", None)
        if not callable(snapshot_fn):
            task_table.clear(columns=False)
            run_table.clear(columns=False)
            status_widget.update("Meta status is unavailable in this runtime.")
            artifacts_widget.update("No meta artifacts.")
            return

        try:
            snapshot = dict(snapshot_fn() or {})
        except Exception as exc:
            task_table.clear(columns=False)
            run_table.clear(columns=False)
            status_widget.update(f"Failed to load meta status: {exc}")
            artifacts_widget.update("No meta artifacts.")
            return

        task_table.clear(columns=False)
        run_table.clear(columns=False)
        if not bool(snapshot.get("available")):
            status_widget.update("No active meta-agent.")
            artifacts_widget.update("No meta artifacts.")
            return

        tasks = list(snapshot.get("tasks", []))
        for task in tasks:
            depends_on = ",".join(str(item) for item in task.get("depends_on", []))
            task_table.add_row(
                str(task.get("id", "")),
                str(task.get("agent_name", "")),
                str(task.get("status", "")),
                depends_on or "-",
            )

        task_runs = list(snapshot.get("task_runs", []))
        for run in task_runs:
            run_table.add_row(
                str(run.get("task_id", "")),
                str(run.get("agent_name", "")),
                str(run.get("status", "")),
                str(run.get("sub_agent_dir", "")),
                str(len(run.get("artifacts", []))),
            )

        completed = sum(1 for task in tasks if str(task.get("status")) == "completed")
        failed = sum(1 for task in tasks if str(task.get("status")) == "failed")
        status_widget.update(
            (
                f"Meta status={snapshot.get('status')} | "
                f"thread_alive={snapshot.get('thread_alive')} | "
                f"tasks={len(tasks)} completed={completed} failed={failed}"
            )
        )

        artifact_lines = []
        final_result_path = str(snapshot.get("final_result_path", "")).strip()
        if final_result_path:
            artifact_lines.append(f"final_result: {final_result_path}")
        meta_log_path = str(snapshot.get("meta_log_path", "")).strip()
        if meta_log_path:
            artifact_lines.append(f"meta_log: {meta_log_path}")
        for value in list(snapshot.get("artifacts", [])):
            artifact_lines.append(str(value))
        if not artifact_lines:
            artifacts_widget.update("No meta artifacts recorded yet.")
            return
        artifacts_widget.update("\n".join(artifact_lines[:25]))

    def _initialize_schedule_controls(self) -> None:
        try:
            schedule_table = self.query_one("#agents-schedule-table", DataTable)
        except Exception:
            return
        if len(getattr(schedule_table, "columns", {})) == 0:
            schedule_table.add_columns(
                "ID",
                "Agent",
                "Interval",
                "Enabled",
                "Status",
                "Next Run",
                "Last Run",
            )
        schedule_table.cursor_type = "row"
        schedule_table.zebra_stripes = True
        try:
            concurrency_select = self.query_one("#agents-schedule-concurrency-select", Select)
            concurrency_select.value = "skip"
            confirmation_select = self.query_one("#agents-schedule-confirmation-select", Select)
            confirmation_select.value = "auto"
        except Exception:
            return

    def _refresh_memory_candidates(self) -> None:
        try:
            candidate_select = self.query_one("#agents-memory-select", Select)
            output_widget = self.query_one("#agents-memory-output", Static)
        except Exception:
            return

        manager = get_agent_runtime_manager()
        try:
            candidates = manager.get_memory_candidates(
                self.app,
                status="pending",
                limit=200,
            )
        except Exception as exc:
            candidate_select.set_options([])
            output_widget.update(str(exc))
            return

        options = []
        for candidate in candidates:
            memory = candidate.memory
            label = (
                f"{candidate.id[:12]} | {memory.key} "
                f"({memory.type}/{memory.scope})"
            )
            options.append((label, candidate.id))

        candidate_select.set_options(options)
        if options:
            candidate_select.value = options[0][1]
            output_widget.update(f"Loaded {len(options)} pending memory candidate(s).")
            return
        output_widget.update("No pending memory candidates found.")

    def _refresh_schedule_entries(self) -> None:
        try:
            schedule_table = self.query_one("#agents-schedule-table", DataTable)
            output_widget = self.query_one("#agents-schedule-output", Static)
        except Exception:
            return

        manager = get_agent_runtime_manager()
        try:
            schedules = manager.list_schedules(self.app)
        except Exception as exc:
            schedule_table.clear(columns=False)
            self._schedule_ids = []
            self._schedule_row_index = 0
            output_widget.update(str(exc))
            return

        schedule_table.clear(columns=False)
        self._schedule_ids = []
        current_schedule_last_run_by_id: dict[str, str] = {}
        current_schedule_agent_by_id: dict[str, str] = {}
        for row in schedules:
            schedule_id = str(row["id"])
            agent_name = str(row["agent_name"])
            enabled = "yes" if row.get("enabled") else "no"
            status = str(row.get("last_status", "idle"))
            next_run = str(row.get("next_run_at") or "-")
            last_run = str(row.get("last_run_at") or "-")
            normalized_last_run = "" if last_run == "-" else last_run
            current_schedule_last_run_by_id[schedule_id] = normalized_last_run
            current_schedule_agent_by_id[schedule_id] = agent_name
            schedule_table.add_row(
                schedule_id,
                agent_name,
                str(row["interval"]),
                enabled,
                status,
                next_run,
                last_run,
            )
            self._schedule_ids.append(schedule_id)

        self._emit_schedule_trigger_notifications(
            current_schedule_last_run_by_id=current_schedule_last_run_by_id,
            current_schedule_agent_by_id=current_schedule_agent_by_id,
        )
        self._schedule_last_run_by_id = current_schedule_last_run_by_id
        self._schedule_agent_by_id = current_schedule_agent_by_id
        self._schedule_notifications_initialized = True

        if self._schedule_ids:
            self._schedule_row_index = min(self._schedule_row_index, len(self._schedule_ids) - 1)
            try:
                schedule_table.move_cursor(row=self._schedule_row_index, column=0)
            except Exception:
                pass
            output_widget.update(f"Loaded {len(self._schedule_ids)} schedule(s).")
            return
        output_widget.update("No schedules configured.")

    def _add_schedule(self) -> None:
        agent_input = self.query_one("#agents-schedule-agent-input", Input)
        interval_input = self.query_one("#agents-schedule-interval-input", Input)
        params_input = self.query_one("#agents-schedule-params-input", Input)
        concurrency_select = self.query_one("#agents-schedule-concurrency-select", Select)
        confirmation_select = self.query_one("#agents-schedule-confirmation-select", Select)

        agent_name = agent_input.value.strip()
        interval = interval_input.value.strip()
        if not agent_name:
            self._set_schedule_status("Enter an agent name before adding a schedule.")
            return
        if not interval:
            self._set_schedule_status("Enter an interval before adding a schedule.")
            return

        params_text = params_input.value.strip()
        params: dict = {}
        if params_text:
            try:
                parsed = json.loads(params_text)
            except json.JSONDecodeError as exc:
                self._set_schedule_status(f"Invalid params JSON: {exc}")
                return
            if not isinstance(parsed, dict):
                self._set_schedule_status("Params JSON must decode to an object.")
                return
            params = parsed

        concurrency = (
            str(concurrency_select.value).strip().lower()
            if isinstance(concurrency_select.value, str)
            else "skip"
        )
        confirmation = (
            str(confirmation_select.value).strip().lower()
            if isinstance(confirmation_select.value, str)
            else "auto"
        )

        manager = get_agent_runtime_manager()
        try:
            output = manager.add_schedule(
                self.app,
                agent_name=agent_name,
                interval=interval,
                params=params,
                concurrency_policy=concurrency,
                confirmation_mode=confirmation,
            )
        except Exception as exc:
            output = f"Failed to add schedule: {exc}\n"
        self._refresh_schedule_entries()
        self._set_schedule_status(output.strip())
        self._append_log(output)

    def _remove_selected_schedule(self) -> None:
        schedule_id = self._selected_schedule_id()
        if not schedule_id:
            self._set_schedule_status("Select a schedule first.")
            return
        manager = get_agent_runtime_manager()
        try:
            output = manager.remove_schedule(self.app, schedule_id)
        except Exception as exc:
            output = f"Failed to remove schedule: {exc}\n"
        self._refresh_schedule_entries()
        self._set_schedule_status(output.strip())
        self._append_log(output)

    def _enable_selected_schedule(self) -> None:
        schedule_id = self._selected_schedule_id()
        if not schedule_id:
            self._set_schedule_status("Select a schedule first.")
            return
        manager = get_agent_runtime_manager()
        try:
            output = manager.set_schedule_enabled(self.app, schedule_id, enabled=True)
        except Exception as exc:
            output = f"Failed to enable schedule: {exc}\n"
        self._refresh_schedule_entries()
        self._set_schedule_status(output.strip())
        self._append_log(output)

    def _disable_selected_schedule(self) -> None:
        schedule_id = self._selected_schedule_id()
        if not schedule_id:
            self._set_schedule_status("Select a schedule first.")
            return
        manager = get_agent_runtime_manager()
        try:
            output = manager.set_schedule_enabled(self.app, schedule_id, enabled=False)
        except Exception as exc:
            output = f"Failed to disable schedule: {exc}\n"
        self._refresh_schedule_entries()
        self._set_schedule_status(output.strip())
        self._append_log(output)

    def _show_schedule_status(self) -> None:
        manager = get_agent_runtime_manager()
        try:
            output = manager.format_schedule_status(self.app)
        except Exception as exc:
            output = f"Failed to load schedule status: {exc}\n"
        self._set_schedule_status(output.strip())
        self._append_log(output)

    def _approve_memory_candidate(self) -> None:
        candidate_id = self._selected_memory_candidate_id()
        if not candidate_id:
            self._set_memory_status("Select a memory candidate first.")
            return
        manager = get_agent_runtime_manager()
        output = manager.promote_memory_candidate(self.app, candidate_id)
        self._refresh_memory_candidates()
        self._set_memory_status(output.strip())
        self._append_log(output)

    def _reject_memory_candidate(self) -> None:
        candidate_id = self._selected_memory_candidate_id()
        if not candidate_id:
            self._set_memory_status("Select a memory candidate first.")
            return
        manager = get_agent_runtime_manager()
        output = manager.reject_memory_candidate(self.app, candidate_id)
        self._refresh_memory_candidates()
        self._set_memory_status(output.strip())
        self._append_log(output)

    def _edit_memory_candidate_ttl(self) -> None:
        candidate_id = self._selected_memory_candidate_id()
        if not candidate_id:
            self._set_memory_status("Select a memory candidate first.")
            return

        ttl_input = self.query_one("#agents-memory-ttl-input", Input)
        ttl_text = ttl_input.value.strip()
        if not ttl_text:
            self._set_memory_status("Enter TTL days before editing.")
            return
        try:
            ttl_days = int(ttl_text)
        except ValueError:
            self._set_memory_status("TTL days must be an integer.")
            return
        if ttl_days < 1:
            self._set_memory_status("TTL days must be >= 1.")
            return

        manager = get_agent_runtime_manager()
        output = manager.edit_memory_candidate_ttl(self.app, candidate_id, ttl_days)
        self._refresh_memory_candidates()
        self._set_memory_status(output.strip())
        self._append_log(output)

    def _selected_memory_candidate_id(self) -> str:
        candidate_select = self.query_one("#agents-memory-select", Select)
        value = candidate_select.value
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _selected_schedule_id(self) -> str:
        if not self._schedule_ids:
            return ""
        try:
            schedule_table = self.query_one("#agents-schedule-table", DataTable)
            coordinate = getattr(schedule_table, "cursor_coordinate", None)
            if coordinate is not None and hasattr(coordinate, "row"):
                row_index = int(coordinate.row)
            else:
                row_index = int(self._schedule_row_index)
        except Exception:
            row_index = int(self._schedule_row_index)

        if row_index < 0 or row_index >= len(self._schedule_ids):
            return ""
        self._schedule_row_index = row_index
        return self._schedule_ids[row_index].strip()

    def _assert_ui_thread(self, action: str) -> bool:
        checker = getattr(getattr(self, "app", None), "assert_ui_thread", None)
        if not callable(checker):
            return True
        try:
            return bool(checker(action=action))
        except TypeError:
            return bool(checker())

    def _set_status(self, message: str) -> None:
        self._update_static("#agents-status-output", message)

    def _set_memory_status(self, message: str) -> None:
        self._update_static("#agents-memory-output", message)

    def _set_schedule_status(self, message: str) -> None:
        self._update_static("#agents-schedule-output", message)

    def _set_meta_status(self, message: str) -> None:
        self._update_static("#agents-meta-output", message)

    def _notify_event(
        self,
        message: str,
        *,
        severity: str = "info",
        timeout: Optional[float] = None,
    ) -> None:
        notify = getattr(self.app, "notify_event", None)
        if not callable(notify):
            return
        try:
            kwargs: dict[str, Any] = {"severity": severity}
            if timeout is not None:
                kwargs["timeout"] = timeout
            notify(message, **kwargs)
        except Exception:
            logger.exception("Failed to emit agents notification")

    def _emit_running_agent_notifications(
        self,
        *,
        manager: Any,
        current_running_agent_ids: set[str],
        current_running_agent_names: dict[str, str],
    ) -> None:
        if not self._running_notifications_initialized:
            return

        started_ids = sorted(current_running_agent_ids - self._known_running_agent_ids)
        for agent_id in started_ids:
            agent_name = current_running_agent_names.get(agent_id, "agent")
            self._notify_event(f"Agent '{agent_name}' started.", severity="info")

        completed_ids = sorted(self._known_running_agent_ids - current_running_agent_ids)
        for agent_id in completed_ids:
            fallback_name = self._known_running_agent_names.get(agent_id, "agent")
            status_output = ""
            try:
                status_output = str(manager.status(agent_id=agent_id) or "")
            except Exception:
                status_output = ""
            status_value = self._extract_agent_status(status_output)
            agent_name = self._extract_agent_name(status_output) or fallback_name
            if status_value == "failed":
                self._notify_event(f"Agent '{agent_name}' failed.", severity="error")
                continue
            if status_value and status_value != "completed":
                self._notify_event(
                    f"Agent '{agent_name}' completed ({status_value}).",
                    severity="info",
                )
                continue
            self._notify_event(f"Agent '{agent_name}' completed.", severity="info")

    def _emit_interaction_notifications(
        self,
        pending_rows: list[dict[str, Any]],
        pending_ids: set[str],
    ) -> None:
        if not self._interaction_notifications_initialized:
            return
        new_ids = pending_ids - self._known_pending_interaction_ids
        if not new_ids:
            return
        for row in pending_rows:
            row_id = self._pending_interaction_key(row)
            if row_id not in new_ids:
                continue
            agent_name = str(row.get("agent_name") or "agent").strip() or "agent"
            request_type = str(row.get("request_type") or "interaction").strip() or "interaction"
            self._notify_event(
                f"Agent '{agent_name}' is waiting for user interaction ({request_type}).",
                severity="warning",
                timeout=10,
            )

    def _emit_schedule_trigger_notifications(
        self,
        *,
        current_schedule_last_run_by_id: dict[str, str],
        current_schedule_agent_by_id: dict[str, str],
    ) -> None:
        if not self._schedule_notifications_initialized:
            return
        for schedule_id, last_run_at in current_schedule_last_run_by_id.items():
            previous_last_run_at = self._schedule_last_run_by_id.get(schedule_id, "")
            if not last_run_at or last_run_at == previous_last_run_at:
                continue
            agent_name = (
                current_schedule_agent_by_id.get(schedule_id)
                or self._schedule_agent_by_id.get(schedule_id)
                or "agent"
            )
            self._notify_event(
                f"Schedule '{schedule_id}' triggered for agent '{agent_name}'.",
                severity="info",
            )

    def _pending_interaction_key(self, row: dict[str, Any]) -> str:
        request_id = str(row.get("request_id", "")).strip()
        if request_id:
            return request_id
        agent_id = str(row.get("agent_id", "")).strip()
        request_type = str(row.get("request_type", "")).strip()
        return f"{agent_id}:{request_type}"

    def _extract_agent_status(self, status_output: str) -> str:
        return self._extract_value_from_status_output(
            status_output,
            key="status",
            normalize=True,
        )

    def _extract_agent_name(self, status_output: str) -> str:
        return self._extract_value_from_status_output(
            status_output,
            key="agent",
            normalize=False,
        )

    def _extract_value_from_status_output(
        self,
        status_output: str,
        *,
        key: str,
        normalize: bool,
    ) -> str:
        prefix = f"{key.strip().lower()}:"
        for line in str(status_output or "").splitlines():
            normalized = str(line).strip()
            if not normalized.lower().startswith(prefix):
                continue
            value = normalized.split(":", 1)[1].strip()
            return value.lower() if normalize else value
        return ""

    def _append_log(self, message: Any) -> None:
        if message is None:
            return
        if isinstance(message, str) and not message:
            return
        self._write_log(message, replace=False)

    def _replace_log(self, message: Any, *, force_scroll_end: bool = False) -> None:
        self._write_log(
            message,
            replace=True,
            force_scroll_end=force_scroll_end,
        )

    def _write_log(
        self,
        message: Any,
        *,
        replace: bool,
        force_scroll_end: bool = False,
    ) -> None:
        self._assert_ui_thread("_write_log")
        if message is None:
            return
        if isinstance(message, str) and not message:
            return
        log_widget = self.query_one("#agents-log", RichLog)
        should_scroll_end = (
            force_scroll_end
            or self._log_follow_selected
            or self._is_log_scrolled_to_bottom(log_widget)
        )
        if replace:
            clear_fn = getattr(log_widget, "clear", None)
            if callable(clear_fn):
                try:
                    clear_fn()
                except Exception:
                    pass
        if isinstance(message, str):
            entry = message.rstrip() + "\n"
        else:
            entry = message
        write_fn = getattr(log_widget, "write", None)
        if not callable(write_fn):
            return
        try:
            write_fn(entry, scroll_end=should_scroll_end)
        except TypeError:
            write_fn(entry)
        if force_scroll_end:
            self._scroll_log_to_end(log_widget)

    def _scroll_log_to_end(self, log_widget: RichLog) -> None:
        scroll_end_fn = getattr(log_widget, "scroll_end", None)
        if not callable(scroll_end_fn):
            return
        try:
            scroll_end_fn(animate=False, immediate=True, x_axis=False, y_axis=True)
        except TypeError:
            try:
                scroll_end_fn(animate=False, immediate=True)
            except TypeError:
                try:
                    scroll_end_fn()
                except Exception:
                    return
        except Exception:
            return

    def _is_log_scrolled_to_bottom(self, log_widget: RichLog) -> bool:
        at_end_attr = getattr(log_widget, "is_vertical_scroll_end", None)
        if callable(at_end_attr):
            try:
                return bool(at_end_attr())
            except Exception:
                pass
        if isinstance(at_end_attr, bool):
            return at_end_attr
        scroll_y = self._coerce_float(getattr(log_widget, "scroll_y", None))
        max_scroll_y = self._coerce_float(getattr(log_widget, "max_scroll_y", None))
        if scroll_y is not None and max_scroll_y is not None:
            return scroll_y >= max_scroll_y
        return True

    def _format_stream_log_entry(self, row: dict[str, Any]) -> Text:
        actor = str(row.get("actor", "")).strip().lower()
        prefix_label, prefix_style = self._stream_prefix(actor)
        content = str(row.get("content", "")).strip()
        action = str(row.get("action", "")).strip()
        args_summary = self._summarize_stream_args(row.get("action_arguments"))
        summary = self._summarize_stream_content(content)

        if actor == "tool":
            action_name = action or "tool"
            if args_summary:
                body = f"{action_name}({args_summary}) -> {summary}"
            else:
                body = f"{action_name} -> {summary}"
        elif actor == "llm":
            if action:
                body = f"{summary} (action={action})"
            else:
                body = summary
        else:
            body = summary

        line = Text(f"[{prefix_label}] ", style=prefix_style)
        line.append(body)
        return line

    def _format_stream_drop_notice(self, dropped_count: int) -> Text:
        line = Text("[STREAM] ", style="bold yellow")
        line.append(
            f"Dropped {dropped_count} log entr{'y' if dropped_count == 1 else 'ies'} due to queue pressure.",
            style="yellow",
        )
        return line

    def _stream_prefix(self, actor: str) -> tuple[str, str]:
        normalized = str(actor or "").strip().lower()
        if normalized == "llm":
            return "LLM", "bright_cyan"
        if normalized == "tool":
            return "TOOL", "bright_magenta"
        if normalized == "user":
            return "USER", "bright_yellow"
        if normalized == "agent":
            return "AGENT", "bright_green"
        return "EVENT", "grey70"

    def _summarize_stream_args(self, value: Any) -> str:
        if value is None:
            return ""
        if value == "":
            return ""
        if isinstance(value, dict) and not value:
            return ""
        try:
            if isinstance(value, str):
                text = value
            else:
                text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except Exception:
            text = str(value)
        text = text.replace("\n", " ").strip()
        if len(text) > 100:
            return text[:97].rstrip() + "..."
        return text

    def _summarize_stream_content(self, content: str) -> str:
        text = str(content or "").replace("\n", " ").strip()
        if not text:
            return "(empty)"
        if len(text) > 180:
            return text[:177].rstrip() + "..."
        return text

    def _coerce_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
