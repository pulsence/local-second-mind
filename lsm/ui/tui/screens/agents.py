"""
Agents screen for launching and monitoring agent runs.
"""

from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Input, Button, Select, RichLog, DataTable
from textual.widget import Widget

from lsm.agents.factory import AgentRegistry
from lsm.logging import get_logger
from lsm.ui.shell.commands.agents import get_agent_runtime_manager

logger = get_logger(__name__)


class AgentsScreen(Widget):
    """
    UI surface for starting and controlling agents.
    """

    BINDINGS = [
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._schedule_ids: list[str] = []
        self._schedule_row_index: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="agents-layout"):
            with Horizontal(id="agents-top"):
                with Vertical(id="agents-left"):
                    with Container(id="agents-control-panel"):
                        yield Static("Agents", classes="agents-section-title")
                        yield Static("Agent", classes="agents-label")
                        yield Select([], id="agents-select")
                        yield Static("Topic", classes="agents-label")
                        yield Input(
                            placeholder="Research topic",
                            id="agents-topic-input",
                        )
                        with Horizontal(id="agents-buttons"):
                            yield Button("Start", id="agents-start-button", variant="primary")
                            yield Button("Status", id="agents-status-button")
                            yield Button("Pause", id="agents-pause-button")
                            yield Button("Resume", id="agents-resume-button")
                            yield Button("Stop", id="agents-stop-button", variant="error")
                            yield Button("Log", id="agents-log-button")
                    with Container(id="agents-status-panel"):
                        yield Static("Status", classes="agents-section-title")
                        yield Static("No active agent.", id="agents-status-output", markup=False)
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
                        with Horizontal(id="agents-schedule-buttons"):
                            yield Button("Add", id="agents-schedule-add-button", variant="primary")
                            yield Button("Remove", id="agents-schedule-remove-button", variant="error")
                            yield Button("Refresh", id="agents-schedule-refresh-button")
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
                        with Horizontal(id="agents-memory-buttons"):
                            yield Button("Refresh", id="agents-memory-refresh-button")
                            yield Button("Approve", id="agents-memory-approve-button")
                            yield Button("Reject", id="agents-memory-reject-button")
                            yield Button("Edit TTL", id="agents-memory-ttl-button")
                        yield Static(
                            "No memory candidates loaded.",
                            id="agents-memory-output",
                            markup=False,
                        )

                with Container(id="agents-log-panel"):
                    yield Static("Agent Log", classes="agents-section-title")
                    with ScrollableContainer(id="agents-log-scroll"):
                        yield RichLog(id="agents-log", auto_scroll=True, wrap=True)

    def on_mount(self) -> None:
        """Initialize agent select options and focus."""
        self._refresh_agent_options()
        self._initialize_meta_controls()
        self._refresh_meta_panel()
        self._initialize_schedule_controls()
        self._refresh_schedule_entries()
        self._refresh_memory_candidates()
        self._focus_default_input()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle control button presses."""
        button_id = event.button.id or ""
        if button_id == "agents-start-button":
            self._start_agent()
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

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = getattr(event, "data_table", None)
        if table is None or table.id != "agents-schedule-table":
            return
        row_index = getattr(event, "cursor_row", None)
        if isinstance(row_index, int) and row_index >= 0:
            self._schedule_row_index = row_index

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
        self._refresh_meta_panel()

    def _show_status(self) -> None:
        manager = get_agent_runtime_manager()
        output = manager.status()
        self._set_status(output.strip())
        self._refresh_meta_panel()

    def _run_control_action(self, action: str) -> None:
        manager = get_agent_runtime_manager()
        if action == "pause":
            output = manager.pause()
        elif action == "resume":
            output = manager.resume()
        elif action == "stop":
            output = manager.stop()
        else:
            output = "Unsupported control action."
        self._set_status(output.strip())
        self._append_log(output)
        self._refresh_meta_panel()

    def _show_log(self) -> None:
        manager = get_agent_runtime_manager()
        output = manager.log()
        self._append_log(output)
        self._refresh_meta_panel()

    def _show_meta_log(self) -> None:
        manager = get_agent_runtime_manager()
        if not hasattr(manager, "meta_log"):
            self._set_meta_status("Meta log is unavailable in this runtime.")
            return
        output = manager.meta_log()
        self._append_log(output)
        self._refresh_meta_panel()

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
        for row in schedules:
            enabled = "yes" if row.get("enabled") else "no"
            status = str(row.get("last_status", "idle"))
            next_run = str(row.get("next_run_at") or "-")
            last_run = str(row.get("last_run_at") or "-")
            schedule_table.add_row(
                str(row["id"]),
                str(row["agent_name"]),
                str(row["interval"]),
                enabled,
                status,
                next_run,
                last_run,
            )
            self._schedule_ids.append(str(row["id"]))

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

    def _set_status(self, message: str) -> None:
        self.query_one("#agents-status-output", Static).update(message)

    def _set_memory_status(self, message: str) -> None:
        self.query_one("#agents-memory-output", Static).update(message)

    def _set_schedule_status(self, message: str) -> None:
        self.query_one("#agents-schedule-output", Static).update(message)

    def _set_meta_status(self, message: str) -> None:
        try:
            self.query_one("#agents-meta-output", Static).update(message)
        except Exception:
            return

    def _append_log(self, message: str) -> None:
        if not message:
            return
        log_widget = self.query_one("#agents-log", RichLog)
        log_widget.write(message.rstrip() + "\n")
