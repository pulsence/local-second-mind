"""
Agents screen for launching and monitoring agent runs.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Input, Button, Select, RichLog
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

    def _show_status(self) -> None:
        manager = get_agent_runtime_manager()
        output = manager.status()
        self._set_status(output.strip())

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

    def _show_log(self) -> None:
        manager = get_agent_runtime_manager()
        output = manager.log()
        self._append_log(output)

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

    def _set_status(self, message: str) -> None:
        self.query_one("#agents-status-output", Static).update(message)

    def _set_memory_status(self, message: str) -> None:
        self.query_one("#agents-memory-output", Static).update(message)

    def _append_log(self, message: str) -> None:
        if not message:
            return
        log_widget = self.query_one("#agents-log", RichLog)
        log_widget.write(message.rstrip() + "\n")
