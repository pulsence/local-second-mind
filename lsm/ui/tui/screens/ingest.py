"""
Ingest screen for LSM TUI.

Provides the document ingestion interface with:
- File tree browser
- Build progress display
- Stats panel
- Command input for ingest operations with history and autocomplete
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, Deque, List, Callable
from collections import deque
import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, DirectoryTree, ProgressBar, Tree, TabbedContent
from textual.widget import Widget
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.tui.screens.base import ManagedScreenMixin
from lsm.ui.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.ui.tui.completions import create_completer
from lsm.ui.tui.commands.ingest import (
    CommandResult,
    handle_command as handle_ingest_command,
    run_build,
    run_tag,
    run_wipe,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class IngestScreen(ManagedScreenMixin, Widget):
    """
    Document ingestion interface screen.

    Provides:
    - Directory tree for browsing indexed files
    - Stats display panel
    - Progress bar for build operations
    - Command input for ingest commands
    """

    BINDINGS = [
        Binding("ctrl+b", "run_build", "Build", show=True),
        Binding("ctrl+t", "run_tagging", "Tag", show=True),
        Binding("ctrl+shift+r", "refresh_stats", "Refresh", show=True),
        Binding("escape", "clear_input", "Clear", show=False),
    ]

    # Reactive state
    is_building: reactive[bool] = reactive(False)
    build_progress: reactive[float] = reactive(0.0)
    chunk_count: reactive[int] = reactive(0)
    file_count: reactive[int] = reactive(0)
    _INGEST_COMMAND_WORKER_TIMEOUT_SECONDS = 45.0
    _INGEST_STATS_WORKER_TIMEOUT_SECONDS = 15.0

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the ingest screen."""
        super().__init__(*args, **kwargs)
        self._completer = create_completer("ingest")
        self._pending_command: Optional[str] = None
        self._pending_responses: List[str] = []
        self._stats_initialized = False

    def compose(self) -> ComposeResult:
        """Compose the ingest screen layout."""
        with ScrollableContainer(id="ingest-layout"):
            # Left pane: Directory tree
            with Vertical(classes="ingest-tree-container"):
                yield Static("Indexed Files", classes="stats-title")
                yield DirectoryTree(".", id="file-tree")

            # Right pane: Stats and controls
            with Vertical(id="ingest-right-pane"):
                # Top info panel
                with Vertical(id="ingest-info-container"):
                    with Container(classes="stats-panel"):
                        yield Static("Collection Statistics", classes="stats-title")
                        yield Static(
                            "Loading stats...",
                            id="stats-content",
                        )

                    yield Static(
                        "Common Commands\n"
                        "/stats   /explore   /build   /tag   /wipe",
                        id="ingest-common-commands",
                    )

                    # General command status/progress bar
                    yield ProgressBar(
                        id="command-progress",
                        show_eta=False,
                    )
                    yield Static(
                        "Ready.",
                        id="ingest-command-status",
                    )
                    yield Static(
                        "Selection: none",
                        id="ingest-selection-stats",
                    )

                # Results/output area
                with ScrollableContainer(id="ingest-output-container"):
                    yield Static(
                        "Ingest Commands:\n\n"
                        "/build [--force]  - Run ingest pipeline\n"
                        "/tag [--max N]    - Run AI tagging\n"
                        "/stats            - Show detailed statistics\n"
                        "/explore [query]  - Browse indexed files\n"
                        "/wipe             - Clear collection\n\n"
                        "Use Ctrl+B to build, Ctrl+T to tag, Ctrl+Shift+R to refresh.",
                        id="ingest-output",
                    )
                    yield Tree("Explore Results", id="ingest-explore-tree")

        # Command input at bottom with history and autocomplete
        yield CommandInput(
            placeholder="Enter ingest command (e.g., /build, /stats, /explore)...",
            completer=self._completer,
            id="ingest-command-input",
        )

    def on_mount(self) -> None:
        """Handle screen mount."""
        logger.debug("Ingest screen mounted")
        # Hide explore tree until needed
        self.query_one("#ingest-explore-tree", Tree).styles.display = "none"
        self._set_command_status("Ready.")
        if getattr(self.app, "current_context", None) == "ingest":
            self._activate_ingest_context()

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Refresh ingest stats when the ingest tab becomes active."""
        tab_id = event.tab.id
        if not tab_id:
            return
        context = tab_id.replace("-tab", "")
        if context == "ingest":
            self._activate_ingest_context()

    def _activate_ingest_context(self) -> None:
        """Perform ingest-tab-only initialization when tab is active."""
        if not self._stats_initialized:
            self._start_managed_worker(
                worker_key="ingest-refresh-stats",
                work_factory=self._refresh_stats,
                timeout_s=self._INGEST_STATS_WORKER_TIMEOUT_SECONDS,
                exclusive=True,
            )
            self._stats_initialized = True
        self.query_one("#ingest-command-input", CommandInput).focus()

    def on_unmount(self) -> None:
        self._cancel_managed_workers(reason="ingest-unmount")
        self._cancel_managed_timers(reason="ingest-unmount")

    async def on_command_submitted(self, event: CommandSubmitted) -> None:
        """Handle command input submission from CommandInput widget."""
        command = event.command.strip()
        if not command:
            return

        self._start_managed_worker(
            worker_key="ingest-command",
            work_factory=lambda: self._process_command(command),
            timeout_s=self._INGEST_COMMAND_WORKER_TIMEOUT_SECONDS,
            exclusive=True,
        )

    async def _process_command(self, command: str) -> None:
        """
        Process an ingest command.

        Args:
            command: Command string
        """
        output_widget = self.query_one("#ingest-output", Static)
        if self._pending_command:
            self._pending_responses.append(command)
            command_to_run = self._pending_command
        else:
            self._pending_responses = []
            command_to_run = command

        self._set_output_mode("text")
        if not self._pending_command:
            self._update_output_text(f"Running {command_to_run}...")
            self._set_command_status(f"Running: {command_to_run}")
            self._update_command_progress(None, None)

        output, prompt = await self._run_ingest_command(command_to_run, self._pending_responses)
        if output:
            output_widget.update(output)

        if prompt:
            self._pending_command = command_to_run
            prefix = (output + "\n") if output else ""
            output_widget.update(prefix + prompt)
            self._set_command_status("Waiting for input...")
            return

        self._pending_command = None
        self._pending_responses = []
        self._set_command_status("Ready.")
        self._update_command_progress(None, None)
        self._notify_build_result(command_to_run, output)

    def _set_output_mode(self, mode: str) -> None:
        """Toggle output widgets between text and tree."""
        output_widget = self.query_one("#ingest-output", Static)
        tree_widget = self.query_one("#ingest-explore-tree", Tree)
        if mode == "tree":
            output_widget.styles.display = "none"
            tree_widget.styles.display = "block"
        else:
            tree_widget.styles.display = "none"
            output_widget.styles.display = "block"

    def _update_output_text(self, text: str) -> None:
        """Update the output text area."""
        self._set_output_mode("text")
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update(text)

    def _set_command_status(self, message: str) -> None:
        """Update the command status line."""
        status_widget = self.query_one("#ingest-command-status", Static)
        status_widget.update(message)

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
            kwargs: Dict[str, Any] = {"severity": severity}
            if timeout is not None:
                kwargs["timeout"] = timeout
            notify(message, **kwargs)
        except Exception:
            logger.exception("Failed to emit ingest notification")

    def _notify_build_result(self, command: str, output: str) -> None:
        normalized_command = str(command or "").strip().lower()
        if not normalized_command.startswith("/build"):
            return

        normalized_output = str(output or "").lower()
        if "ingest completed successfully" in normalized_output:
            self._notify_event("Ingest build completed.", severity="info")
            return
        if "error during ingest" in normalized_output:
            self._notify_event("Ingest build failed.", severity="error")

    def _update_command_progress(self, current: Optional[int], total: Optional[int]) -> None:
        """Update the command progress bar."""
        progress_bar = self.query_one("#command-progress", ProgressBar)
        if current is None or total in (None, 0):
            progress_bar.update(total=1, progress=0)
            return
        progress_bar.update(total=total, progress=min(current, total))

    def _render_explore_tree(self, root: Dict[str, Any], label: str) -> None:
        """Render explore results into the tree widget."""
        tree_widget = self.query_one("#ingest-explore-tree", Tree)
        tree_widget.root.set_label(
            f"{label}/ ({root['file_count']:,} files, {root['chunk_count']:,} chunks)"
        )
        tree_widget.root.remove_children()

        def add_nodes(parent, node, current_path: Path):
            children = sorted(node["children"].values(), key=lambda n: n["name"].lower())
            files = sorted(node["files"].items(), key=lambda item: item[0].lower())

            for child in children:
                child_path = current_path / child["name"]
                label_text = (
                    f"{child['name']}/ ({child['file_count']:,} files, "
                    f"{child['chunk_count']:,} chunks)"
                )
                child_node = parent.add(label_text)
                child_node.data = {
                    "type": "dir",
                    "path": str(child_path),
                    "file_count": child["file_count"],
                    "chunk_count": child["chunk_count"],
                }
                add_nodes(child_node, child, child_path)

            for name, count in files:
                file_path = current_path / name
                file_node = parent.add(f"{name} ({count:,} chunks)")
                file_node.data = {
                    "type": "file",
                    "path": str(file_path),
                    "chunk_count": count,
                }

        add_nodes(tree_widget.root, root, Path(label))
        tree_widget.root.expand()
        self._set_output_mode("tree")
        tree_widget.focus()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle selection changes in the explore tree."""
        if event.node.tree.id != "ingest-explore-tree":
            return

        selection_widget = self.query_one("#ingest-selection-stats", Static)
        data = getattr(event.node, "data", None) or {}
        if data.get("type") == "file":
            selection_widget.update(
                f"Selection: {data.get('path')} ({data.get('chunk_count'):,} chunks)"
            )
        elif data.get("type") == "dir":
            selection_widget.update(
                f"Selection: {data.get('path')} ({data.get('file_count'):,} files, "
                f"{data.get('chunk_count'):,} chunks)"
            )
        else:
            selection_widget.update("Selection: none")

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Handle highlight changes in the explore tree."""
        if event.node.tree.id != "ingest-explore-tree":
            return
        selection_widget = self.query_one("#ingest-selection-stats", Static)
        data = getattr(event.node, "data", None) or {}
        if data.get("type") == "file":
            selection_widget.update(
                f"Selection: {data.get('path')} ({data.get('chunk_count'):,} chunks)"
            )
        elif data.get("type") == "dir":
            selection_widget.update(
                f"Selection: {data.get('path')} ({data.get('file_count'):,} files, "
                f"{data.get('chunk_count'):,} chunks)"
            )
        else:
            selection_widget.update("Selection: none")

    def _execute_ingest_command(
        self,
        command: str,
        responses: Deque[str],
        build_progress_callback,
        stats_progress_callback,
    ) -> tuple[CommandResult, Optional[str], List[str]]:
        """Run the ingest command handler and return CommandResult."""
        consumed: List[str] = []

        try:
            # Call new CommandResult-based handler
            result = handle_ingest_command(
                command,
                self.app.ingest_provider,
                self.app.config,
                progress_callback=stats_progress_callback,
            )

            # Handle actions that need user confirmation
            if result.action == "build_confirm":
                # Force build needs confirmation
                if responses:
                    confirm = responses.popleft()
                    consumed.append(confirm)
                    if confirm.lower() == "yes":
                        build_result = run_build(
                            result.action_data["config"],
                            force=True,
                            progress_callback=build_progress_callback,
                        )
                        return CommandResult(output=result.output + "\n" + build_result), None, consumed
                    else:
                        return CommandResult(output="Cancelled."), None, consumed
                else:
                    return result, "Continue? (yes/no): ", consumed

            elif result.action == "build_run":
                # Non-force build - run directly
                build_result = run_build(
                    result.action_data["config"],
                    force=False,
                    progress_callback=build_progress_callback,
                )
                return CommandResult(output=result.output + "\n" + build_result), None, consumed

            elif result.action == "wipe_confirm":
                # Wipe needs double confirmation
                if len(responses) >= 2:
                    confirm1 = responses.popleft()
                    consumed.append(confirm1)
                    confirm2 = responses.popleft()
                    consumed.append(confirm2)
                    if confirm1 == "DELETE" and confirm2.lower() == "yes":
                        wipe_result = run_wipe(result.action_data["config"])
                        return CommandResult(output=result.output + "\n" + wipe_result), None, consumed
                    else:
                        return CommandResult(output="Cancelled."), None, consumed
                elif len(responses) == 1:
                    confirm1 = responses.popleft()
                    consumed.append(confirm1)
                    if confirm1 == "DELETE":
                        return result, "Are you absolutely sure? (yes/no): ", consumed
                    else:
                        return CommandResult(output="Cancelled."), None, consumed
                else:
                    return result, "Type 'DELETE' to confirm: ", consumed

            elif result.action == "tag_confirm":
                # Tag needs confirmation
                if responses:
                    confirm = responses.popleft()
                    consumed.append(confirm)
                    if confirm.lower() == "yes":
                        tag_result = run_tag(
                            result.action_data["provider"],
                            result.action_data["config"],
                            result.action_data.get("max_chunks"),
                        )
                        return CommandResult(output=result.output + "\n" + tag_result), None, consumed
                    else:
                        return CommandResult(output="Cancelled."), None, consumed
                else:
                    return result, "Proceed with tagging? (yes/no): ", consumed

            return result, None, consumed

        except Exception as exc:
            return CommandResult(output=f"\nError: {exc}", handled=False), None, consumed

    async def _run_ingest_command(
        self,
        command: str,
        responses: List[str],
    ) -> tuple[str, Optional[str]]:
        """Run an ingest command using the new CommandResult-based handlers."""
        app = self.app

        if not hasattr(app, "ingest_provider") or app.ingest_provider is None:
            try:
                await app._async_init_ingest_context()
            except Exception as exc:
                return f"Failed to initialize ingest provider: {exc}", None

        if app.ingest_provider is None:
            return "Ingest provider not initialized.", None

        responses_queue: Deque[str] = deque(responses)

        def stats_progress_callback(analyzed: int, total: Optional[int]) -> None:
            def update() -> None:
                if total:
                    pct = (analyzed / total) * 100 if total > 0 else 0.0
                    self._set_command_status(
                        f"Analyzing: {analyzed:,} / {total:,} ({pct:.1f}%)"
                    )
                    self._update_command_progress(analyzed, total)
                else:
                    self._set_command_status(f"Analyzing: {analyzed:,} chunks")
                    self._update_command_progress(None, None)
            if hasattr(self.app, "run_on_ui_thread"):
                self.app.run_on_ui_thread(update)
            else:
                try:
                    self.app.call_from_thread(update)
                except RuntimeError as exc:
                    if "must run in a different thread" in str(exc):
                        update()
                    else:
                        raise

        def build_progress_callback(event: str, current: int, total: int, message: str) -> None:
            def update() -> None:
                self._set_command_status(f"{event}: {message}")
                if total > 0:
                    self._update_command_progress(current, total)
                else:
                    self._update_command_progress(None, None)
            if hasattr(self.app, "run_on_ui_thread"):
                self.app.run_on_ui_thread(update)
            else:
                try:
                    self.app.call_from_thread(update)
                except RuntimeError as exc:
                    if "must run in a different thread" in str(exc):
                        update()
                    else:
                        raise

        result, prompt, consumed = await asyncio.to_thread(
            self._execute_ingest_command,
            command,
            responses_queue,
            build_progress_callback,
            stats_progress_callback,
        )

        if prompt:
            self._pending_responses = consumed

        # Check for exit
        if result.should_exit:
            self.app.exit()

        output = result.output
        return output.rstrip() if output else "", prompt

    async def _refresh_stats(self) -> None:
        """Refresh the stats display."""
        stats_widget = self.query_one("#stats-content", Static)

        try:
            app = self.app

            # Initialize provider if needed
            if not hasattr(app, 'ingest_provider') or app.ingest_provider is None:
                try:
                    await app._async_init_ingest_context()
                except Exception as e:
                    stats_widget.update(f"Not initialized: {e}")
                    return

            provider = app.ingest_provider
            count = provider.count()
            self.chunk_count = count
            app.update_chunk_count(count)

            stats_widget.update(
                f"Collection: {app.config.collection}\n"
                f"Chunks: {count:,}\n"
                f"Provider: {app.config.vectordb.provider}"
            )

        except Exception as e:
            logger.warning(f"Failed to refresh stats: {e}")
            stats_widget.update(f"Error: {e}")

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_run_build(self) -> None:
        """Run the build operation."""
        self._start_managed_worker(
            worker_key="ingest-command",
            work_factory=lambda: self._process_command("/build"),
            timeout_s=self._INGEST_COMMAND_WORKER_TIMEOUT_SECONDS,
            exclusive=True,
        )

    def action_run_tagging(self) -> None:
        """Run the tagging operation."""
        self._start_managed_worker(
            worker_key="ingest-command",
            work_factory=lambda: self._process_command("/tag"),
            timeout_s=self._INGEST_COMMAND_WORKER_TIMEOUT_SECONDS,
            exclusive=True,
        )

    def action_refresh_stats(self) -> None:
        """Refresh statistics."""
        self._start_managed_worker(
            worker_key="ingest-refresh-stats",
            work_factory=self._refresh_stats,
            timeout_s=self._INGEST_STATS_WORKER_TIMEOUT_SECONDS,
            exclusive=True,
        )

    def action_clear_input(self) -> None:
        """Clear the input field."""
        command_input = self.query_one("#ingest-command-input", CommandInput)
        command_input.clear()

    # Worker/timer lifecycle methods inherited from ManagedScreenMixin.
