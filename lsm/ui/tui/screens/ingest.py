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
from typing import TYPE_CHECKING, Optional, Dict, Any, Deque, List
from collections import deque
import io
import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, DirectoryTree, ProgressBar, Tree
from textual.widget import Widget
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.ui.tui.completions import create_completer
from lsm.ingest.commands import handle_command as handle_ingest_command, CommandResult, run_build, run_wipe, run_tag

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class IngestScreen(Widget):
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

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the ingest screen."""
        super().__init__(*args, **kwargs)
        self._stats: Dict[str, Any] = {}
        self._completer = create_completer("ingest")
        self._pending_command: Optional[str] = None
        self._pending_responses: List[str] = []
        self._pending_prompt: Optional[str] = None
        self._explore_tree_active = False

    def compose(self) -> ComposeResult:
        """Compose the ingest screen layout."""
        with Horizontal(id="ingest-layout"):
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
        # Refresh stats on mount
        self.run_worker(self._refresh_stats(), exclusive=True)
        # Focus command input
        self.query_one("#ingest-command-input", CommandInput).focus()
        # Hide explore tree until needed
        self.query_one("#ingest-explore-tree", Tree).styles.display = "none"
        self._set_command_status("Ready.")

    async def on_command_submitted(self, event: CommandSubmitted) -> None:
        """Handle command input submission from CommandInput widget."""
        command = event.command.strip()
        if not command:
            return

        # Process command
        await self._process_command(command)

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
            self._pending_prompt = prompt
            prefix = (output + "\n") if output else ""
            output_widget.update(prefix + prompt)
            self._set_command_status("Waiting for input...")
            return

        self._pending_command = None
        self._pending_prompt = None
        self._pending_responses = []
        self._set_command_status("Ready.")
        self._update_command_progress(None, None)

    def _show_help(self) -> None:
        """Display help text."""
        output_widget = self.query_one("#ingest-output", Static)
        help_text = """INGEST COMMANDS

/info              Show collection information
/stats             Show detailed statistics
/explore [query]   Browse indexed files
/show <path>       Show chunks for a file
/search <query>    Search metadata
/build [--force]   Run ingest pipeline
/tag [--max N]     Run AI tagging
/tags              Show all tags
/vectordb-providers  List vector DB providers
/vectordb-status   Show vector DB status
/wipe              Clear collection (requires confirmation)

KEYBOARD SHORTCUTS

Ctrl+B             Run build
Ctrl+T             Run tagging
Ctrl+Shift+R        Refresh stats

Options:
  --force          Force full rebuild (skip incremental)
  --max N          Limit tagging to N chunks"""

        output_widget.update(help_text)

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
        stream_callback,
        stats_progress_callback,
        explore_progress_callback,
        explore_emit_tree,
    ) -> tuple[CommandResult, str, Optional[str], List[str]]:
        """Run the ingest command handler and return CommandResult."""
        buffer = io.StringIO()
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
                        build_result = run_build(result.action_data["config"], force=True)
                        return CommandResult(output=result.output + "\n" + build_result), buffer.getvalue(), None, consumed
                    else:
                        return CommandResult(output="Cancelled."), buffer.getvalue(), None, consumed
                else:
                    return result, buffer.getvalue(), "Continue? (yes/no): ", consumed

            elif result.action == "build_run":
                # Non-force build - run directly
                build_result = run_build(result.action_data["config"], force=False)
                return CommandResult(output=result.output + "\n" + build_result), buffer.getvalue(), None, consumed

            elif result.action == "wipe_confirm":
                # Wipe needs double confirmation
                if len(responses) >= 2:
                    confirm1 = responses.popleft()
                    consumed.append(confirm1)
                    confirm2 = responses.popleft()
                    consumed.append(confirm2)
                    if confirm1 == "DELETE" and confirm2.lower() == "yes":
                        wipe_result = run_wipe(result.action_data["collection"])
                        return CommandResult(output=result.output + "\n" + wipe_result), buffer.getvalue(), None, consumed
                    else:
                        return CommandResult(output="Cancelled."), buffer.getvalue(), None, consumed
                elif len(responses) == 1:
                    confirm1 = responses.popleft()
                    consumed.append(confirm1)
                    if confirm1 == "DELETE":
                        return result, buffer.getvalue(), "Are you absolutely sure? (yes/no): ", consumed
                    else:
                        return CommandResult(output="Cancelled."), buffer.getvalue(), None, consumed
                else:
                    return result, buffer.getvalue(), "Type 'DELETE' to confirm: ", consumed

            elif result.action == "tag_confirm":
                # Tag needs confirmation
                if responses:
                    confirm = responses.popleft()
                    consumed.append(confirm)
                    if confirm.lower() == "yes":
                        tag_result = run_tag(
                            result.action_data["collection"],
                            result.action_data["config"],
                            result.action_data.get("max_chunks"),
                        )
                        return CommandResult(output=result.output + "\n" + tag_result), buffer.getvalue(), None, consumed
                    else:
                        return CommandResult(output="Cancelled."), buffer.getvalue(), None, consumed
                else:
                    return result, buffer.getvalue(), "Proceed with tagging? (yes/no): ", consumed

            return result, buffer.getvalue(), None, consumed

        except Exception as exc:
            output = buffer.getvalue()
            output += f"\nError: {exc}"
            return CommandResult(output=output, handled=False), "", None, consumed

    async def _run_ingest_command(
        self,
        command: str,
        responses: List[str],
    ) -> tuple[str, Optional[str]]:
        """Run an ingest command using the new CommandResult-based handlers."""
        app = self.app
        self._explore_tree_active = False

        if not hasattr(app, "ingest_provider") or app.ingest_provider is None:
            try:
                await app._async_init_ingest_context()
            except Exception as exc:
                return f"Failed to initialize ingest provider: {exc}", None

        if app.ingest_provider is None:
            return "Ingest provider not initialized.", None

        responses_queue: Deque[str] = deque(responses)

        def stream_callback(buffer: io.StringIO) -> None:
            def update() -> None:
                if self._explore_tree_active:
                    return
                self._update_output_text(buffer.getvalue())
            self.app.call_from_thread(update)

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
            self.app.call_from_thread(update)

        def explore_progress_callback(scanned: int, total: Optional[int]) -> None:
            def update() -> None:
                if total:
                    pct = (scanned / total) * 100 if total > 0 else 0.0
                    self._set_command_status(
                        f"Scanning: {scanned:,} / {total:,} ({pct:.1f}%)"
                    )
                    self._update_command_progress(scanned, total)
                else:
                    self._set_command_status(f"Scanning: {scanned:,} chunks")
                    self._update_command_progress(None, None)
            self.app.call_from_thread(update)

        def explore_emit_tree(tree_data: Dict[str, Any], label: str) -> None:
            def update() -> None:
                self._explore_tree_active = True
                self._render_explore_tree(tree_data, label)
            self.app.call_from_thread(update)

        result, extra_output, prompt, consumed = await asyncio.to_thread(
            self._execute_ingest_command,
            command,
            responses_queue,
            stream_callback,
            stats_progress_callback,
            explore_progress_callback,
            explore_emit_tree,
        )

        if prompt:
            self._pending_responses = consumed

        # Check for exit
        if result.should_exit:
            self.app.exit()

        # Combine output
        output = result.output
        if extra_output:
            output = extra_output + "\n" + output if output else extra_output

        return output.rstrip() if output else "", prompt

    async def _show_stats(self) -> None:
        """Show detailed collection statistics."""
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update("Loading statistics...")

        try:
            app = self.app
            if not hasattr(app, 'ingest_provider') or app.ingest_provider is None:
                # Try to initialize
                await app._async_init_ingest_context()

            provider = app.ingest_provider
            if provider is None:
                output_widget.update("Ingest provider not initialized.")
                return

            # Get stats
            import asyncio
            stats = await asyncio.to_thread(provider.get_stats)
            count = provider.count()

            # Format output
            lines = [
                "Collection Statistics",
                "=" * 40,
                f"Total chunks: {count:,}",
            ]

            if "unique_files" in stats:
                lines.append(f"Unique files: {stats['unique_files']:,}")

            if "file_types" in stats:
                lines.append("\nFile types:")
                for ext, cnt in sorted(stats["file_types"].items(), key=lambda x: -x[1])[:10]:
                    lines.append(f"  {ext}: {cnt:,}")

            output_widget.update("\n".join(lines))
            self.chunk_count = count

        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            output_widget.update(f"Error getting stats: {e}")

    async def _show_info(self) -> None:
        """Show basic collection info."""
        output_widget = self.query_one("#ingest-output", Static)

        try:
            app = self.app
            provider = app.ingest_provider

            if provider is None:
                output_widget.update("Ingest provider not initialized.")
                return

            count = provider.count()
            info = f"Collection: {app.config.collection}\nChunks: {count:,}"
            output_widget.update(info)

        except Exception as e:
            output_widget.update(f"Error: {e}")

    async def _run_build(self, force: bool = False) -> None:
        """
        Run the ingest build pipeline.

        Args:
            force: If True, force full rebuild
        """
        output_widget = self.query_one("#ingest-output", Static)
        progress_bar = self.query_one("#build-progress", ProgressBar)

        self.is_building = True
        progress_bar.update(total=100, progress=0)

        try:
            output_widget.update(f"Starting {'full' if force else 'incremental'} build...")

            # Run build in thread
            app = self.app

            # Import and run ingest pipeline
            from lsm.ingest.pipeline import run_ingest_pipeline

            def do_build():
                return run_ingest_pipeline(
                    app.config,
                    force=force,
                    progress_callback=self._update_progress,
                )

            import asyncio
            result = await asyncio.to_thread(do_build)

            # Update display
            progress_bar.update(progress=100)
            output_widget.update(f"Build complete!\n\n{result}")

            # Refresh stats
            await self._refresh_stats()

        except Exception as e:
            logger.error(f"Build failed: {e}", exc_info=True)
            output_widget.update(f"Build failed: {e}")
        finally:
            self.is_building = False

    def _update_progress(self, current: int, total: int, message: str = "") -> None:
        """
        Update build progress.

        Args:
            current: Current progress value
            total: Total value
            message: Optional status message
        """
        progress_bar = self.query_one("#command-progress", ProgressBar)
        progress_bar.update(total=total, progress=current)

        if message:
            output_widget = self.query_one("#ingest-output", Static)
            output_widget.update(f"Building: {message}\n\nProgress: {current}/{total}")
            self._set_command_status(f"Building: {message}")

    async def _run_tagging(self, max_chunks: Optional[int] = None) -> None:
        """
        Run AI tagging on untagged chunks.

        Args:
            max_chunks: Maximum number of chunks to tag
        """
        output_widget = self.query_one("#ingest-output", Static)

        try:
            output_widget.update("Running AI tagging...")

            # Placeholder - integrate with actual tagging
            output_widget.update(
                f"AI Tagging {'(max ' + str(max_chunks) + ' chunks)' if max_chunks else ''}\n\n"
                "[Tagging integration placeholder]"
            )

        except Exception as e:
            output_widget.update(f"Tagging failed: {e}")

    async def _show_tags(self) -> None:
        """Show all tags in the collection."""
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update("Tags:\n\n[Tags display placeholder]")

    async def _explore(self, query: Optional[str] = None) -> None:
        """
        Explore indexed files.

        Args:
            query: Optional filter query
        """
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update(f"Exploring files{': ' + query if query else ''}...\n\n[Explorer placeholder]")

    async def _show_file(self, path: str) -> None:
        """
        Show chunks for a specific file.

        Args:
            path: File path
        """
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update(f"Showing chunks for: {path}\n\n[File chunks placeholder]")

    async def _search(self, query: str) -> None:
        """
        Search metadata.

        Args:
            query: Search query
        """
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update(f"Searching for: {query}\n\n[Search results placeholder]")

    async def _confirm_wipe(self) -> None:
        """Request confirmation for wipe operation."""
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update(
            "WARNING: This will delete all chunks from the collection!\n\n"
            "Type '/wipe confirm' to proceed."
        )

    def _show_vectordb_providers(self) -> None:
        """Show available vector DB providers."""
        output_widget = self.query_one("#ingest-output", Static)
        output_widget.update(
            "Available Vector DB Providers:\n\n"
            "- chromadb (default)\n"
            "- postgres (pgvector)"
        )

    async def _show_vectordb_status(self) -> None:
        """Show vector DB provider status."""
        output_widget = self.query_one("#ingest-output", Static)

        try:
            app = self.app
            provider = app.ingest_provider

            if provider is None:
                output_widget.update("Vector DB not initialized.")
                return

            stats = provider.get_stats()
            status = f"Vector DB Status:\n\nProvider: {stats.get('provider', 'unknown')}\nChunks: {provider.count():,}"
            output_widget.update(status)

        except Exception as e:
            output_widget.update(f"Error: {e}")

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
        self.run_worker(self._process_command("/build"), exclusive=True)

    def action_run_tagging(self) -> None:
        """Run the tagging operation."""
        self.run_worker(self._process_command("/tag"), exclusive=True)

    def action_refresh_stats(self) -> None:
        """Refresh statistics."""
        self.run_worker(self._refresh_stats(), exclusive=True)

    def action_clear_input(self) -> None:
        """Clear the input field."""
        command_input = self.query_one("#ingest-command-input", CommandInput)
        command_input.clear()
