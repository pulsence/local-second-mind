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
from typing import TYPE_CHECKING, Optional, Dict, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, DirectoryTree, ProgressBar
from textual.widget import Widget
from textual.reactive import reactive

from lsm.gui.shell.logging import get_logger
from lsm.gui.shell.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.gui.shell.tui.completions import create_completer

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
        Binding("ctrl+r", "refresh_stats", "Refresh", show=True),
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

    def compose(self) -> ComposeResult:
        """Compose the ingest screen layout."""
        with Horizontal(id="ingest-layout"):
            # Left pane: Directory tree
            with Vertical(classes="ingest-tree-container"):
                yield Static("Indexed Files", classes="stats-title")
                yield DirectoryTree(".", id="file-tree")

            # Right pane: Stats and controls
            with Vertical(classes="ingest-info-container"):
                # Stats panel
                with Container(classes="stats-panel"):
                    yield Static("Collection Statistics", classes="stats-title")
                    yield Static(
                        "Loading stats...",
                        id="stats-content",
                    )

                # Progress bar (hidden when not building)
                yield ProgressBar(
                    id="build-progress",
                    show_eta=True,
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
                        "Use Ctrl+B to build, Ctrl+T to tag, Ctrl+R to refresh.",
                        id="ingest-output",
                    )

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
        self.call_later(self._refresh_stats)
        # Focus command input
        self.query_one("#ingest-command-input", CommandInput).focus()

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
        cmd_lower = command.lower().strip()

        if cmd_lower == "/help":
            self._show_help()
        elif cmd_lower == "/stats":
            await self._show_stats()
        elif cmd_lower == "/info":
            await self._show_info()
        elif cmd_lower.startswith("/build"):
            force = "--force" in command.lower()
            await self._run_build(force=force)
        elif cmd_lower.startswith("/tag"):
            # Parse --max N if present
            max_chunks = None
            if "--max" in cmd_lower:
                parts = command.split()
                for i, p in enumerate(parts):
                    if p == "--max" and i + 1 < len(parts):
                        try:
                            max_chunks = int(parts[i + 1])
                        except ValueError:
                            pass
            await self._run_tagging(max_chunks=max_chunks)
        elif cmd_lower == "/tags":
            await self._show_tags()
        elif cmd_lower.startswith("/explore"):
            query = command[8:].strip() if len(command) > 8 else None
            await self._explore(query)
        elif cmd_lower.startswith("/show"):
            path = command[5:].strip()
            await self._show_file(path)
        elif cmd_lower.startswith("/search"):
            query = command[7:].strip()
            await self._search(query)
        elif cmd_lower == "/wipe":
            await self._confirm_wipe()
        elif cmd_lower == "/vectordb-providers":
            self._show_vectordb_providers()
        elif cmd_lower == "/vectordb-status":
            await self._show_vectordb_status()
        elif cmd_lower in ("/exit", "/quit"):
            self.app.exit()
        else:
            output_widget.update(f"Unknown command: {command}\n\nType /help for available commands.")

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
Ctrl+R             Refresh stats

Options:
  --force          Force full rebuild (skip incremental)
  --max N          Limit tagging to N chunks"""

        output_widget.update(help_text)

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
            stats = await app.run_in_thread(lambda: provider.get_stats())
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

            result = await app.run_in_thread(do_build)

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
        progress_bar = self.query_one("#build-progress", ProgressBar)
        progress_bar.update(total=total, progress=current)

        if message:
            output_widget = self.query_one("#ingest-output", Static)
            output_widget.update(f"Building: {message}\n\nProgress: {current}/{total}")

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

            stats_widget.update(f"Chunks: {count:,}\nCollection: {app.config.collection}")

        except Exception as e:
            logger.warning(f"Failed to refresh stats: {e}")
            stats_widget.update(f"Error: {e}")

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_run_build(self) -> None:
        """Run the build operation."""
        self.call_later(lambda: self._run_build())

    def action_run_tagging(self) -> None:
        """Run the tagging operation."""
        self.call_later(lambda: self._run_tagging())

    def action_refresh_stats(self) -> None:
        """Refresh statistics."""
        self.call_later(self._refresh_stats)

    def action_clear_input(self) -> None:
        """Clear the input field."""
        command_input = self.query_one("#ingest-command-input", CommandInput)
        command_input.clear()
