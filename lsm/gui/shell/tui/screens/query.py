"""
Query screen for LSM TUI.

Provides the query interface with:
- Query input area with history and autocomplete
- Results display with expandable citations
- Remote search results section
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any
from contextlib import redirect_stdout, redirect_stderr
import asyncio
import io

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Log
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from lsm.gui.shell.logging import get_logger
from lsm.gui.shell.tui.widgets.results import ResultsPanel, CitationSelected, CitationExpanded
from lsm.gui.shell.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.gui.shell.tui.completions import create_completer
from lsm.gui.shell.query.commands import handle_command as handle_query_command
from lsm.query.session import SessionState
from lsm.query.cost_tracking import CostTracker

if TYPE_CHECKING:
    from lsm.query.session import Candidate

logger = get_logger(__name__)


class QuerySubmitted(Message):
    """Message sent when a query is submitted."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class QueryScreen(Widget):
    """
    Query interface screen.

    Provides:
    - Text input for queries and commands with history and autocomplete
    - Scrollable results panel with expandable citations
    - Citation expansion
    - Streaming response display
    """

    BINDINGS = [
        Binding("enter", "submit_query", "Submit", show=False),
        Binding("ctrl+e", "expand_citation", "Expand", show=False),
        Binding("ctrl+o", "open_source", "Open Source", show=False),
        Binding("escape", "clear_input", "Clear", show=False),
    ]

    # Reactive state
    is_loading: reactive[bool] = reactive(False)
    selected_citation: reactive[Optional[int]] = reactive(None)

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the query screen."""
        super().__init__(*args, **kwargs)
        self._last_response: str = ""
        self._last_candidates: List["Candidate"] = []
        self._completer = create_completer("query", self._get_candidates)

    def _get_candidates(self) -> Optional[List[str]]:
        """Get current candidates for completion."""
        if self._last_candidates:
            return [f"S{i}" for i in range(1, len(self._last_candidates) + 1)]
        return None

    def compose(self) -> ComposeResult:
        """Compose the query screen layout."""
        with Vertical(id="query-layout"):
            with Horizontal(id="query-top"):
                # Results area with ResultsPanel widget
                yield ResultsPanel(id="query-results-panel")

                # Log output panel
                with Container(id="query-log-panel"):
                    yield Static("Logs", classes="log-panel-title")
                    yield Log(id="query-log", auto_scroll=True)

            # Input area with CommandInput widget
            with Container(id="query-input-container"):
                yield CommandInput(
                    placeholder="Enter your question or command...",
                    completer=self._completer,
                    id="query-command-input",
                )

    def on_mount(self) -> None:
        """Handle screen mount - focus the input."""
        logger.debug("Query screen mounted")
        self.query_one("#query-command-input", CommandInput).focus()
        if hasattr(self.app, "_tui_log_buffer"):
            log_widget = self.query_one("#query-log", Log)
            for message in self.app._tui_log_buffer:
                log_widget.write(f"{message}\n")
            log_widget.scroll_end()

    async def on_command_submitted(self, event: CommandSubmitted) -> None:
        """Handle command input submission from CommandInput widget."""
        command = event.command.strip()
        if not command:
            return

        # Process the input
        await self._process_input(command)

    def on_citation_selected(self, event: CitationSelected) -> None:
        """Handle citation selection from ResultsPanel."""
        self.selected_citation = event.index
        logger.debug(f"Citation S{event.index} selected")

    def on_citation_expanded(self, event: CitationExpanded) -> None:
        """Handle citation expansion from ResultsPanel."""
        logger.debug(f"Citation S{event.index} expanded")

    async def _process_input(self, text: str) -> None:
        """
        Process input text (query or command).

        Args:
            text: Input text from user
        """
        # Check if it's a command
        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._handle_query(text)

    async def _handle_command(self, command: str) -> None:
        """
        Handle a slash command.

        Args:
            command: Command string (e.g., "/help", "/show S1")
        """
        normalized = command.strip().lower()
        if normalized in {"/ingest", "/i"}:
            self.app.action_switch_ingest()
            return
        if normalized in {"/query", "/q"}:
            self.app.action_switch_query()
            return

        output = await self._run_query_command(command)
        if output:
            self._show_message(output)

    def _ensure_query_state(self) -> None:
        """Ensure a session state exists for command handling."""
        app = self.app
        if getattr(app, "query_state", None) is None:
            query_config = app.config.llm.get_query_config()
            app._query_state = SessionState(
                path_contains=app.config.query.path_contains,
                ext_allow=app.config.query.ext_allow,
                ext_deny=app.config.query.ext_deny,
                model=query_config.model,
                cost_tracker=CostTracker(),
            )

    def _execute_query_command(self, command: str) -> tuple[bool, str]:
        """Run the query command handler and capture output."""
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                handle_query_command(
                    command,
                    self.app.query_state,
                    self.app.config,
                    self.app.query_embedder,
                    self.app.query_provider,
                )
        except SystemExit:
            return True, buffer.getvalue()
        except Exception as exc:
            output = buffer.getvalue()
            return False, f"{output}\nError: {exc}".strip()

        return False, buffer.getvalue()

    async def _run_query_command(self, command: str) -> str:
        """Run a query command using the shared REPL handlers."""
        self._ensure_query_state()

        normalized = command.strip()
        if normalized.lower() == "/quit":
            normalized = "/exit"

        did_exit, output = await asyncio.to_thread(self._execute_query_command, normalized)

        if normalized.lower() == "/exit" or did_exit:
            self.app.exit()
        elif normalized.lower().startswith("/mode"):
            self.app.current_mode = self.app.config.query.mode

        return output.rstrip()

    async def _handle_query(self, query: str) -> None:
        """
        Handle a natural language query.

        Args:
            query: Query text
        """
        results_panel = self.query_one("#query-results-panel", ResultsPanel)
        self.is_loading = True

        try:
            # Show loading state
            self._show_message(
                f"Searching for: {query}\n\nProcessing...",
                preserve_candidates=False,
            )

            # Get app reference and check if query context is ready
            app = self.app
            if not hasattr(app, "query_provider") or app.query_provider is None:
                self._show_message(
                    "Query context not initialized.\n\n"
                    "Please wait for initialization or check settings.",
                    preserve_candidates=False,
                )
                return

            # Run the query in a thread to not block UI
            response, candidates = await self._run_query(query)

            # Store candidates for completion
            self._last_candidates = candidates
            self._last_response = response

            # Update display with results using ResultsPanel
            results_panel.update_results(response, candidates)
            self.app.current_mode = self.app.config.query.mode

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            self._show_message(f"Error: {e}", preserve_candidates=False)
        finally:
            self.is_loading = False

    async def _run_query(self, query: str) -> tuple[str, List["Candidate"]]:
        """
        Run a query against the knowledge base.

        Args:
            query: Query text

        Returns:
            Tuple of (response text, list of candidates)
        """
        app = self.app
        candidates: List["Candidate"] = []

        try:
            # Import query execution functions
            from lsm.gui.shell.query.repl import run_query_turn_async

            # Run query - this function is designed for async/TUI use
            result = await run_query_turn_async(
                query,
                app.config,
                app.query_state,
                app.query_embedder,
                app.query_provider,
            )

            if result:
                response_text = result.get("response", "No response generated.")
                candidates = result.get("candidates", [])
                remote_sources = result.get("remote_sources", []) or []

                if remote_sources:
                    lines = ["", "=" * 60, "REMOTE SOURCES", "=" * 60]
                    for i, remote in enumerate(remote_sources, 1):
                        title = remote.get("title") or "(no title)"
                        url = remote.get("url") or ""
                        snippet = remote.get("snippet") or ""
                        lines.append(f"\\n{i}. {title}")
                        if url:
                            lines.append(f"   {url}")
                        if snippet:
                            snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet
                            lines.append(f"   {snippet}")
                    response_text += "\\n".join(lines)

                # Update cost tracking
                if "cost" in result and hasattr(app, 'update_cost'):
                    app.update_cost(result["cost"])

                return response_text, candidates

        except ImportError as e:
            logger.warning(f"Query module not available: {e}, using fallback")
        except Exception as e:
            logger.error(f"Query execution error: {e}", exc_info=True)
            return f"Query error: {e}", []

        # Fallback for when query module is not available
        return self._sync_query(query), []

    def _sync_query(self, query: str) -> str:
        """
        Run a synchronous query (for fallback).

        Args:
            query: Query text

        Returns:
            Response text
        """
        # This is a placeholder - the actual implementation
        # will integrate with the existing query system
        return f"Query: {query}\n\n[Query execution placeholder - integrate with existing system]"

    def _show_message(self, message: str, preserve_candidates: bool = True) -> None:
        """
        Display a message in the results panel.

        Args:
            message: Message to display
            preserve_candidates: If True, keep prior citations in the panel
        """
        results_panel = self.query_one("#query-results-panel", ResultsPanel)
        candidates = self._last_candidates if preserve_candidates else []
        results_panel.update_results(message, candidates)

    def _show_help(self) -> None:
        """Display help text."""
        help_text = """QUERY COMMANDS

/help           Show this help
/mode           Show current query mode
/mode <name>    Switch to a different query mode
/show S#        Show the cited chunk (e.g., /show S2)
/expand S#      Show full chunk text (no truncation)
/costs          Show session cost summary
/debug          Show retrieval diagnostics

KEYBOARD SHORTCUTS

Ctrl+I          Switch to Ingest tab
Ctrl+Q          Switch to Query tab
Ctrl+S          Switch to Settings tab
F1              Show help modal
Ctrl+C          Quit

Enter your question to search your knowledge base.
Use /mode to switch between grounded, insight, and hybrid modes."""

        self._show_message(help_text)

    def _show_mode(self) -> None:
        """Display current mode."""
        app = self.app

        if hasattr(app, 'current_mode'):
            mode = app.current_mode
            self._show_message(
                f"Current mode: {mode}\n\n"
                "Available modes:\n- grounded\n- insight\n- hybrid\n\n"
                "Use /mode <name> to switch."
            )
        else:
            self._show_message("Mode information not available.")

    async def _set_mode(self, mode: str) -> None:
        """Set the query mode."""
        app = self.app

        valid_modes = ["grounded", "insight", "hybrid"]
        if mode.lower() in valid_modes:
            app.current_mode = mode.lower()
            self._show_message(f"Switched to {mode} mode.")
        else:
            self._show_message(f"Invalid mode: {mode}\n\nValid modes: {', '.join(valid_modes)}")

    def _show_citation(self, citation: str) -> None:
        """Show a specific citation."""
        # Parse citation number (S1, S2, etc.)
        try:
            index = int(citation.upper().replace("S", ""))
            results_panel = self.query_one("#query-results-panel", ResultsPanel)
            candidate = results_panel.get_candidate(index)

            if candidate:
                meta = candidate.meta or {}
                source_path = meta.get("source_path", "unknown")
                chunk_index = meta.get("chunk_index", "N/A")
                text = (candidate.text or "").strip()

                self._show_message(
                    f"Citation S{index}\n"
                    f"{'=' * 40}\n"
                    f"Source: {source_path}\n"
                    f"Chunk: {chunk_index}\n"
                    f"Distance: {candidate.distance:.4f}\n\n"
                    f"{text[:500]}{'...' if len(text) > 500 else ''}"
                )
            else:
                self._show_message(f"Citation S{index} not found.")
        except (ValueError, AttributeError):
            self._show_message(f"Invalid citation format: {citation}\n\nUse S# format (e.g., S1, S2)")

    def _expand_citation(self, citation: str) -> None:
        """Expand a specific citation to show full text."""
        try:
            index = int(citation.upper().replace("S", ""))
            results_panel = self.query_one("#query-results-panel", ResultsPanel)
            results_panel.expand_citation(index)
        except (ValueError, AttributeError):
            self._show_message(f"Invalid citation format: {citation}\n\nUse S# format (e.g., S1, S2)")

    def _show_costs(self) -> None:
        """Show session costs."""
        app = self.app

        if hasattr(app, 'total_cost'):
            cost = app.total_cost
            self._show_message(f"Session Cost Summary\n\nTotal: ${cost:.4f}")
        else:
            self._show_message("Cost tracking not available.")

    def _show_debug(self) -> None:
        """Show debug information."""
        app = self.app
        debug_info = ["Debug Information", "=" * 40]

        if hasattr(app, 'current_mode'):
            debug_info.append(f"Mode: {app.current_mode}")
        if hasattr(app, 'chunk_count'):
            debug_info.append(f"Chunks: {app.chunk_count:,}")
        if hasattr(app, 'query_provider') and app.query_provider:
            debug_info.append("Query provider: initialized")
        else:
            debug_info.append("Query provider: not initialized")
        if hasattr(app, 'query_embedder') and app.query_embedder:
            debug_info.append("Embedder: initialized")
        else:
            debug_info.append("Embedder: not initialized")

        debug_info.append(f"\nLast candidates: {len(self._last_candidates)}")

        self._show_message("\n".join(debug_info))

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_submit_query(self) -> None:
        """Submit the current query."""
        command_input = self.query_one("#query-command-input", CommandInput)
        if command_input.value:
            # Trigger submission by posting the message
            self.post_message(CommandSubmitted(command_input.value))
            command_input.clear()

    def action_expand_citation(self) -> None:
        """Expand the selected citation."""
        if self.selected_citation is not None:
            self._expand_citation(f"S{self.selected_citation}")
        else:
            self.app.notify("No citation selected", severity="warning")

    def action_open_source(self) -> None:
        """Open the source file for the selected citation."""
        if self.selected_citation is not None:
            results_panel = self.query_one("#query-results-panel", ResultsPanel)
            candidate = results_panel.get_candidate(self.selected_citation)
            if candidate:
                meta = candidate.meta or {}
                source_path = meta.get("source_path")
                if source_path:
                    import subprocess
                    import sys
                    try:
                        if sys.platform == "win32":
                            subprocess.run(["start", "", source_path], shell=True)
                        elif sys.platform == "darwin":
                            subprocess.run(["open", source_path])
                        else:
                            subprocess.run(["xdg-open", source_path])
                        self.app.notify(f"Opening {source_path}")
                    except Exception as e:
                        self.app.notify(f"Failed to open file: {e}", severity="error")
                else:
                    self.app.notify("No source path available", severity="warning")
            else:
                self.app.notify("Citation not found", severity="warning")
        else:
            self.app.notify("No citation selected", severity="warning")

    def action_clear_input(self) -> None:
        """Clear the input field."""
        command_input = self.query_one("#query-command-input", CommandInput)
        command_input.clear()
