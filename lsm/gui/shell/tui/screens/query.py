"""
Query screen for LSM TUI.

Provides the query interface with:
- Query input area
- Results display with expandable citations
- Remote search results section
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, TextArea, Input
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from lsm.gui.shell.logging import get_logger

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
    - Text input for queries and commands
    - Scrollable results panel
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

    def compose(self) -> ComposeResult:
        """Compose the query screen layout."""
        with Vertical(id="query-layout"):
            # Results area (scrollable)
            with ScrollableContainer(id="results-container", classes="query-results-container"):
                yield Static(
                    "Enter a question to query your knowledge base.\n\n"
                    "Commands: /help, /mode, /show S#, /expand S#, /costs",
                    id="results-content",
                    classes="results-panel",
                )

            # Input area
            with Container(classes="query-input-container"):
                yield Input(
                    placeholder="Enter your question or command...",
                    id="query-input",
                    classes="command-input",
                )

    def on_mount(self) -> None:
        """Handle screen mount - focus the input."""
        logger.debug("Query screen mounted")
        self.query_one("#query-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        query = event.value.strip()
        if not query:
            return

        # Clear the input
        event.input.value = ""

        # Process the input
        await self._process_input(query)

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
        cmd_lower = command.lower().strip()
        results_widget = self.query_one("#results-content", Static)

        if cmd_lower == "/help":
            self._show_help()
        elif cmd_lower == "/mode":
            self._show_mode()
        elif cmd_lower.startswith("/mode "):
            mode = command[6:].strip()
            await self._set_mode(mode)
        elif cmd_lower.startswith("/show "):
            citation = command[6:].strip()
            self._show_citation(citation)
        elif cmd_lower.startswith("/expand "):
            citation = command[8:].strip()
            self._expand_citation(citation)
        elif cmd_lower == "/costs":
            self._show_costs()
        elif cmd_lower == "/debug":
            self._show_debug()
        elif cmd_lower in ("/exit", "/quit"):
            self.app.exit()
        else:
            results_widget.update(f"Unknown command: {command}\n\nType /help for available commands.")

    async def _handle_query(self, query: str) -> None:
        """
        Handle a natural language query.

        Args:
            query: Query text
        """
        results_widget = self.query_one("#results-content", Static)
        self.is_loading = True

        try:
            # Show loading state
            results_widget.update(f"Searching for: {query}\n\nProcessing...")

            # Get app reference and check if query context is ready
            app = self.app
            if not hasattr(app, 'query_provider') or app.query_provider is None:
                results_widget.update(
                    "Query context not initialized.\n\n"
                    "Please wait for initialization or check settings."
                )
                return

            # Run the query in a thread to not block UI
            response = await self._run_query(query)

            # Update display with results
            self._display_results(query, response)

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            results_widget.update(f"Error: {e}")
        finally:
            self.is_loading = False

    async def _run_query(self, query: str) -> str:
        """
        Run a query against the knowledge base.

        Args:
            query: Query text

        Returns:
            Response text
        """
        app = self.app

        # Import query execution
        from lsm.gui.shell.query.repl import run_query_turn_async

        # Check if async query turn exists, otherwise use sync version
        if hasattr(run_query_turn_async, '__call__'):
            result = await run_query_turn_async(
                query,
                app.config,
                app.query_state,
                app.query_embedder,
                app.query_provider,
            )
        else:
            # Fallback to sync version in thread
            from lsm.gui.shell.query.repl import run_query_turn
            result = await app.run_in_thread(
                lambda: self._sync_query(query)
            )

        return result or "No response generated."

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

    def _display_results(self, query: str, response: str) -> None:
        """
        Display query results.

        Args:
            query: Original query
            response: Response text
        """
        results_widget = self.query_one("#results-content", Static)
        self._last_response = response

        # Format the output
        output = f"Q: {query}\n\n{response}"

        # Add citation hints if we have candidates
        if self._last_candidates:
            output += f"\n\n[{len(self._last_candidates)} sources found - use /show S# to view]"

        results_widget.update(output)

    def _show_help(self) -> None:
        """Display help text."""
        results_widget = self.query_one("#results-content", Static)
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

        results_widget.update(help_text)

    def _show_mode(self) -> None:
        """Display current mode."""
        results_widget = self.query_one("#results-content", Static)
        app = self.app

        if hasattr(app, 'current_mode'):
            mode = app.current_mode
            results_widget.update(f"Current mode: {mode}\n\nAvailable modes:\n- grounded\n- insight\n- hybrid\n\nUse /mode <name> to switch.")
        else:
            results_widget.update("Mode information not available.")

    async def _set_mode(self, mode: str) -> None:
        """Set the query mode."""
        results_widget = self.query_one("#results-content", Static)
        app = self.app

        valid_modes = ["grounded", "insight", "hybrid"]
        if mode.lower() in valid_modes:
            app.current_mode = mode.lower()
            results_widget.update(f"Switched to {mode} mode.")
        else:
            results_widget.update(f"Invalid mode: {mode}\n\nValid modes: {', '.join(valid_modes)}")

    def _show_citation(self, citation: str) -> None:
        """Show a specific citation."""
        results_widget = self.query_one("#results-content", Static)
        results_widget.update(f"Showing citation: {citation}\n\n[Citation display placeholder]")

    def _expand_citation(self, citation: str) -> None:
        """Expand a specific citation."""
        results_widget = self.query_one("#results-content", Static)
        results_widget.update(f"Expanded citation: {citation}\n\n[Expanded view placeholder]")

    def _show_costs(self) -> None:
        """Show session costs."""
        results_widget = self.query_one("#results-content", Static)
        app = self.app

        if hasattr(app, 'total_cost'):
            cost = app.total_cost
            results_widget.update(f"Session Cost Summary\n\nTotal: ${cost:.4f}")
        else:
            results_widget.update("Cost tracking not available.")

    def _show_debug(self) -> None:
        """Show debug information."""
        results_widget = self.query_one("#results-content", Static)
        results_widget.update("Debug information:\n\n[Debug placeholder]")

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_submit_query(self) -> None:
        """Submit the current query."""
        input_widget = self.query_one("#query-input", Input)
        if input_widget.value:
            # Trigger submission
            input_widget.action_submit()

    def action_expand_citation(self) -> None:
        """Expand the selected citation."""
        if self.selected_citation is not None:
            self._expand_citation(f"S{self.selected_citation}")

    def action_open_source(self) -> None:
        """Open the source file for the selected citation."""
        self.app.notify("Open source: not yet implemented", severity="warning")

    def action_clear_input(self) -> None:
        """Clear the input field."""
        input_widget = self.query_one("#query-input", Input)
        input_widget.value = ""
