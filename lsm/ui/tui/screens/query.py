"""
Query screen for LSM TUI.

Provides the query interface with:
- Query input area with history and autocomplete
- Results display with expandable citations
- Remote search results section
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any, Callable
import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, RichLog, TabbedContent
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.tui.widgets.results import ResultsPanel, CitationSelected, CitationExpanded
from lsm.ui.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.ui.tui.completions import create_completer
from lsm.ui.helpers.commands.query import execute_query_command
from lsm.query.session import SessionState
from lsm.query.cost_tracking import CostTracker
from lsm.providers import create_provider
from lsm.ui.tui.presenters.query import (
    format_model_selection,
    format_models_list,
    format_providers,
    format_provider_status,
    format_vectordb_providers,
    format_vectordb_status,
    format_remote_providers,
)
from lsm.ui.tui.screens.base import ManagedScreenMixin
from lsm.ui.tui.widgets.status import StatusBar

if TYPE_CHECKING:
    from lsm.query.session import Candidate

logger = get_logger(__name__)


class QuerySubmitted(Message):
    """Message sent when a query is submitted."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class CommandResult:
    """Result from a command handler."""

    def __init__(
        self,
        output: str = "",
        handled: bool = True,
        should_exit: bool = False,
    ) -> None:
        self.output = output
        self.handled = handled
        self.should_exit = should_exit


class QueryScreen(ManagedScreenMixin, Widget):
    """
    Query interface screen.

    Provides:
    - Text input for queries and commands with history and autocomplete
    - Scrollable results panel with expandable citations
    - Citation expansion
    - Streaming response display
    """

    BINDINGS = [
        Binding("enter", "submit_query", "Submit", show=True),
        Binding("ctrl+e", "expand_citation", "Expand", show=True),
        Binding("ctrl+o", "open_source", "Open Source", show=True),
        Binding("ctrl+shift+r", "refresh_logs", "Refresh Logs", show=True),
        Binding("escape", "clear_input", "Clear", show=True),
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    # Reactive state
    is_loading: reactive[bool] = reactive(False)
    selected_citation: reactive[Optional[int]] = reactive(None)
    _QUERY_INPUT_WORKER_TIMEOUT_SECONDS = 45.0

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
            with ScrollableContainer(id="query-top"):
                # Results area with ResultsPanel widget
                yield ResultsPanel(id="query-results-panel")

                # Log output panel
                with Container(id="query-log-panel"):
                    yield Static("Logs", classes="log-panel-title")
                    yield RichLog(id="query-log", auto_scroll=True, wrap=True)

            # Input area with CommandInput widget
            yield CommandInput(
                placeholder="Enter your question or command...",
                completer=self._completer,
                id="query-command-input",
            )

    def on_mount(self) -> None:
        """Handle screen mount - focus the input."""
        logger.debug("Query screen mounted")
        self._focus_command_input()
        if hasattr(self.app, "_tui_log_buffer"):
            log_widget = self.query_one("#query-log", RichLog)
            for message in self.app._tui_log_buffer:
                log_widget.write(f"{message}\n")
            log_widget.scroll_end()

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Focus input when the query tab becomes active."""
        tab_id = event.tab.id
        if not tab_id:
            return
        context = tab_id.replace("-tab", "")
        if context == "query":
            self._focus_command_input()

    async def on_command_submitted(self, event: CommandSubmitted) -> None:
        """Handle command input submission from CommandInput widget."""
        command = event.command.strip()
        if not command:
            return

        self._start_managed_worker(
            worker_key="query-input",
            work_factory=lambda: self._process_input(command),
            timeout_s=self._QUERY_INPUT_WORKER_TIMEOUT_SECONDS,
            exclusive=True,
        )

    def on_unmount(self) -> None:
        self._cancel_managed_workers(reason="query-unmount")
        self._cancel_managed_timers(reason="query-unmount")

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

    def _format_model_selection(self) -> str:
        return format_model_selection(self.app.config.llm)

    def _format_models(self, command: str) -> str:
        def _store_models(ids: list[str]) -> None:
            self.app.query_state.available_models = ids

        return format_models_list(
            self.app.config.llm, command, store_callback=_store_models,
        )

    def _format_providers(self) -> str:
        return format_providers(self.app.config.llm)

    def _format_provider_status(self) -> str:
        return format_provider_status(self.app.config.llm)

    def _format_vectordb_providers(self) -> str:
        return format_vectordb_providers(self.app.config.vectordb)

    def _format_vectordb_status(self) -> str:
        return format_vectordb_status(self.app.config.vectordb, self.app.config.llm)

    def _format_remote_providers(self) -> str:
        return format_remote_providers(self.app.config)

    def _execute_query_command(self, command: str) -> CommandResult:
        """Run the query command handler via shared command helpers."""
        try:
            helper_result = execute_query_command(self, command)
            return CommandResult(
                output=helper_result.output,
                handled=helper_result.handled,
                should_exit=helper_result.should_exit,
            )
        except SystemExit:
            return CommandResult(output="", should_exit=True)
        except Exception as exc:
            return CommandResult(output=f"Error: {exc}", handled=False)

    async def _run_query_command(self, command: str) -> str:
        """Run a query command using the shared REPL handlers."""
        self._ensure_query_state()

        normalized = command.strip()
        if normalized.lower() == "/quit":
            normalized = "/exit"

        result = await asyncio.to_thread(self._execute_query_command, normalized)

        if normalized.lower() == "/exit" or result.should_exit:
            self.app.exit()
        elif normalized.lower().startswith("/mode"):
            self.app.current_mode = self.app.config.query.mode

        return result.output.rstrip()

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
                    f"Searching for: {query}\n\nInitializing query context...",
                    preserve_candidates=False,
                )
                try:
                    await app._async_init_query_context()
                except Exception as exc:
                    self._show_message(
                        "Query context not initialized.\n\n"
                        f"Initialization failed: {exc}",
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
            from lsm.query.api import QueryProgress, query as run_query

            def on_progress(progress: QueryProgress) -> None:
                message = f"{progress.stage}: {progress.message}"

                def update() -> None:
                    try:
                        status_bar = self.app.query_one("#main-status-bar", StatusBar)
                        status_bar.provider_status = message
                    except Exception:
                        pass

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

            result = await run_query(
                query,
                app.config,
                app.query_state,
                app.query_embedder,
                app.query_provider,
                progress_callback=on_progress,
            )

            response_text = f"{result.answer}\n{result.sources_display}"
            candidates = result.candidates
            remote_sources = result.remote_sources or []

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

            if result.cost and hasattr(app, 'update_cost'):
                app.update_cost(result.cost)

            try:
                status_bar = self.app.query_one("#main-status-bar", StatusBar)
                status_bar.provider_status = "ready"
            except Exception:
                pass

            return response_text, candidates

        except ImportError as e:
            logger.warning(f"Query module not available: {e}, using fallback")
        except Exception as e:
            logger.error(f"Query execution error: {e}", exc_info=True)
            try:
                status_bar = self.app.query_one("#main-status-bar", StatusBar)
                status_bar.provider_status = "error"
            except Exception:
                pass
            return f"Query error: {e}", []

        return self._sync_query(query), []

    def _sync_query(self, query: str) -> str:
        """
        Run a synchronous query (for fallback).

        Args:
            query: Query text

        Returns:
            Response text
        """
        from lsm.query.api import query_sync as run_query_sync

        try:
            result = run_query_sync(
                question=query,
                config=self._config,
                state=self._session_state,
                embedder=self._embedder,
                collection=self._collection,
            )
            self._last_candidates = result.candidates
            if result.debug_info:
                self._session_state.last_debug = result.debug_info
            if result.cost is not None:
                self._session_state.last_cost = result.cost
            return result.answer
        except Exception as exc:
            logger.error(f"Synchronous query fallback failed: {exc}", exc_info=True)
            return f"Query error: {exc}"

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
        self._show_message(self._get_help_text())

    def _focus_command_input(self) -> None:
        """Focus the command input when the query context is active."""
        if getattr(self.app, "current_context", None) != "query":
            return
        command_input = self.query_one("#query-command-input", CommandInput)
        self.call_after_refresh(command_input.focus)

    def _get_help_text(self) -> str:
        """Build help text for the query screen."""
        return """QUERY COMMANDS

Navigation
/help, /?                     Show this help
/exit, /quit                  Exit the application
/ingest, /i                   Switch to Ingest tab
/query, /q                    Switch to Query tab

Modes and sources
/mode                         Show current query mode
/mode <name>                  Switch to a different query mode
/mode chat|single             Switch conversation mode
/mode set <setting> <on|off>  Toggle model_knowledge, remote, notes, llm_cache

Models and providers
/model                        Show current model selections
/model <task> <provider> <model>  Set model for query/tag/rerank
/models [provider]            List available models
/providers                    Show configured LLM providers
/provider-status              Show provider health status
/vectordb-providers           Show available vector DB providers
/vectordb-status              Show vector DB status
/remote-providers             Show remote providers summary

Agents
/agent start <name> <topic>  Start an agent run
/agent status                Show active agent status
/agent pause                 Pause active agent
/agent resume                Resume active agent
/agent stop                  Stop active agent
/agent log                   Show active agent logs
/agent schedule add <name> <interval> [--params '{"topic":"..."}'] [--concurrency_policy ...] [--confirmation_mode ...]
/agent schedule list         List configured schedules
/agent schedule enable <id>  Enable a schedule
/agent schedule disable <id> Disable a schedule
/agent schedule remove <id>  Remove a schedule
/agent schedule status       Show schedule runtime status

Memory
/memory candidates [status]  List memory candidates (default: pending)
/memory promote <id>         Promote memory candidate
/memory reject <id>          Reject memory candidate
/memory ttl <id> <days>      Edit candidate TTL in days

UI
/ui                          Show UI status
/ui density                  Show density mode and active density
/ui density <mode>           Set density mode: auto|compact|comfortable

Results
/show S#                      Show the cited chunk (e.g., /show S2)
/expand S#                    Show full chunk text (no truncation)
/open S#                      Open cited source file

Notes and citations
/note [filename]              Save a note for the last query
/notes [filename]             Same as /note
/export-citations [fmt] [path]  Export citations (fmt: bibtex|zotero)

Filters and pinned chunks
/load <file_path>             Pin chunks from a file
/load clear                   Clear pinned chunks
/context                      Show context anchors
/context doc <path...>        Set document anchors
/context chunk <id...>        Set chunk anchors
/context clear                Clear context anchors
/set path_contains <substring> [more...]
/set ext_allow .md .pdf
/set ext_deny .txt
/clear path_contains|ext_allow|ext_deny

Costs and diagnostics
/costs                        Show session cost summary
/costs export [path]          Export cost data to CSV
/budget                       Show budget limit
/budget set <amount>          Set budget limit
/cost-estimate <query>        Estimate query cost
/debug                        Show retrieval diagnostics

KEYBOARD SHORTCUTS

Ctrl+N          Switch to Ingest tab
Ctrl+Q          Switch to Query tab
Ctrl+S          Switch to Settings tab
Ctrl+R          Switch to Remote tab
Ctrl+P          Switch to Remote tab
Ctrl+E          Expand selected citation
Ctrl+O          Open selected citation source
Ctrl+Shift+R    Refresh logs
Esc             Clear input
F1              Show help modal
Ctrl+C          Quit

Enter your question to search your knowledge base.
Use /mode to switch between grounded, insight, and hybrid modes."""

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

    def action_refresh_logs(self) -> None:
        """Refresh the log panel from buffered logs."""
        if not hasattr(self.app, "_tui_log_buffer"):
            return
        log_widget = self.query_one("#query-log", RichLog)
        log_widget.clear()
        for message in self.app._tui_log_buffer:
            log_widget.write(f"{message}\n")
        log_widget.scroll_end()

    # Worker/timer lifecycle methods inherited from ManagedScreenMixin.
