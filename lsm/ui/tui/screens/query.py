"""
Query screen for LSM TUI.

Provides the query interface with:
- Query input area with history and autocomplete
- Results display with expandable citations
- Remote search results section
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any
import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, RichLog
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.tui.widgets.results import ResultsPanel, CitationSelected, CitationExpanded
from lsm.ui.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.ui.tui.completions import create_completer
from lsm.query.commands import (
    get_command_handlers,
    CommandResult,
    display_provider_name,
    format_feature_label,
)
from lsm.query.session import SessionState
from lsm.query.cost_tracking import CostTracker
from lsm.providers import create_provider
from lsm.vectordb import create_vectordb_provider, list_available_providers
from lsm.remote import get_registered_providers

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
        Binding("ctrl+shift+r", "refresh_logs", "Refresh Logs", show=False),
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
        self.query_one("#query-command-input", CommandInput).focus()
        if hasattr(self.app, "_tui_log_buffer"):
            log_widget = self.query_one("#query-log", RichLog)
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
        parts = command.strip().split()
        if normalized in {"/ingest", "/i"}:
            self.app.action_switch_ingest()
            return
        if normalized in {"/query", "/q"}:
            self.app.action_switch_query()
            return
        if normalized.startswith("/models"):
            self._ensure_query_state()
            output = self._format_models(command.strip())
            if output:
                self._show_message(output)
            return
        if normalized == "/model" and len(parts) == 1:
            self._ensure_query_state()
            output = self._format_model_selection()
            if output:
                self._show_message(output)
            return
        if normalized == "/providers":
            output = self._format_providers()
            if output:
                self._show_message(output)
            return
        if normalized == "/provider-status":
            output = self._format_provider_status()
            if output:
                self._show_message(output)
            return
        if normalized == "/vectordb-providers":
            output = self._format_vectordb_providers()
            if output:
                self._show_message(output)
            return
        if normalized == "/vectordb-status":
            output = self._format_vectordb_status()
            if output:
                self._show_message(output)
            return
        if normalized == "/remote-providers":
            output = self._format_remote_providers()
            if output:
                self._show_message(output)
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

    def _get_feature_configs(self) -> dict[str, Any]:
        feature_map = self.app.config.llm.get_feature_provider_map()
        return {
            "query": (
                self.app.config.llm.get_query_config()
                if "query" in feature_map
                else None
            ),
            "tagging": (
                self.app.config.llm.get_tagging_config()
                if "tagging" in feature_map
                else None
            ),
            "ranking": (
                self.app.config.llm.get_ranking_config()
                if "ranking" in feature_map
                else None
            ),
        }

    def _format_model_selection(self) -> str:
        lines = []
        feature_configs = self._get_feature_configs()
        for feature, cfg in feature_configs.items():
            if cfg is None:
                continue
            label = format_feature_label(feature)
            provider = display_provider_name(cfg.provider)
            lines.append(f"{label}: {provider}/{cfg.model}")
        lines.append("")
        return "\n".join(lines)

    def _format_models(self, command: str) -> str:
        parts = command.split()
        provider_filter = parts[1].strip().lower() if len(parts) > 1 else None
        providers = self.app.config.llm.get_provider_names()
        if provider_filter:
            if provider_filter == "claude":
                providers = [p for p in providers if p in {"claude", "anthropic"}]
            else:
                providers = [p for p in providers if p.lower() == provider_filter]
            if not providers:
                return f"Provider not found in config: {provider_filter}\n"

        lines = []
        seen_labels = set()
        for provider_name in providers:
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            provider_config = self.app.config.llm.get_provider_by_name(provider_name)
            if not provider_config:
                continue
            try:
                test_config = provider_config.resolve_first_available()
                lines.append(f"{label}:")
                if not test_config:
                    lines.append("  (not configured for any feature)\n")
                    continue
                provider = create_provider(test_config)
                ids = provider.list_models()
                ids.sort()
                self.app.query_state.available_models = ids
                if not ids:
                    lines.append("  (no models returned or listing unsupported)")
                else:
                    lines.extend([f"  - {model_id}" for model_id in ids])
                lines.append("")
            except Exception as e:
                logger.error(f"Failed to list models for {provider_name}: {e}")
                lines.append(f"  (failed to list models: {e})\n")
        return "\n".join(lines)

    def _format_providers(self) -> str:
        lines = [
            "",
            "=" * 60,
            "AVAILABLE LLM PROVIDERS",
            "=" * 60,
            "",
        ]

        providers = self.app.config.llm.get_provider_names()

        if not providers:
            lines.append("No providers configured.")
            lines.append("")
            return "\n".join(lines)

        lines.append("Selections:")
        feature_configs = self._get_feature_configs()
        for feature, cfg in feature_configs.items():
            if cfg is None:
                continue
            label = format_feature_label(feature)
            provider = display_provider_name(cfg.provider)
            lines.append(f"  {label:7s} {provider}/{cfg.model}")
        lines.append("")

        lines.append(f"Providers ({len(providers)}):")
        lines.append("")

        seen_labels = set()
        for provider_name in providers:
            try:
                provider_config = self.app.config.llm.get_provider_by_name(provider_name)
                test_config = provider_config.resolve_first_available() if provider_config else None

                if test_config:
                    provider = create_provider(test_config)
                    is_available = "ok" if provider.is_available() else "- (API key not configured)"
                elif provider_config:
                    is_available = "- (no feature config)"
                else:
                    is_available = "- (not configured)"

                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"  {label:20s} {is_available:30s}")

            except Exception as e:
                logger.debug(f"Error checking provider {provider_name}: {e}")
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"  {label:20s} {'- (error)':30s}")

        lines.append("")
        lines.append("To switch providers, update your config.json:")
        lines.append('  "llms": [ { "provider_name": "provider_name", ... } ]')
        lines.append("")

        return "\n".join(lines)

    def _format_provider_status(self) -> str:
        lines = [
            "",
            "=" * 60,
            "PROVIDER HEALTH STATUS",
            "=" * 60,
            "",
        ]

        providers = self.app.config.llm.get_provider_names()
        if not providers:
            lines.append("No providers registered.")
            lines.append("")
            return "\n".join(lines)

        current_provider = self.app.config.llm.get_query_config().provider

        seen_labels = set()
        for provider_name in providers:
            try:
                provider_config = self.app.config.llm.get_provider_by_name(provider_name)
                test_config = provider_config.resolve_first_available() if provider_config else None
                if not test_config:
                    status = "not_configured" if not provider_config else "missing_config"
                    label = display_provider_name(provider_name)
                    if label in seen_labels:
                        continue
                    seen_labels.add(label)
                    lines.append(f"{label:20s} status={status:12s}")
                    continue

                provider = create_provider(test_config)
                health = provider.health_check()

                status = health.get("status", "unknown")
                stats = health.get("stats", {})
                success = stats.get("success_count", 0)
                failure = stats.get("failure_count", 0)
                last_error = stats.get("last_error")
                current_label = " (current)" if provider_name == current_provider else ""
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)

                lines.append(
                    f"{label:20s} status={status:12s} success={success:4d} "
                    f"failure={failure:4d}{current_label}"
                )
                if last_error:
                    lines.append(f"{'':20s} last_error={last_error}")
            except Exception as e:
                logger.debug(f"Error checking provider status {provider_name}: {e}")
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"{label:20s} status=error        error={e}")

        lines.append("")

        return "\n".join(lines)

    def _format_vectordb_providers(self) -> str:
        providers = list_available_providers()

        lines = [
            "",
            "=" * 60,
            "AVAILABLE VECTOR DB PROVIDERS",
            "=" * 60,
            "",
        ]

        if not providers:
            lines.append("No vector DB providers registered.")
            lines.append("")
            return "\n".join(lines)

        current_provider = self.app.config.vectordb.provider
        lines.append(f"Current Provider: {current_provider}")
        lines.append(f"Collection:       {self.app.config.vectordb.collection}")
        lines.append("")

        lines.append(f"Available Providers ({len(providers)}):")
        lines.append("")

        for provider_name in providers:
            is_current = "ACTIVE" if provider_name == current_provider else ""
            status = ""
            if provider_name == current_provider:
                try:
                    provider = create_vectordb_provider(self.app.config.vectordb)
                    status = "ok" if provider.is_available() else "unavailable"
                except Exception as e:
                    status = f"error ({e})"
            lines.append(f"  {provider_name:20s} {status:20s} {is_current}")

        lines.append("")
        lines.append("To switch providers, update your config.json:")
        lines.append('  "vectordb": { "provider": "provider_name", ... }')
        lines.append("")

        return "\n".join(lines)

    def _format_vectordb_status(self) -> str:
        lines = [
            "",
            "=" * 60,
            "VECTOR DB STATUS",
            "=" * 60,
            "",
        ]

        try:
            provider = create_vectordb_provider(self.app.config.vectordb)
            health = provider.health_check()
            stats = provider.get_stats()

            lines.append(f"Provider: {health.get('provider', 'unknown')}")
            lines.append(f"Status:   {health.get('status', 'unknown')}")
            if health.get("error"):
                lines.append(f"Error:    {health.get('error')}")
            lines.append(f"Count:    {stats.get('count', 'n/a')}")
        except Exception as e:
            lines.append(f"Error: {e}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("PROVIDER HEALTH STATUS")
        lines.append("=" * 60)
        lines.append("")

        providers = self.app.config.llm.get_provider_names()
        if not providers:
            lines.append("No providers registered.")
            lines.append("")
            return "\n".join(lines)

        current_provider = self.app.config.llm.get_query_config().provider

        seen_labels = set()
        for provider_name in providers:
            try:
                provider_config = self.app.config.llm.get_provider_by_name(provider_name)
                test_config = provider_config.resolve_first_available() if provider_config else None
                if not test_config:
                    status = "not_configured" if not provider_config else "missing_config"
                    label = display_provider_name(provider_name)
                    if label in seen_labels:
                        continue
                    seen_labels.add(label)
                    lines.append(f"{label:20s} status={status:12s}")
                    continue

                provider = create_provider(test_config)
                health = provider.health_check()

                status = health.get("status", "unknown")
                stats = health.get("stats", {})
                success = stats.get("success_count", 0)
                failure = stats.get("failure_count", 0)
                last_error = stats.get("last_error")
                current_label = " (current)" if provider_name == current_provider else ""
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)

                lines.append(
                    f"{label:20s} status={status:12s} "
                    f"success={success:4d} failure={failure:4d}{current_label}"
                )
                if last_error:
                    lines.append(f"{'':20s} last_error={last_error}")
            except Exception as e:
                logger.debug(f"Error checking provider status {provider_name}: {e}")
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"{label:20s} status=error        error={e}")

        lines.append("")

        return "\n".join(lines)

    def _format_remote_providers(self) -> str:
        lines = [
            "",
            "=" * 60,
            "REMOTE SOURCE PROVIDERS",
            "=" * 60,
            "",
        ]

        registered = get_registered_providers()
        lines.append(f"Registered Provider Types ({len(registered)}):")
        for provider_type, provider_class in sorted(registered.items()):
            lines.append(f"  {provider_type:20s} -> {provider_class.__name__}")
        lines.append("")

        configured = self.app.config.remote_providers or []
        if not configured:
            lines.append("No remote providers configured.")
            lines.append("")
            lines.append("Add providers to your config.json:")
            lines.append('  "remote_providers": [{"name": "...", "type": "...", ...}]')
            lines.append("")
            return "\n".join(lines)

        lines.append(f"Configured Providers ({len(configured)}):")
        lines.append("")
        lines.append(f"  {'NAME':<20s} {'TYPE':<20s} {'STATUS':<10s} {'WEIGHT':<8s} {'API KEY'}")
        lines.append("  " + "-" * 70)

        for provider_config in configured:
            name = provider_config.name
            ptype = provider_config.type
            status = "enabled" if provider_config.enabled else "disabled"
            weight = f"{provider_config.weight:.1f}"
            api_key = provider_config.api_key
            has_key = "set" if api_key and not api_key.startswith("INSERT") else "not set"
            if ptype in {"wikipedia", "arxiv", "openalex", "crossref", "oai_pmh"}:
                has_key = "n/a"
            lines.append(f"  {name:<20s} {ptype:<20s} {status:<10s} {weight:<8s} {has_key}")

        lines.append("")

        mode_config = self.app.config.get_mode_config()
        remote_policy = mode_config.source_policy.remote
        lines.append("Current Mode Remote Settings:")
        lines.append(f"  Enabled:       {remote_policy.enabled}")
        lines.append(f"  Max Results:   {remote_policy.max_results}")
        lines.append(f"  Rank Strategy: {remote_policy.rank_strategy}")
        if remote_policy.remote_providers:
            lines.append(f"  Mode Providers: {', '.join(str(p) for p in remote_policy.remote_providers)}")
        lines.append("")

        lines.append("Commands:")
        lines.append("  /remote-search <provider> <query>  Test a provider")
        lines.append("  /remote-search-all <query>         Search all enabled providers")
        lines.append("  /remote-provider enable <name>     Enable a provider")
        lines.append("  /remote-provider disable <name>    Disable a provider")
        lines.append("  /remote-provider weight <name> <value>  Set provider weight")
        lines.append("")

        return "\n".join(lines)

    def _execute_query_command(self, command: str) -> CommandResult:
        """Run the query command handler and capture output."""
        try:
            q = command.strip()
            ql = q.lower()
            for handler in get_command_handlers():
                result = handler(
                    q,
                    ql,
                    self.app.query_state,
                    self.app.config,
                    self.app.query_embedder,
                    self.app.query_provider,
                )
                if result is not None:
                    return result
            return CommandResult(output="", handled=False)
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
            from lsm.query.execution import run_query_turn_async

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
Ctrl+P          Switch to Remote tab
Ctrl+Shift+R    Refresh logs
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

    def action_refresh_logs(self) -> None:
        """Refresh the log panel from buffered logs."""
        if not hasattr(self.app, "_tui_log_buffer"):
            return
        log_widget = self.query_one("#query-log", RichLog)
        log_widget.clear()
        for message in self.app._tui_log_buffer:
            log_widget.write(f"{message}\n")
        log_widget.scroll_end()
