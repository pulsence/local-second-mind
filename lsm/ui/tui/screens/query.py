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
from pathlib import Path
from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, RichLog, TabbedContent
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.tui.widgets.results import ResultsPanel, CitationSelected, CitationExpanded
from lsm.ui.tui.widgets.input import CommandInput, CommandSubmitted
from lsm.ui.tui.completions import create_completer
from lsm.query.cost_tracking import estimate_query_cost
from lsm.ui.utils import (
    display_provider_name,
    format_feature_label,
    open_file,
)
from lsm.query.session import SessionState
from lsm.query.cost_tracking import CostTracker
from lsm.providers import create_provider
from lsm.vectordb import create_vectordb_provider, list_available_providers
from lsm.remote import get_registered_providers
from lsm.vectordb.utils import require_chroma_collection
from lsm.query.citations import export_citations_from_note, export_citations_from_sources
from lsm.query.notes import get_note_filename, generate_note_content
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

        return "\n".join(lines)

    def _execute_query_command(self, command: str) -> CommandResult:
        """Run the query command handler and capture output."""
        try:
            q = command.strip()
            if not q.startswith("/"):
                return CommandResult(output="", handled=False)

            parts = q.split()
            cmd = parts[0].lower()

            if cmd in {"/exit"}:
                return CommandResult(output="", should_exit=True)

            if cmd in {"/help", "/?"}:
                return CommandResult(output=self._get_help_text())

            if cmd == "/debug":
                return CommandResult(output=self.app.query_state.format_debug())

            if cmd == "/costs":
                tracker = self.app.query_state.cost_tracker
                if not tracker:
                    return CommandResult(output="Cost tracking is not initialized.\n")
                if len(parts) == 1:
                    return CommandResult(output=self.app.query_state.format_costs())
                if len(parts) >= 2 and parts[1].lower() == "export":
                    if len(parts) >= 3:
                        export_path = Path(parts[2])
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        export_path = Path(f"costs-{timestamp}.csv")
                    try:
                        tracker.export_csv(export_path)
                        return CommandResult(output=f"Cost data exported to: {export_path}\n")
                    except Exception as exc:
                        return CommandResult(output=f"Failed to export costs: {exc}\n")
                return CommandResult(output="Usage:\n  /costs\n  /costs export <path>\n")

            if cmd == "/budget":
                tracker = self.app.query_state.cost_tracker
                if not tracker:
                    return CommandResult(output="Cost tracking is not initialized.\n")
                if len(parts) == 1:
                    if tracker.budget_limit is None:
                        return CommandResult(output="No budget set.\n")
                    return CommandResult(output=f"Budget limit: ${tracker.budget_limit:.4f}\n")
                if len(parts) == 3 and parts[1].lower() == "set":
                    try:
                        tracker.budget_limit = float(parts[2])
                        return CommandResult(
                            output=f"Budget limit set to: ${tracker.budget_limit:.4f}\n"
                        )
                    except ValueError:
                        return CommandResult(output="Invalid budget amount. Use a numeric value.\n")
                return CommandResult(output="Usage:\n  /budget\n  /budget set <amount>\n")

            if cmd == "/cost-estimate":
                if len(parts) < 2:
                    return CommandResult(output="Usage: /cost-estimate <query>\n")
                query = q.split(maxsplit=1)[1].strip()
                cost = estimate_query_cost(
                    query,
                    self.app.config,
                    self.app.query_state,
                    self.app.query_embedder,
                    self.app.query_provider,
                )
                return CommandResult(output=f"Estimated cost: ${cost:.4f}\n")

            if cmd == "/export-citations":
                fmt = parts[1].strip().lower() if len(parts) > 1 else "bibtex"
                note_path = Path(parts[2]) if len(parts) > 2 else None
                if fmt not in {"bibtex", "zotero"}:
                    return CommandResult(output="Format must be 'bibtex' or 'zotero'.\n")
                try:
                    if note_path:
                        output_path = export_citations_from_note(note_path, fmt=fmt)
                    else:
                        if not self.app.query_state.last_label_to_candidate:
                            return CommandResult(
                                output="No last query sources available to export.\n"
                            )
                        sources = [
                            {
                                "source_path": c.source_path,
                                "source_name": c.source_name,
                                "chunk_index": c.chunk_index,
                                "ext": c.ext,
                                "label": label,
                                "title": (c.meta or {}).get("title"),
                                "author": (c.meta or {}).get("author"),
                                "mtime_ns": (c.meta or {}).get("mtime_ns"),
                                "ingested_at": (c.meta or {}).get("ingested_at"),
                            }
                            for label, c in self.app.query_state.last_label_to_candidate.items()
                        ]
                        output_path = export_citations_from_sources(sources, fmt=fmt)
                    return CommandResult(output=f"Citations exported to: {output_path}\n")
                except Exception as exc:
                    return CommandResult(output=f"Failed to export citations: {exc}\n")

            if cmd == "/model":
                if len(parts) != 4:
                    return CommandResult(
                        output=(
                            "Usage:\n"
                            "  /model                   (show current)\n"
                            "  /model <task> <provider> <model>  (set model for a task)\n"
                            "  /models [provider]       (list available models)\n"
                        )
                    )
                task = parts[1].strip().lower()
                provider_name = parts[2].strip()
                model_name = parts[3].strip()
                task_map = {
                    "query": "query",
                    "tag": "tagging",
                    "tagging": "tagging",
                    "rerank": "ranking",
                    "ranking": "ranking",
                }
                feature = task_map.get(task)
                if not feature:
                    return CommandResult(output="Unknown task. Use: query, tag, rerank\n")
                try:
                    provider_names = self.app.config.llm.get_provider_names()
                    normalized = provider_name
                    if provider_name == "anthropic" and "claude" in provider_names:
                        normalized = "claude"
                    elif provider_name == "claude" and "anthropic" in provider_names:
                        normalized = "anthropic"

                    self.app.config.llm.set_feature_selection(feature, normalized, model_name)
                    if feature == "query":
                        self.app.query_state.model = model_name
                    label = format_feature_label(feature)
                    return CommandResult(
                        output=(
                            f"Model set: {label} = {display_provider_name(normalized)}/{model_name}\n"
                        )
                    )
                except Exception as exc:
                    return CommandResult(output=f"Failed to set model: {exc}\n")

            if cmd == "/mode":
                if len(parts) == 1:
                    current_mode = self.app.config.query.mode
                    mode_config = self.app.config.get_mode_config(current_mode)
                    lines = [
                        f"Current mode: {current_mode}",
                        f"  Synthesis style: {mode_config.synthesis_style}",
                        f"  Local sources: enabled (k={mode_config.source_policy.local.k})",
                        f"  Remote sources: {'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}",
                        f"  Model knowledge: "
                        f"{'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}",
                        f"  Notes: {'enabled' if mode_config.notes.enabled else 'disabled'}",
                        f"\nAvailable modes: {', '.join(self.app.config.modes.keys())}\n",
                    ]
                    return CommandResult(output="\n".join(lines))
                if parts[1].lower() == "set":
                    if len(parts) != 4:
                        return CommandResult(
                            output=(
                                "Usage:\n  /mode set <setting> <on|off>\n"
                                "Settings: model_knowledge, remote, notes\n"
                            )
                        )
                    setting = parts[2].strip().lower()
                    value = parts[3].strip().lower()
                    enabled = value in {"on", "true", "yes", "1"}
                    mode_config = self.app.config.get_mode_config()
                    if setting in {"model_knowledge", "model-knowledge"}:
                        mode_config.source_policy.model_knowledge.enabled = enabled
                    elif setting in {"remote", "remote_sources", "remote-sources"}:
                        mode_config.source_policy.remote.enabled = enabled
                    elif setting in {"notes"}:
                        mode_config.notes.enabled = enabled
                    else:
                        return CommandResult(
                            output=(
                                f"Unknown setting: {setting}\n"
                                "Settings: model_knowledge, remote, notes\n"
                            )
                        )
                    return CommandResult(
                        output=f"Mode setting '{setting}' set to: {'on' if enabled else 'off'}\n"
                    )
                if len(parts) != 2:
                    return CommandResult(
                        output=(
                            "Usage:\n  /mode           (show current)\n"
                            "  /mode <name>    (switch to a different mode)\n"
                        )
                    )
                mode_name = parts[1].strip()
                if mode_name not in self.app.config.modes:
                    return CommandResult(
                        output=(
                            f"Mode not found: {mode_name}\n"
                            f"Available modes: {', '.join(self.app.config.modes.keys())}\n"
                        )
                    )
                self.app.config.query.mode = mode_name
                mode_config = self.app.config.get_mode_config(mode_name)
                lines = [
                    f"Mode switched to: {mode_name}",
                    f"  Synthesis style: {mode_config.synthesis_style}",
                    f"  Remote sources: "
                    f"{'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}",
                    f"  Model knowledge: "
                    f"{'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}\n",
                ]
                return CommandResult(output="\n".join(lines))

            if cmd in {"/note", "/notes"}:
                if not self.app.query_state.last_question:
                    return CommandResult(output="No query to save. Run a query first.\n")
                try:
                    mode_config = self.app.config.get_mode_config()
                    notes_config = mode_config.notes

                    if self.app.config.config_path:
                        base_dir = self.app.config.config_path.parent
                        notes_dir = base_dir / notes_config.dir
                    else:
                        notes_dir = Path(notes_config.dir)

                    notes_dir.mkdir(parents=True, exist_ok=True)

                    content = generate_note_content(
                        query=self.app.query_state.last_question,
                        answer=self.app.query_state.last_answer or "No answer generated",
                        local_sources=self.app.query_state.last_local_sources_for_notes,
                        remote_sources=self.app.query_state.last_remote_sources,
                        mode=self.app.config.query.mode,
                        use_wikilinks=notes_config.wikilinks,
                        include_backlinks=notes_config.backlinks,
                        include_tags=notes_config.include_tags,
                    )

                    filename_override = q.split(maxsplit=1)[1].strip() if len(parts) > 1 else None
                    if filename_override:
                        filename = filename_override
                        if not filename.lower().endswith(".md"):
                            filename += ".md"
                        note_path = Path(filename)
                        if not note_path.is_absolute():
                            note_path = notes_dir / note_path
                    else:
                        filename = get_note_filename(
                            self.app.query_state.last_question,
                            format=notes_config.filename_format,
                        )
                        note_path = notes_dir / filename

                    note_path.write_text(content, encoding="utf-8")
                    return CommandResult(output=f"Note saved to: {note_path}\n")
                except Exception as exc:
                    logger.error(f"Note save error: {exc}")
                    return CommandResult(output=f"Failed to save note: {exc}\n")

            if cmd == "/load":
                if len(parts) < 2:
                    return CommandResult(
                        output=(
                            "Usage: /load <file_path>\n"
                            "Example: /load /docs/important.md\n\n"
                            "This pins a document for forced inclusion in next query context.\n"
                            "Use /load clear to clear pinned chunks.\n"
                        )
                    )
                arg = q.split(maxsplit=1)[1].strip()
                if arg.lower() == "clear":
                    self.app.query_state.pinned_chunks = []
                    return CommandResult(output="Cleared all pinned chunks.\n")
                file_path = arg
                lines = [f"Loading chunks from: {file_path}", "Searching collection..."]
                try:
                    chroma = require_chroma_collection(self.app.query_provider, "/load")
                    results = chroma.get(
                        where={"source_path": {"$eq": file_path}},
                        include=["metadatas"],
                    )
                    if not results or not results.get("ids"):
                        lines.append(f"\nNo chunks found for path: {file_path}")
                        lines.append("Tip: Path must match exactly. Use /explore to find exact paths.\n")
                        return CommandResult(output="\n".join(lines))

                    chunk_ids = results["ids"]
                    for chunk_id in chunk_ids:
                        if chunk_id not in self.app.query_state.pinned_chunks:
                            self.app.query_state.pinned_chunks.append(chunk_id)

                    lines.append(f"\nPinned {len(chunk_ids)} chunks from {file_path}")
                    lines.append(f"Total pinned chunks: {len(self.app.query_state.pinned_chunks)}")
                    lines.append("\nThese chunks will be forcibly included in your next query.")
                    lines.append("Use /load clear to unpin all chunks.\n")
                except Exception as exc:
                    lines.append(f"Error loading chunks: {exc}\n")
                    logger.error(f"Load command error: {exc}")
                return CommandResult(output="\n".join(lines))

            if cmd in {"/show", "/expand"}:
                label = parts[1].strip() if len(parts) > 1 else None
                expanded = cmd == "/expand"
                if not label:
                    usage = "/show S#   (e.g., /show S2)" if not expanded else "/expand S#   (e.g., /expand S2)"
                    return CommandResult(output=f"Usage: {usage}\n")
                normalized_label = label.strip().upper()
                candidate = self.app.query_state.last_label_to_candidate.get(normalized_label)
                if not candidate:
                    return CommandResult(output=f"No such label in last results: {normalized_label}\n")
                output = candidate.format(label=normalized_label, expanded=expanded)
                return CommandResult(output=output)

            if cmd == "/open":
                label = parts[1].strip() if len(parts) > 1 else None
                if not label:
                    return CommandResult(output="Usage: /open S#   (e.g., /open S2)\n")
                normalized_label = label.strip().upper()
                candidate = self.app.query_state.last_label_to_candidate.get(normalized_label)
                if not candidate:
                    return CommandResult(output=f"No such label in last results: {normalized_label}\n")
                path = (candidate.meta or {}).get("source_path")
                if not path:
                    return CommandResult(output="No source_path available for this citation.\n")
                if open_file(path):
                    return CommandResult(output=f"Opened: {path}\n")
                return CommandResult(output=f"Failed to open file: {path}\n")

            if cmd == "/set":
                key = parts[1].strip() if len(parts) > 1 else None
                values = parts[2:] if len(parts) > 2 else []
                if not key or not values:
                    return CommandResult(
                        output=(
                            "Usage:\n  /set path_contains <substring> [more...]\n"
                            "  /set ext_allow .md .pdf\n"
                            "  /set ext_deny .txt\n"
                        )
                    )
                if key == "path_contains":
                    self.app.query_state.path_contains = values if len(values) > 1 else values[0]
                    return CommandResult(
                        output=f"path_contains set to: {self.app.query_state.path_contains}\n"
                    )
                if key == "ext_allow":
                    self.app.query_state.ext_allow = values
                    return CommandResult(output=f"ext_allow set to: {self.app.query_state.ext_allow}\n")
                if key == "ext_deny":
                    self.app.query_state.ext_deny = values
                    return CommandResult(output=f"ext_deny set to: {self.app.query_state.ext_deny}\n")
                return CommandResult(output=f"Unknown filter key: {key}\n")

            if cmd == "/clear":
                key = parts[1].strip() if len(parts) > 1 else None
                if not key:
                    return CommandResult(output="Usage: /clear path_contains|ext_allow|ext_deny\n")
                if key == "path_contains":
                    self.app.query_state.path_contains = None
                    return CommandResult(output="path_contains cleared.\n")
                if key == "ext_allow":
                    self.app.query_state.ext_allow = None
                    return CommandResult(output="ext_allow cleared.\n")
                if key == "ext_deny":
                    self.app.query_state.ext_deny = None
                    return CommandResult(output="ext_deny cleared.\n")
                return CommandResult(output=f"Unknown filter key: {key}\n")

            return CommandResult(output=self._get_help_text())
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
            from lsm.query.api import QueryProgress, query as run_query

            def on_progress(progress: QueryProgress) -> None:
                message = f"{progress.stage}: {progress.message}"

                def update() -> None:
                    try:
                        status_bar = self.app.query_one("#main-status-bar", StatusBar)
                        status_bar.provider_status = message
                    except Exception:
                        pass

                self.app.call_from_thread(update)

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
/mode set <setting> <on|off>  Toggle model_knowledge, remote, notes

Models and providers
/model                        Show current model selections
/model <task> <provider> <model>  Set model for query/tag/rerank
/models [provider]            List available models
/providers                    Show configured LLM providers
/provider-status              Show provider health status
/vectordb-providers           Show available vector DB providers
/vectordb-status              Show vector DB status
/remote-providers             Show remote providers summary

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
