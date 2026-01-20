"""
Remote providers screen for LSM TUI.

Displays configured remote providers and allows running a test query against one.
"""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Input, Button, Select, RichLog
from textual.widget import Widget
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.utils import run_remote_search
from lsm.remote import get_registered_providers

logger = get_logger(__name__)


class RemoteScreen(Widget):
    """
    Remote providers screen.

    Shows configured providers and runs test searches.
    """

    is_loading: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        """Compose the remote providers layout."""
        with Vertical(id="remote-layout"):
            with Horizontal(id="remote-top"):
                with Vertical(id="remote-left"):
                    with Container(id="remote-provider-panel"):
                        yield Static("Remote Providers", classes="remote-section-title")
                        yield Static("", id="remote-provider-list", markup=False)
                    with Container(id="remote-log-panel"):
                        yield Static("Logs", classes="remote-section-title")
                        yield RichLog(id="remote-log", auto_scroll=True, wrap=True)

                with Container(id="remote-results-panel"):
                    yield Static("Remote Results", classes="remote-section-title")
                    with ScrollableContainer(id="remote-results-scroll"):
                        yield Static(
                            "Run a search to see results.",
                            id="remote-results-output",
                            markup=False,
                        )

            with Horizontal(id="remote-controls"):
                yield Static("Provider", classes="remote-label")
                yield Select([], id="remote-provider-select")
                yield Static("Query", classes="remote-label")
                yield Input(
                    placeholder="search query",
                    id="remote-query-input",
                )
                yield Button("Search", id="remote-search-button", variant="primary")
                yield Button("Refresh", id="remote-refresh-button")

    def on_mount(self) -> None:
        """Initialize provider list and focus."""
        logger.debug("Remote screen mounted")
        self._refresh_provider_list()
        self._focus_default_input()
        if hasattr(self.app, "_tui_log_buffer"):
            log_widget = self.query_one("#remote-log", RichLog)
            for message in self.app._tui_log_buffer:
                log_widget.write(f"{message}\n")
            log_widget.scroll_end()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id or ""
        if button_id == "remote-search-button":
            self.run_worker(self._run_search(), exclusive=True)
            return
        if button_id == "remote-refresh-button":
            self._refresh_provider_list()
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter from input fields."""
        if event.input.id == "remote-query-input":
            self.run_worker(self._run_search(), exclusive=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Move focus to query input after provider selection."""
        if event.select.id == "remote-provider-select":
            if getattr(self.app, "current_context", None) == "remote":
                self.query_one("#remote-query-input", Input).focus()

    def _focus_default_input(self) -> None:
        """Focus the provider input unless already set."""
        if getattr(self.app, "current_context", None) != "remote":
            return
        provider_select = self.query_one("#remote-provider-select", Select)
        query_input = self.query_one("#remote-query-input", Input)
        if provider_select.value:
            query_input.focus()
        else:
            provider_select.focus()

    def _refresh_provider_list(self) -> None:
        """Refresh the provider list display."""
        output = self._format_provider_list()
        self.query_one("#remote-provider-list", Static).update(output)
        self._ensure_provider_options()

    def _ensure_provider_options(self) -> None:
        """Set provider select options from config."""
        provider_select = self.query_one("#remote-provider-select", Select)
        providers = getattr(self.app.config, "remote_providers", None) or []
        options = [(provider.name, provider.name) for provider in providers]
        provider_select.set_options(options)
        if options and not provider_select.value:
            provider_select.value = options[0][0]

    def _format_provider_list(self) -> str:
        """Format configured providers for display."""
        config = self.app.config
        providers = config.remote_providers or []
        mode_config = config.get_mode_config()
        remote_policy = mode_config.source_policy.remote
        registered = sorted(get_registered_providers().keys())

        lines = [
            "Configured Providers",
            "=" * 40,
        ]

        if not providers:
            lines.append("No remote providers configured.")
            lines.append("Add to config.json: remote_providers")
        else:
            lines.append(f"{'NAME':<18s} {'TYPE':<16s} {'STATUS':<9s} {'WEIGHT':<6s}")
            lines.append("-" * 60)
            for provider in providers:
                status = "enabled" if provider.enabled else "disabled"
                weight = f"{provider.weight:.2f}"
                lines.append(
                    f"{provider.name:<18s} {provider.type:<16s} {status:<9s} {weight:<6s}"
                )

        lines.append("")
        lines.append("Mode Settings")
        lines.append("-" * 40)
        lines.append(f"Mode:          {config.query.mode}")
        lines.append(f"Remote enabled: {remote_policy.enabled}")
        lines.append(f"Max results:    {remote_policy.max_results}")
        lines.append(f"Rank strategy:  {remote_policy.rank_strategy}")
        if remote_policy.remote_providers:
            names = ", ".join(str(p) for p in remote_policy.remote_providers)
            lines.append(f"Mode providers: {names}")
        else:
            lines.append("Mode providers: (all enabled)")

        lines.append("")
        lines.append("Registered Types")
        lines.append("-" * 40)
        lines.append(", ".join(registered) if registered else "none")

        return "\n".join(lines)

    def _show_message(self, message: str) -> None:
        """Update the results output panel."""
        self.query_one("#remote-results-output", Static).update(message)

    async def _run_search(self) -> None:
        """Run a remote provider search."""
        provider = self.query_one("#remote-provider-select", Select).value or ""
        query = self.query_one("#remote-query-input", Input).value.strip()

        if not provider or not query:
            self._show_message("Enter both a provider name and a query.")
            return

        self.is_loading = True
        self._show_message(f"Searching {provider} for: {query} ...")

        try:
            output = await asyncio.to_thread(
                run_remote_search,
                provider,
                query,
                self.app.config,
            )
            self._show_message(output or "No output returned.")
        except Exception as exc:
            logger.error(f"Remote search failed: {exc}", exc_info=True)
            self._show_message(f"Remote search failed: {exc}")
        finally:
            self.is_loading = False
