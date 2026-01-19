"""
Settings screen for LSM TUI.

Provides configuration and status display:
- Mode selector
- Provider status display
- Configuration options
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List

from textual.app import ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Static, Select, Switch, Label
from textual.widget import Widget
from textual.reactive import reactive

from lsm.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SettingsScreen(Widget):
    """
    Settings and configuration interface.

    Provides:
    - Query mode selection
    - LLM provider status
    - Vector DB status
    - Remote provider configuration
    """

    # Reactive state
    current_mode: reactive[str] = reactive("grounded")

    def compose(self) -> ComposeResult:
        """Compose the settings screen layout."""
        with ScrollableContainer(classes="settings-panel"):
            # Mode Selection Section
            with Container(classes="settings-section"):
                yield Static("Query Mode", classes="settings-section-title")
                yield Select(
                    [
                        ("Grounded - Strict retrieval-based answers", "grounded"),
                        ("Insight - Creative synthesis with model knowledge", "insight"),
                        ("Hybrid - Balanced approach", "hybrid"),
                    ],
                    id="mode-select",
                    value="grounded",
                    allow_blank=False,
                )

            # Mode Settings Section
            with Container(classes="settings-section"):
                yield Static("Mode Settings", classes="settings-section-title")

                with Container(id="mode-settings"):
                    yield Label("Use model knowledge:")
                    yield Switch(id="switch-model-knowledge", value=False)

                    yield Label("Enable remote sources:")
                    yield Switch(id="switch-remote", value=False)

                    yield Label("Include notes:")
                    yield Switch(id="switch-notes", value=True)

            # LLM Provider Status Section
            with Container(classes="settings-section"):
                yield Static("LLM Providers", classes="settings-section-title")
                yield Static(
                    "Loading provider status...",
                    id="provider-status",
                )

            # Vector DB Status Section
            with Container(classes="settings-section"):
                yield Static("Vector Database", classes="settings-section-title")
                yield Static(
                    "Loading vector DB status...",
                    id="vectordb-status",
                )

            # Remote Providers Section
            with Container(classes="settings-section"):
                yield Static("Remote Providers", classes="settings-section-title")
                yield Static(
                    "Loading remote provider status...",
                    id="remote-status",
                )

            # Session Info Section
            with Container(classes="settings-section"):
                yield Static("Session Information", classes="settings-section-title")
                yield Static(
                    "Loading session info...",
                    id="session-info",
                )

    def on_mount(self) -> None:
        """Handle screen mount."""
        logger.debug("Settings screen mounted")
        self.run_worker(self._refresh_all_status(), exclusive=True)

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle mode selection change."""
        if event.select.id == "mode-select":
            mode = str(event.value)
            self.current_mode = mode
            self.app.current_mode = mode
            self.app.notify(f"Switched to {mode} mode")
            logger.info(f"Mode changed to: {mode}")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch toggle."""
        switch_id = event.switch.id
        value = event.value

        if switch_id == "switch-model-knowledge":
            logger.info(f"Model knowledge: {value}")
            self.app.notify(f"Model knowledge: {'enabled' if value else 'disabled'}")
        elif switch_id == "switch-remote":
            logger.info(f"Remote sources: {value}")
            self.app.notify(f"Remote sources: {'enabled' if value else 'disabled'}")
        elif switch_id == "switch-notes":
            logger.info(f"Notes: {value}")
            self.app.notify(f"Notes: {'enabled' if value else 'disabled'}")

    async def _refresh_all_status(self) -> None:
        """Refresh all status displays."""
        await self._refresh_provider_status()
        await self._refresh_vectordb_status()
        await self._refresh_remote_status()
        self._refresh_session_info()

    async def _refresh_provider_status(self) -> None:
        """Refresh LLM provider status display."""
        status_widget = self.query_one("#provider-status", Static)

        try:
            app = self.app
            config = app.config

            lines = []

            # Get feature provider map
            feature_map = config.llm.get_feature_provider_map()

            for feature in ("query", "tagging", "ranking"):
                if feature not in feature_map:
                    continue

                cfg = {
                    "query": config.llm.get_query_config(),
                    "tagging": config.llm.get_tagging_config(),
                    "ranking": config.llm.get_ranking_config(),
                }.get(feature)

                if cfg:
                    provider = cfg.provider
                    if provider in {"anthropic", "claude"}:
                        provider = "claude"
                    model = cfg.model
                    label = {"query": "Query", "tagging": "Tagging", "ranking": "Ranking"}[feature]
                    lines.append(f"{label}: {provider}/{model}")

            if lines:
                status_widget.update("\n".join(lines))
            else:
                status_widget.update("No providers configured")

        except Exception as e:
            logger.warning(f"Failed to get provider status: {e}")
            status_widget.update(f"Error: {e}")

    async def _refresh_vectordb_status(self) -> None:
        """Refresh vector DB status display."""
        status_widget = self.query_one("#vectordb-status", Static)

        try:
            app = self.app
            config = app.config

            provider_type = config.vectordb.provider
            collection = config.collection

            # Try to get count if provider is initialized
            count = "N/A"
            if hasattr(app, '_query_provider') and app._query_provider is not None:
                try:
                    count = f"{app._query_provider.count():,}"
                except Exception:
                    pass

            status_widget.update(
                f"Provider: {provider_type}\n"
                f"Collection: {collection}\n"
                f"Chunks: {count}"
            )

        except Exception as e:
            logger.warning(f"Failed to get vectordb status: {e}")
            status_widget.update(f"Error: {e}")

    async def _refresh_remote_status(self) -> None:
        """Refresh remote provider status display."""
        status_widget = self.query_one("#remote-status", Static)

        try:
            app = self.app
            config = app.config

            if not config.remote_providers:
                status_widget.update("No remote providers configured")
                return

            lines = []
            for provider in config.remote_providers:
                status = "enabled" if provider.enabled else "disabled"
                lines.append(f"{provider.name}: {status}")

            status_widget.update("\n".join(lines))

        except Exception as e:
            logger.warning(f"Failed to get remote status: {e}")
            status_widget.update(f"Error: {e}")

    def _refresh_session_info(self) -> None:
        """Refresh session information display."""
        info_widget = self.query_one("#session-info", Static)

        try:
            app = self.app

            cost = getattr(app, 'total_cost', 0.0)
            chunks = getattr(app, 'chunk_count', 0)
            mode = getattr(app, 'current_mode', 'unknown')

            info_widget.update(
                f"Mode: {mode}\n"
                f"Chunks: {chunks:,}\n"
                f"Session cost: ${cost:.4f}"
            )

        except Exception as e:
            info_widget.update(f"Error: {e}")
