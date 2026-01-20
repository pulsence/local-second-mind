"""
Settings screen for LSM TUI.

Displays the currently loaded configuration in editable sections.
"""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, ScrollableContainer, Horizontal
from textual.widgets import Static, Input, Switch, Button, Select, TabbedContent, TabPane
from textual.widget import Widget

from lsm.logging import get_logger

logger = get_logger(__name__)


class SettingsScreen(Widget):
    """
    Settings display screen.

    Shows the current configuration loaded into the app.
    """

    BINDINGS = [
        Binding("ctrl+1", "settings_tab_1", "Config", show=True),
        Binding("ctrl+2", "settings_tab_2", "Ingest", show=True),
        Binding("ctrl+3", "settings_tab_3", "Query", show=True),
        Binding("ctrl+4", "settings_tab_4", "Mode", show=True),
        Binding("ctrl+5", "settings_tab_5", "Vector DB", show=True),
        Binding("ctrl+6", "settings_tab_6", "LLM", show=True),
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Compose the settings screen layout."""
        with Vertical(id="settings-layout"):
            with TabbedContent(id="settings-tabs", initial="settings-config"):
                with TabPane("Configuration (^1)", id="settings-config"):
                    with ScrollableContainer(classes="settings-scroll"):
                        with Container(classes="settings-section"):
                            yield Static("Configuration", classes="settings-section-title")
                            yield self._field(
                                "Config file",
                                "settings-config-path",
                                placeholder="config.json",
                                disabled=True,
                            )

                with TabPane("Ingest (^2)", id="settings-ingest"):
                    with ScrollableContainer(classes="settings-scroll"):
                        with Container(classes="settings-section"):
                            yield Static("Ingest", classes="settings-section-title")
                            yield Static("Roots (one path per line)", classes="settings-label")
                            yield Container(id="settings-ingest-roots-list")
                            with Horizontal(classes="settings-actions"):
                                yield Button("Add root", id="settings-ingest-root-add")
                            yield self._field("Persist dir", "settings-ingest-persist-dir")
                            yield self._field("Collection", "settings-ingest-collection")
                            yield self._field("Embed model", "settings-ingest-embed-model")
                            yield self._field("Device", "settings-ingest-device")
                            yield self._field("Batch size", "settings-ingest-batch-size")
                            yield self._field("Chunk size", "settings-ingest-chunk-size")
                            yield self._field("Chunk overlap", "settings-ingest-chunk-overlap")
                            yield self._field("Tagging model", "settings-ingest-tagging-model")
                            yield self._field("Tags per chunk", "settings-ingest-tags-per-chunk")
                            yield self._field(
                                "Enable OCR",
                                "settings-ingest-enable-ocr",
                                field_type="switch",
                            )
                            yield self._field(
                                "Enable AI tagging",
                                "settings-ingest-enable-ai-tagging",
                                field_type="switch",
                            )

                with TabPane("Query (^3)", id="settings-query"):
                    with ScrollableContainer(classes="settings-scroll"):
                        with Container(classes="settings-section"):
                            yield Static("Query", classes="settings-section-title")
                            yield self._select_field("Mode", "settings-query-mode")
                            yield self._field("k", "settings-query-k")
                            yield self._field("retrieve_k", "settings-query-retrieve-k")
                            yield self._field("min relevance", "settings-query-min-relevance")
                            yield self._field("k_rerank", "settings-query-k-rerank")
                            yield self._field("rerank strategy", "settings-query-rerank-strategy")
                            yield self._field("local_pool", "settings-query-local-pool")
                            yield self._field("max per file", "settings-query-max-per-file")
                            yield self._field("path contains", "settings-query-path-contains")
                            yield self._field("ext allow", "settings-query-ext-allow")
                            yield self._field("ext deny", "settings-query-ext-deny")
                            yield self._field(
                                "no rerank",
                                "settings-query-no-rerank",
                                field_type="switch",
                            )

                with TabPane("Mode (^4)", id="settings-mode"):
                    with ScrollableContainer(classes="settings-scroll"):
                        with Container(classes="settings-section"):
                            yield Static("Selected Mode Settings", classes="settings-section-title")
                            yield self._field("Synthesis style", "settings-mode-synthesis-style", disabled=True)
                            yield self._field("Local enabled", "settings-mode-local-enabled", field_type="switch", disabled=True)
                            yield self._field("Local min relevance", "settings-mode-local-min-relevance", disabled=True)
                            yield self._field("Local k", "settings-mode-local-k", disabled=True)
                            yield self._field("Local k_rerank", "settings-mode-local-k-rerank", disabled=True)
                            yield self._field("Remote enabled", "settings-mode-remote-enabled", field_type="switch", disabled=True)
                            yield self._field("Remote rank strategy", "settings-mode-remote-rank-strategy", disabled=True)
                            yield self._field("Remote max results", "settings-mode-remote-max-results", disabled=True)
                            yield self._field("Remote providers", "settings-mode-remote-providers", disabled=True)
                            yield self._field("Model knowledge enabled", "settings-mode-knowledge-enabled", field_type="switch", disabled=True)
                            yield self._field("Model knowledge label", "settings-mode-knowledge-require-label", field_type="switch", disabled=True)
                            yield self._field("Notes enabled", "settings-mode-notes-enabled", field_type="switch", disabled=True)
                            yield self._field("Notes dir", "settings-mode-notes-dir", disabled=True)
                            yield self._field("Notes template", "settings-mode-notes-template", disabled=True)
                            yield self._field("Notes filename format", "settings-mode-notes-filename-format", disabled=True)
                            yield self._field("Notes integration", "settings-mode-notes-integration", disabled=True)
                            yield self._field("Notes wikilinks", "settings-mode-notes-wikilinks", field_type="switch", disabled=True)
                            yield self._field("Notes backlinks", "settings-mode-notes-backlinks", field_type="switch", disabled=True)
                            yield self._field("Notes include tags", "settings-mode-notes-include-tags", field_type="switch", disabled=True)

                with TabPane("Vector DB (^5)", id="settings-vdb"):
                    with ScrollableContainer(classes="settings-scroll"):
                        with Container(classes="settings-section"):
                            yield Static("Vector DB", classes="settings-section-title")
                            yield self._field("Provider", "settings-vdb-provider")
                            yield self._field("Collection", "settings-vdb-collection")
                            yield self._field("Persist dir", "settings-vdb-persist-dir")
                            yield self._field("HNSW space", "settings-vdb-hnsw-space")
                            yield self._field("Connection string", "settings-vdb-connection-string")
                            yield self._field("Host", "settings-vdb-host")
                            yield self._field("Port", "settings-vdb-port")
                            yield self._field("Database", "settings-vdb-database")
                            yield self._field("User", "settings-vdb-user")
                            yield self._field(
                                "Password",
                                "settings-vdb-password",
                                placeholder="(hidden)",
                            )
                            yield self._field("Index type", "settings-vdb-index-type")
                            yield self._field("Pool size", "settings-vdb-pool-size")

                with TabPane("LLM Providers (^6)", id="settings-llm"):
                    with ScrollableContainer(classes="settings-scroll"):
                        with Container(classes="settings-section"):
                            yield Static("LLM Providers", classes="settings-section-title")
                            yield Container(id="settings-llm-container")

    def on_mount(self) -> None:
        """Populate the settings view after mount."""
        logger.debug("Settings screen mounted")
        self._refresh_settings()
        self._focus_active_tab()

    def _focus_active_tab(self) -> None:
        """Focus the settings tabs when this screen is active."""
        if getattr(self.app, "current_context", None) != "settings":
            return
        tabs = self.query_one("#settings-tabs", TabbedContent)
        self.call_after_refresh(tabs.focus)

    def _refresh_settings(self) -> None:
        """Render the current configuration into the UI."""
        app = self.app

        if not hasattr(app, "config") or app.config is None:
            return

        config = app.config
        self._set_input("settings-config-path", str(config.config_path or ""))

        self._render_ingest_roots([str(path) for path in config.ingest.roots])
        self._set_input("settings-ingest-persist-dir", str(config.ingest.persist_dir))
        self._set_input("settings-ingest-collection", config.ingest.collection)
        self._set_input("settings-ingest-embed-model", config.ingest.embed_model)
        self._set_input("settings-ingest-device", config.ingest.device)
        self._set_input("settings-ingest-batch-size", str(config.ingest.batch_size))
        self._set_input("settings-ingest-chunk-size", str(config.ingest.chunk_size))
        self._set_input("settings-ingest-chunk-overlap", str(config.ingest.chunk_overlap))
        self._set_input("settings-ingest-tagging-model", config.ingest.tagging_model)
        self._set_input("settings-ingest-tags-per-chunk", str(config.ingest.tags_per_chunk))
        self._set_switch("settings-ingest-enable-ocr", config.ingest.enable_ocr)
        self._set_switch("settings-ingest-enable-ai-tagging", config.ingest.enable_ai_tagging)

        self._set_select_options("settings-query-mode", list(config.modes.keys()) if config.modes else [])
        self._set_select_value("settings-query-mode", config.query.mode)
        self._set_input("settings-query-k", str(config.query.k))
        self._set_input("settings-query-retrieve-k", self._format_optional(config.query.retrieve_k))
        self._set_input("settings-query-min-relevance", str(config.query.min_relevance))
        self._set_input("settings-query-k-rerank", str(config.query.k_rerank))
        self._set_input("settings-query-rerank-strategy", config.query.rerank_strategy)
        self._set_input("settings-query-local-pool", self._format_optional(config.query.local_pool))
        self._set_input("settings-query-max-per-file", str(config.query.max_per_file))
        self._set_input("settings-query-path-contains", self._format_list(config.query.path_contains))
        self._set_input("settings-query-ext-allow", self._format_list(config.query.ext_allow))
        self._set_input("settings-query-ext-deny", self._format_list(config.query.ext_deny))
        self._set_switch("settings-query-no-rerank", config.query.no_rerank)
        self._update_mode_settings(config.query.mode)

        self._set_input("settings-vdb-provider", config.vectordb.provider)
        self._set_input("settings-vdb-collection", config.vectordb.collection)
        self._set_input("settings-vdb-persist-dir", str(config.vectordb.persist_dir))
        self._set_input("settings-vdb-hnsw-space", config.vectordb.chroma_hnsw_space)
        self._set_input("settings-vdb-connection-string", self._format_optional(config.vectordb.connection_string))
        self._set_input("settings-vdb-host", self._format_optional(config.vectordb.host))
        self._set_input("settings-vdb-port", self._format_optional(config.vectordb.port))
        self._set_input("settings-vdb-database", self._format_optional(config.vectordb.database))
        self._set_input("settings-vdb-user", self._format_optional(config.vectordb.user))
        self._set_input("settings-vdb-password", "")
        self._set_input("settings-vdb-index-type", config.vectordb.index_type)
        self._set_input("settings-vdb-pool-size", str(config.vectordb.pool_size))

        self._build_llm_sections(config.llm.llms)

    def _build_llm_sections(self, providers) -> None:
        """Create editable sections for each LLM provider."""
        container = self.query_one("#settings-llm-container", Container)
        container.remove_children()

        for index, provider in enumerate(providers, start=1):
            section = Container(
                Static(
                    f"Provider {index}: {provider.provider_name}",
                    classes="settings-subsection-title",
                ),
                self._field("Provider name", f"settings-llm-{index}-provider-name"),
                self._field("Model", f"settings-llm-{index}-model"),
                self._field("Temperature", f"settings-llm-{index}-temperature"),
                self._field("Max tokens", f"settings-llm-{index}-max-tokens"),
                self._field("Base URL", f"settings-llm-{index}-base-url"),
                self._field("Endpoint", f"settings-llm-{index}-endpoint"),
                self._field("API version", f"settings-llm-{index}-api-version"),
                self._field("Deployment name", f"settings-llm-{index}-deployment-name"),
                self._field(
                    "API key",
                    f"settings-llm-{index}-api-key",
                    placeholder="(hidden)",
                ),
                Static("Feature overrides", classes="settings-subsection-title"),
                self._field("Query model", f"settings-llm-{index}-query-model"),
                self._field("Tagging model", f"settings-llm-{index}-tagging-model"),
                self._field("Ranking model", f"settings-llm-{index}-ranking-model"),
                classes="settings-subsection",
            )

            container.mount(section)

            self._set_input(f"settings-llm-{index}-provider-name", provider.provider_name)
            self._set_input(f"settings-llm-{index}-model", self._format_optional(provider.model))
            self._set_input(f"settings-llm-{index}-temperature", self._format_optional(provider.temperature))
            self._set_input(f"settings-llm-{index}-max-tokens", self._format_optional(provider.max_tokens))
            self._set_input(f"settings-llm-{index}-base-url", self._format_optional(provider.base_url))
            self._set_input(f"settings-llm-{index}-endpoint", self._format_optional(provider.endpoint))
            self._set_input(f"settings-llm-{index}-api-version", self._format_optional(provider.api_version))
            self._set_input(f"settings-llm-{index}-deployment-name", self._format_optional(provider.deployment_name))
            self._set_input(f"settings-llm-{index}-api-key", "")

            self._set_input(
                f"settings-llm-{index}-query-model",
                self._format_optional(provider.query.model if provider.query else None),
            )
            self._set_input(
                f"settings-llm-{index}-tagging-model",
                self._format_optional(provider.tagging.model if provider.tagging else None),
            )
            self._set_input(
                f"settings-llm-{index}-ranking-model",
                self._format_optional(provider.ranking.model if provider.ranking else None),
            )

    def _render_ingest_roots(self, roots: list[str]) -> None:
        """Render editable ingest root paths."""
        container = self.query_one("#settings-ingest-roots-list", Container)
        container.remove_children()
        for index, value in enumerate(roots, start=1):
            row = Horizontal(
                Input(value=value, id=f"settings-ingest-root-{index}"),
                Button("Remove", id=f"settings-ingest-root-remove-{index}"),
                classes="settings-field",
            )
            container.mount(row)

    def _collect_ingest_roots(self) -> list[str]:
        """Collect ingest roots from the UI."""
        container = self.query_one("#settings-ingest-roots-list", Container)
        roots: list[str] = []
        for child in container.children:
            input_widget = child.query_one(Input)
            value = input_widget.value.strip()
            if value:
                roots.append(value)
        return roots

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle add/remove actions for ingest roots."""
        button_id = event.button.id or ""
        if button_id == "settings-ingest-root-add":
            roots = self._collect_ingest_roots()
            roots.append("")
            self._render_ingest_roots(roots)
            return
        if button_id.startswith("settings-ingest-root-remove-"):
            roots = self._collect_ingest_roots()
            try:
                index = int(button_id.rsplit("-", 1)[-1]) - 1
            except ValueError:
                return
            if 0 <= index < len(roots):
                roots.pop(index)
            self._render_ingest_roots(roots)

    def _activate_tab(self, tab_id: str) -> None:
        """Switch to a specific settings sub-tab."""
        tabs = self.query_one("#settings-tabs", TabbedContent)
        tabs.active = tab_id

    def action_settings_tab_1(self) -> None:
        """Switch to Configuration tab."""
        self._activate_tab("settings-config")

    def action_settings_tab_2(self) -> None:
        """Switch to Ingest tab."""
        self._activate_tab("settings-ingest")

    def action_settings_tab_3(self) -> None:
        """Switch to Query tab."""
        self._activate_tab("settings-query")

    def action_settings_tab_4(self) -> None:
        """Switch to Mode tab."""
        self._activate_tab("settings-mode")

    def action_settings_tab_5(self) -> None:
        """Switch to Vector DB tab."""
        self._activate_tab("settings-vdb")

    def action_settings_tab_6(self) -> None:
        """Switch to LLM Providers tab."""
        self._activate_tab("settings-llm")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update mode settings when the mode selection changes."""
        if event.select.id == "settings-query-mode":
            self._update_mode_settings(event.value)

    def _update_mode_settings(self, mode_name: Optional[str]) -> None:
        """Update the mode settings panel based on selected mode."""
        app = self.app
        if not hasattr(app, "config") or app.config is None:
            return
        if not mode_name:
            return

        try:
            mode = app.config.get_mode_config(mode_name)
        except Exception:
            return

        self._set_input("settings-mode-synthesis-style", mode.synthesis_style)
        self._set_switch("settings-mode-local-enabled", mode.source_policy.local.enabled)
        self._set_input("settings-mode-local-min-relevance", str(mode.source_policy.local.min_relevance))
        self._set_input("settings-mode-local-k", str(mode.source_policy.local.k))
        self._set_input("settings-mode-local-k-rerank", str(mode.source_policy.local.k_rerank))
        self._set_switch("settings-mode-remote-enabled", mode.source_policy.remote.enabled)
        self._set_input("settings-mode-remote-rank-strategy", mode.source_policy.remote.rank_strategy)
        self._set_input("settings-mode-remote-max-results", str(mode.source_policy.remote.max_results))
        self._set_input(
            "settings-mode-remote-providers",
            self._format_list(mode.source_policy.remote.remote_providers),
        )
        self._set_switch("settings-mode-knowledge-enabled", mode.source_policy.model_knowledge.enabled)
        self._set_switch("settings-mode-knowledge-require-label", mode.source_policy.model_knowledge.require_label)
        self._set_switch("settings-mode-notes-enabled", mode.notes.enabled)
        self._set_input("settings-mode-notes-dir", mode.notes.dir)
        self._set_input("settings-mode-notes-template", mode.notes.template)
        self._set_input("settings-mode-notes-filename-format", mode.notes.filename_format)
        self._set_input("settings-mode-notes-integration", mode.notes.integration)
        self._set_switch("settings-mode-notes-wikilinks", mode.notes.wikilinks)
        self._set_switch("settings-mode-notes-backlinks", mode.notes.backlinks)
        self._set_switch("settings-mode-notes-include-tags", mode.notes.include_tags)

    def _field(
        self,
        label: str,
        field_id: str,
        *,
        placeholder: str = "",
        disabled: bool = False,
        field_type: str = "input",
    ) -> Widget:
        """Create a labeled input/switch field row."""
        label_widget = Static(label, classes="settings-label")
        if field_type == "switch":
            field_widget = Switch(id=field_id)
        else:
            field_widget = Input(placeholder=placeholder, id=field_id)
        if disabled:
            field_widget.disabled = True
        return Horizontal(label_widget, field_widget, classes="settings-field")

    def _select_field(self, label: str, field_id: str) -> Widget:
        """Create a labeled select field row."""
        label_widget = Static(label, classes="settings-label")
        field_widget = Select([], id=field_id)
        return Horizontal(label_widget, field_widget, classes="settings-field")

    def _set_input(self, field_id: str, value: str) -> None:
        """Set a value on an Input if it exists."""
        try:
            widget = self.query_one(f"#{field_id}", Input)
        except Exception:
            return
        widget.value = value or ""

    def _set_switch(self, field_id: str, value: bool) -> None:
        """Set a value on a Switch if it exists."""
        try:
            widget = self.query_one(f"#{field_id}", Switch)
        except Exception:
            return
        widget.value = bool(value)

    def _set_select_options(self, field_id: str, values: list[str]) -> None:
        """Set options on a Select field."""
        try:
            widget = self.query_one(f"#{field_id}", Select)
        except Exception:
            return
        options = [(value, value) for value in values]
        widget.set_options(options)

    def _set_select_value(self, field_id: str, value: str) -> None:
        """Set a selected value on a Select field."""
        try:
            widget = self.query_one(f"#{field_id}", Select)
        except Exception:
            return
        widget.value = value

    def _format_list(self, value: Optional[list[Any]]) -> str:
        """Format a list for display in an input field."""
        if not value:
            return ""
        return ", ".join(str(item) for item in value)

    def _format_optional(self, value: Optional[Any]) -> str:
        """Format optional values for input fields."""
        if value is None:
            return ""
        return str(value)
        ingest = config.ingest
        query = config.query
        vectordb = config.vectordb
        llm = config.llm

        provider_map = llm.get_feature_provider_map()
        mode_names = sorted(config.modes.keys()) if config.modes else []
        remote_providers = config.remote_providers or []

        lines = [
            "CONFIGURATION",
            "=" * 40,
            f"Config file: {config.config_path or 'unknown'}",
            "",
            "INGEST",
            f"Roots: {', '.join(str(p) for p in ingest.roots)}",
            f"Persist dir: {ingest.persist_dir}",
            f"Collection: {ingest.collection}",
            f"Embed model: {ingest.embed_model}",
            f"Device: {ingest.device}",
            f"Batch size: {ingest.batch_size}",
            "",
            "QUERY",
            f"Mode: {query.mode}",
            f"k: {query.k}  k_rerank: {query.k_rerank}",
            f"Min relevance: {query.min_relevance}",
            f"Rerank strategy: {query.rerank_strategy}",
            "",
            "VECTOR DB",
            f"Provider: {vectordb.provider}",
            f"Persist dir: {vectordb.persist_dir}",
            f"Collection: {vectordb.collection}",
            "",
            "LLM",
            f"Providers: {', '.join(llm.get_provider_names()) or 'none'}",
            f"Feature map: {provider_map or 'none'}",
            "",
            "MODES",
            f"Available: {', '.join(mode_names) or 'none'}",
            "",
            "REMOTE PROVIDERS",
            f"Configured: {', '.join(p.name for p in remote_providers) or 'none'}",
        ]

        return "\n".join(lines)
