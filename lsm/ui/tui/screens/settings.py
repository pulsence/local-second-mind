"""
Settings screen for LSM TUI.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch, TabbedContent, TabPane

from lsm.config.loader import load_config_from_file, save_config_to_file
from lsm.config.models import RemoteProviderRef
from lsm.logging import get_logger

logger = get_logger(__name__)


class SettingsScreen(Widget):
    """Settings editor aligned to current config sections."""

    current_mode: str = "grounded"
    """Current selected mode name (for tests and UI state)."""

    BINDINGS = [
        Binding("ctrl+o", "settings_tab_1", "Global", show=True),
        Binding("ctrl+g", "settings_tab_2", "Ingest", show=True),
        Binding("ctrl+q", "settings_tab_3", "Query", show=True),
        Binding("ctrl+l", "settings_tab_4", "LLM", show=True),
        Binding("ctrl+b", "settings_tab_5", "Vector DB", show=True),
        Binding("ctrl+d", "settings_tab_6", "Modes", show=True),
        Binding("ctrl+r", "settings_tab_7", "Remote", show=True),
        Binding("ctrl+n", "settings_tab_8", "Chats/Notes", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-layout"):
            yield Static("", id="settings-status")
            with TabbedContent(id="settings-tabs", initial="settings-global"):
                with TabPane("Global (^O)", id="settings-global"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._global_section()
                with TabPane("Ingest (^G)", id="settings-ingest"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._ingest_section()
                with TabPane("Query (^Q)", id="settings-query"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._query_section()
                with TabPane("LLM (^L)", id="settings-llm"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._llm_section()
                with TabPane("Vector DB (^B)", id="settings-vdb"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._vdb_section()
                with TabPane("Modes (^D)", id="settings-modes"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._modes_section()
                with TabPane("Remote (^R)", id="settings-remote"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._remote_section()
                with TabPane("Chats/Notes (^N)", id="settings-chats-notes"):
                    with ScrollableContainer(classes="settings-scroll"):
                        yield self._chats_notes_section()

    def on_mount(self) -> None:
        self._refresh_settings()
        if getattr(self.app, "current_context", None) == "settings":
            tabs = self.query_one("#settings-tabs", TabbedContent)
            self.call_after_refresh(tabs.focus)

    def _global_section(self) -> Widget:
        return Container(
            Static("Global Settings", classes="settings-section-title"),
            self._field("Config file", "settings-config-path", disabled=True),
            self._field("Global folder", "settings-global-folder"),
            self._field("Embed model", "settings-global-embed-model"),
            self._field("Device", "settings-global-device"),
            self._field("Batch size", "settings-global-batch-size"),
            self._field("Embedding dimension", "settings-global-embedding-dimension"),
            self._save_reset_row("global"),
            classes="settings-section",
        )

    def _ingest_section(self) -> Widget:
        return Container(
            Static("Ingest Settings", classes="settings-section-title"),
            self._field("Roots (JSON)", "settings-ingest-roots-json"),
            self._field("Persist dir", "settings-ingest-persist-dir"),
            self._field("Collection", "settings-ingest-collection"),
            self._select_field("Chunking strategy", "settings-ingest-chunking-strategy", [("structure", "structure"), ("fixed", "fixed")]),
            self._field("Chunk size", "settings-ingest-chunk-size"),
            self._field("Chunk overlap", "settings-ingest-chunk-overlap"),
            self._field("Tags per chunk", "settings-ingest-tags-per-chunk"),
            self._field("Translation target", "settings-ingest-translation-target"),
            self._field("Max files", "settings-ingest-max-files"),
            self._field("Max seconds", "settings-ingest-max-seconds"),
            self._field("Enable OCR", "settings-ingest-enable-ocr", field_type="switch"),
            self._field("Enable AI tagging", "settings-ingest-enable-ai-tagging", field_type="switch"),
            self._field("Enable language detection", "settings-ingest-enable-language-detection", field_type="switch"),
            self._field("Enable translation", "settings-ingest-enable-translation", field_type="switch"),
            self._field("Enable versioning", "settings-ingest-enable-versioning", field_type="switch"),
            self._save_reset_row("ingest"),
            classes="settings-section",
        )

    def _query_section(self) -> Widget:
        return Container(
            Static("Query Settings", classes="settings-section-title"),
            self._select_field("Mode", "settings-query-mode"),
            self._field("k", "settings-query-k"),
            self._field("k_rerank", "settings-query-k-rerank"),
            self._field("Min relevance", "settings-query-min-relevance"),
            self._select_field("Rerank strategy", "settings-query-rerank-strategy", [("none", "none"), ("lexical", "lexical"), ("llm", "llm"), ("hybrid", "hybrid")]),
            self._select_field("Chat mode", "settings-query-chat-mode", [("single", "single"), ("chat", "chat")]),
            self._field("Query cache TTL", "settings-query-cache-ttl"),
            self._field("Query cache size", "settings-query-cache-size"),
            self._field("Enable query cache", "settings-query-enable-cache", field_type="switch"),
            self._field("Enable LLM server cache", "settings-query-enable-llm-server-cache", field_type="switch"),
            self._save_reset_row("query"),
            classes="settings-section",
        )

    def _llm_section(self) -> Widget:
        return Container(
            Static("LLM Providers and Services", classes="settings-section-title"),
            self._field("Providers (JSON)", "settings-llm-providers-json"),
            self._field("Services (JSON)", "settings-llm-services-json"),
            self._save_reset_row("llm"),
            classes="settings-section",
        )

    def _vdb_section(self) -> Widget:
        return Container(
            Static("Vector DB", classes="settings-section-title"),
            self._select_field("Provider", "settings-vdb-provider", [("chromadb", "chromadb"), ("postgresql", "postgresql")]),
            self._field("Collection", "settings-vdb-collection"),
            self._field("Persist dir", "settings-vdb-persist-dir"),
            self._field("HNSW space", "settings-vdb-hnsw-space"),
            self._field("Connection string", "settings-vdb-connection-string"),
            self._field("Host", "settings-vdb-host"),
            self._field("Port", "settings-vdb-port"),
            self._field("Database", "settings-vdb-database"),
            self._field("User", "settings-vdb-user"),
            self._field("Password", "settings-vdb-password", placeholder="(hidden)"),
            self._field("Index type", "settings-vdb-index-type"),
            self._field("Pool size", "settings-vdb-pool-size"),
            Horizontal(Button("Run Migration", id="settings-vdb-migrate"), classes="settings-actions"),
            self._save_reset_row("vdb"),
            classes="settings-section",
        )

    def _modes_section(self) -> Widget:
        return Container(
            Static("Mode Browser", classes="settings-section-title"),
            self._select_field("Active mode", "settings-modes-mode"),
            self._field("Synthesis style", "settings-modes-synthesis-style", disabled=True),
            self._field("Local policy", "settings-modes-local-policy", disabled=True),
            self._field("Remote policy", "settings-modes-remote-policy", disabled=True),
            self._field("Model knowledge", "settings-modes-model-policy", disabled=True),
            self._save_reset_row("modes"),
            classes="settings-section",
        )

    def _remote_section(self) -> Widget:
        return Container(
            Static("Remote Providers and Chains", classes="settings-section-title"),
            self._field("Remote providers (JSON)", "settings-remote-providers-json"),
            self._field("Remote chains (JSON)", "settings-remote-chains-json"),
            self._save_reset_row("remote"),
            classes="settings-section",
        )

    def _chats_notes_section(self) -> Widget:
        return Container(
            Static("Chats and Notes", classes="settings-section-title"),
            self._field("Chats enabled", "settings-chats-enabled", field_type="switch"),
            self._field("Chats dir", "settings-chats-dir"),
            self._field("Chats auto-save", "settings-chats-auto-save", field_type="switch"),
            self._field("Chats format", "settings-chats-format"),
            self._field("Notes enabled", "settings-notes-enabled", field_type="switch"),
            self._field("Notes dir", "settings-notes-dir"),
            self._field("Notes template", "settings-notes-template"),
            self._field("Notes filename format", "settings-notes-filename-format"),
            self._field("Notes integration", "settings-notes-integration"),
            self._field("Notes wikilinks", "settings-notes-wikilinks", field_type="switch"),
            self._field("Notes backlinks", "settings-notes-backlinks", field_type="switch"),
            self._field("Notes include tags", "settings-notes-include-tags", field_type="switch"),
            self._save_reset_row("chats-notes"),
            classes="settings-section",
        )

    def _save_reset_row(self, section: str) -> Widget:
        return Horizontal(
            Button("Save", id=f"settings-save-{section}"),
            Button("Reset", id=f"settings-reset-{section}"),
            classes="settings-actions",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("settings-save-"):
            self._save_config()
            self._set_status("Configuration saved.", False)
            return
        if button_id.startswith("settings-reset-"):
            self._reset_config()
            return
        if button_id == "settings-vdb-migrate":
            self._set_status("Use /migrate in query/ingest context to run migrations.", False)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._apply_live_update(event.input.id or "", event.value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._apply_live_switch_update(event.switch.id or "", bool(event.value))

    def on_select_changed(self, event: Select.Changed) -> None:
        field_id = event.select.id or ""
        value = str(event.value) if event.value is not None else ""
        if field_id == "settings-modes-mode":
            self._update_mode_display(value)
            return
        self._apply_live_update(field_id, value)

    def _refresh_settings(self) -> None:
        config = getattr(self.app, "config", None)
        if config is None:
            return
        self._set_input("settings-config-path", str(config.config_path or ""))
        self._set_input("settings-global-folder", self._format_optional(config.global_folder))
        self._set_input("settings-global-embed-model", config.embed_model)
        self._set_input("settings-global-device", config.device)
        self._set_input("settings-global-batch-size", str(config.batch_size))
        self._set_input("settings-global-embedding-dimension", self._format_optional(config.embedding_dimension))

        roots = [{"path": str(r.path), "tags": r.tags, "content_type": r.content_type} for r in config.ingest.roots]
        self._set_input("settings-ingest-roots-json", json.dumps(roots))
        self._set_input("settings-ingest-persist-dir", str(config.ingest.persist_dir))
        self._set_input("settings-ingest-collection", config.ingest.collection)
        self._set_select_value("settings-ingest-chunking-strategy", config.ingest.chunking_strategy)
        self._set_input("settings-ingest-chunk-size", str(config.ingest.chunk_size))
        self._set_input("settings-ingest-chunk-overlap", str(config.ingest.chunk_overlap))
        self._set_input("settings-ingest-tags-per-chunk", str(config.ingest.tags_per_chunk))
        self._set_input("settings-ingest-translation-target", config.ingest.translation_target)
        self._set_input("settings-ingest-max-files", self._format_optional(config.ingest.max_files))
        self._set_input("settings-ingest-max-seconds", self._format_optional(config.ingest.max_seconds))
        self._set_switch("settings-ingest-enable-ocr", config.ingest.enable_ocr)
        self._set_switch("settings-ingest-enable-ai-tagging", config.ingest.enable_ai_tagging)
        self._set_switch("settings-ingest-enable-language-detection", config.ingest.enable_language_detection)
        self._set_switch("settings-ingest-enable-translation", config.ingest.enable_translation)
        self._set_switch("settings-ingest-enable-versioning", config.ingest.enable_versioning)

        self._set_select_options("settings-query-mode", list(config.modes.keys()) if config.modes else [])
        self._set_select_value("settings-query-mode", config.query.mode)
        self._set_input("settings-query-k", str(config.query.k))
        self._set_input("settings-query-k-rerank", str(config.query.k_rerank))
        self._set_input("settings-query-min-relevance", str(config.query.min_relevance))
        self._set_select_value("settings-query-rerank-strategy", config.query.rerank_strategy)
        self._set_select_value("settings-query-chat-mode", config.query.chat_mode)
        self._set_input("settings-query-cache-ttl", str(config.query.query_cache_ttl))
        self._set_input("settings-query-cache-size", str(config.query.query_cache_size))
        self._set_switch("settings-query-enable-cache", config.query.enable_query_cache)
        self._set_switch("settings-query-enable-llm-server-cache", config.query.enable_llm_server_cache)

        providers_json = [{"provider_name": p.provider_name, "api_key": p.api_key, "base_url": p.base_url, "endpoint": p.endpoint, "api_version": p.api_version, "deployment_name": p.deployment_name} for p in config.llm.providers]
        services_json = {k: {"provider": v.provider, "model": v.model, "temperature": v.temperature, "max_tokens": v.max_tokens} for k, v in config.llm.services.items()}
        self._set_input("settings-llm-providers-json", json.dumps(providers_json))
        self._set_input("settings-llm-services-json", json.dumps(services_json))

        self._set_select_value("settings-vdb-provider", config.vectordb.provider)
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

        self._set_select_options("settings-modes-mode", list(config.modes.keys()) if config.modes else [])
        self._set_select_value("settings-modes-mode", config.query.mode)
        self._update_mode_display(config.query.mode)

        remote_json = [{"name": p.name, "type": p.type, "weight": p.weight, "max_results": p.max_results, "cache_results": p.cache_results, "cache_ttl": p.cache_ttl} for p in (config.remote_providers or [])]
        chains_json = [{"name": c.name, "agent_description": c.agent_description, "links": [{"source": l.source, "map": l.map} for l in c.links]} for c in (config.remote_provider_chains or [])]
        self._set_input("settings-remote-providers-json", json.dumps(remote_json))
        self._set_input("settings-remote-chains-json", json.dumps(chains_json))

        self._set_switch("settings-chats-enabled", config.chats.enabled)
        self._set_input("settings-chats-dir", config.chats.dir)
        self._set_switch("settings-chats-auto-save", config.chats.auto_save)
        self._set_input("settings-chats-format", config.chats.format)
        self._set_switch("settings-notes-enabled", config.notes.enabled)
        self._set_input("settings-notes-dir", config.notes.dir)
        self._set_input("settings-notes-template", config.notes.template)
        self._set_input("settings-notes-filename-format", config.notes.filename_format)
        self._set_input("settings-notes-integration", config.notes.integration)
        self._set_switch("settings-notes-wikilinks", config.notes.wikilinks)
        self._set_switch("settings-notes-backlinks", config.notes.backlinks)
        self._set_switch("settings-notes-include-tags", config.notes.include_tags)

    def _apply_live_update(self, field_id: str, value: str) -> None:
        config = getattr(self.app, "config", None)
        if config is None:
            return
        text = (value or "").strip()
        try:
            self._apply_live_update_inner(config, field_id, text)
            self._set_status(f"Updated {field_id}", False)
        except Exception as exc:
            self._set_status(f"Invalid value for {field_id}: {exc}", True)

    def _apply_live_update_inner(self, config: Any, field_id: str, text: str) -> None:
        from lsm.config.loader import build_remote_provider_chain_config, build_remote_provider_config
        from lsm.config.models import LLMProviderConfig, LLMServiceConfig, RootConfig

        if field_id == "settings-global-folder":
            config.global_settings.global_folder = Path(text) if text else None
        elif field_id == "settings-global-embed-model":
            config.global_settings.embed_model = text
        elif field_id == "settings-global-device":
            config.global_settings.device = text
        elif field_id == "settings-global-batch-size" and text:
            config.global_settings.batch_size = int(text)
        elif field_id == "settings-global-embedding-dimension":
            config.global_settings.embedding_dimension = int(text) if text else None
        elif field_id == "settings-ingest-persist-dir":
            config.ingest.persist_dir = Path(text)
        elif field_id == "settings-ingest-collection":
            config.ingest.collection = text
        elif field_id == "settings-ingest-chunking-strategy":
            config.ingest.chunking_strategy = text
        elif field_id == "settings-ingest-chunk-size" and text:
            config.ingest.chunk_size = int(text)
        elif field_id == "settings-ingest-chunk-overlap" and text:
            config.ingest.chunk_overlap = int(text)
        elif field_id == "settings-ingest-tags-per-chunk" and text:
            config.ingest.tags_per_chunk = int(text)
        elif field_id == "settings-ingest-translation-target":
            config.ingest.translation_target = text or "en"
        elif field_id == "settings-ingest-max-files":
            config.ingest.max_files = int(text) if text else None
        elif field_id == "settings-ingest-max-seconds":
            config.ingest.max_seconds = int(text) if text else None
        elif field_id == "settings-query-mode":
            config.query.mode = text
            self._set_select_value("settings-modes-mode", text)
            self._update_mode_display(text)
        elif field_id == "settings-query-k" and text:
            config.query.k = int(text)
        elif field_id == "settings-query-k-rerank" and text:
            config.query.k_rerank = int(text)
        elif field_id == "settings-query-min-relevance" and text:
            config.query.min_relevance = float(text)
        elif field_id == "settings-query-rerank-strategy":
            config.query.rerank_strategy = text
        elif field_id == "settings-query-chat-mode":
            config.query.chat_mode = text
        elif field_id == "settings-query-cache-ttl" and text:
            config.query.query_cache_ttl = int(text)
        elif field_id == "settings-query-cache-size" and text:
            config.query.query_cache_size = int(text)
        elif field_id == "settings-vdb-provider":
            config.vectordb.provider = text
        elif field_id == "settings-vdb-collection":
            config.vectordb.collection = text
        elif field_id == "settings-vdb-persist-dir":
            config.vectordb.persist_dir = Path(text)
        elif field_id == "settings-vdb-hnsw-space":
            config.vectordb.chroma_hnsw_space = text
        elif field_id == "settings-vdb-connection-string":
            config.vectordb.connection_string = text or None
        elif field_id == "settings-vdb-host":
            config.vectordb.host = text or None
        elif field_id == "settings-vdb-port":
            config.vectordb.port = text or None
        elif field_id == "settings-vdb-database":
            config.vectordb.database = text or None
        elif field_id == "settings-vdb-user":
            config.vectordb.user = text or None
        elif field_id == "settings-vdb-password":
            config.vectordb.password = text or None
        elif field_id == "settings-vdb-index-type":
            config.vectordb.index_type = text or "hnsw"
        elif field_id == "settings-vdb-pool-size" and text:
            config.vectordb.pool_size = int(text)
        elif field_id == "settings-notes-dir":
            config.notes.dir = text or "notes"
        elif field_id == "settings-notes-template":
            config.notes.template = text or "default"
        elif field_id == "settings-notes-filename-format":
            config.notes.filename_format = text or "timestamp"
        elif field_id == "settings-notes-integration":
            config.notes.integration = text or "none"
        elif field_id == "settings-chats-dir":
            config.chats.dir = text or "Chats"
        elif field_id == "settings-chats-format":
            config.chats.format = text or "markdown"
        elif field_id == "settings-ingest-roots-json":
            data = json.loads(text) if text else []
            config.ingest.roots = [RootConfig(path=Path(item.get("path", "")), tags=item.get("tags"), content_type=item.get("content_type")) for item in data]
        elif field_id == "settings-llm-providers-json":
            data = json.loads(text) if text else []
            config.llm.providers = [LLMProviderConfig(**item) for item in data]
        elif field_id == "settings-llm-services-json":
            data = json.loads(text) if text else {}
            config.llm.services = {k: LLMServiceConfig(**v) for k, v in data.items()}
        elif field_id == "settings-remote-providers-json":
            data = json.loads(text) if text else []
            config.remote_providers = [build_remote_provider_config(item) for item in data]
        elif field_id == "settings-remote-chains-json":
            data = json.loads(text) if text else []
            config.remote_provider_chains = [build_remote_provider_chain_config(item) for item in data]

    def _apply_live_switch_update(self, field_id: str, value: bool) -> None:
        config = getattr(self.app, "config", None)
        if config is None:
            return
        if field_id == "settings-ingest-enable-ocr":
            config.ingest.enable_ocr = value
        elif field_id == "settings-ingest-enable-ai-tagging":
            config.ingest.enable_ai_tagging = value
        elif field_id == "settings-ingest-enable-language-detection":
            config.ingest.enable_language_detection = value
        elif field_id == "settings-ingest-enable-translation":
            config.ingest.enable_translation = value
        elif field_id == "settings-ingest-enable-versioning":
            config.ingest.enable_versioning = value
        elif field_id == "settings-query-enable-cache":
            config.query.enable_query_cache = value
        elif field_id == "settings-query-enable-llm-server-cache":
            config.query.enable_llm_server_cache = value
        elif field_id == "settings-notes-enabled":
            config.notes.enabled = value
        elif field_id == "settings-notes-wikilinks":
            config.notes.wikilinks = value
        elif field_id == "settings-notes-backlinks":
            config.notes.backlinks = value
        elif field_id == "settings-notes-include-tags":
            config.notes.include_tags = value
        elif field_id == "settings-chats-enabled":
            config.chats.enabled = value
        elif field_id == "settings-chats-auto-save":
            config.chats.auto_save = value
        else:
            return
        self._set_status(f"Updated {field_id}", False)

    def _save_config(self) -> None:
        cfg = self.app.config
        if cfg and cfg.config_path:
            save_config_to_file(cfg, cfg.config_path)

    def _reset_config(self) -> None:
        cfg = self.app.config
        if not cfg or not cfg.config_path:
            self._set_status("No config path available for reset.", True)
            return
        self.app.config = load_config_from_file(cfg.config_path)
        self._refresh_settings()
        self._set_status("Configuration reloaded from disk.", False)

    def _update_mode_display(self, mode_name: Optional[str]) -> None:
        if not mode_name:
            return
        try:
            mode = self.app.config.get_mode_config(mode_name)
        except Exception:
            return
        local = mode.source_policy.local
        remote = mode.source_policy.remote
        model = mode.source_policy.model_knowledge
        remote_providers = []
        for item in remote.remote_providers or []:
            if isinstance(item, RemoteProviderRef):
                remote_providers.append(f"{item.source} ({item.weight})" if item.weight is not None else item.source)
            else:
                remote_providers.append(str(item))
        self._set_input("settings-modes-synthesis-style", mode.synthesis_style)
        self._set_input("settings-modes-local-policy", f"enabled={local.enabled}, min={local.min_relevance}, k={local.k}, k_rerank={local.k_rerank}")
        self._set_input("settings-modes-remote-policy", f"enabled={remote.enabled}, rank={remote.rank_strategy}, max={remote.max_results}, providers={remote_providers}")
        self._set_input("settings-modes-model-policy", f"enabled={model.enabled}, require_label={model.require_label}")

    def _activate_tab(self, tab_id: str) -> None:
        self.query_one("#settings-tabs", TabbedContent).active = tab_id

    def action_settings_tab_1(self) -> None: self._activate_tab("settings-global")
    def action_settings_tab_2(self) -> None: self._activate_tab("settings-ingest")
    def action_settings_tab_3(self) -> None: self._activate_tab("settings-query")
    def action_settings_tab_4(self) -> None: self._activate_tab("settings-llm")
    def action_settings_tab_5(self) -> None: self._activate_tab("settings-vdb")
    def action_settings_tab_6(self) -> None: self._activate_tab("settings-modes")
    def action_settings_tab_7(self) -> None: self._activate_tab("settings-remote")
    def action_settings_tab_8(self) -> None: self._activate_tab("settings-chats-notes")

    def _set_status(self, message: str, error: bool) -> None:
        try:
            widget = self.query_one("#settings-status", Static)
            widget.update(f"[red]{message}[/red]" if error else message)
        except Exception:
            logger.warning(message)

    def _field(self, label: str, field_id: str, *, placeholder: str = "", disabled: bool = False, field_type: str = "input") -> Widget:
        label_widget = Static(label, classes="settings-label")
        field_widget: Widget = Switch(id=field_id) if field_type == "switch" else Input(placeholder=placeholder, id=field_id)
        field_widget.disabled = disabled
        return Horizontal(label_widget, field_widget, classes="settings-field")

    def _select_field(self, label: str, field_id: str, options: Optional[list[tuple[str, str]]] = None) -> Widget:
        return Horizontal(Static(label, classes="settings-label"), Select(options or [], id=field_id), classes="settings-field")

    def _set_input(self, field_id: str, value: str) -> None:
        try:
            self.query_one(f"#{field_id}", Input).value = value or ""
        except Exception:
            return

    def _set_switch(self, field_id: str, value: bool) -> None:
        try:
            self.query_one(f"#{field_id}", Switch).value = bool(value)
        except Exception:
            return

    def _set_select_options(self, field_id: str, values: list[str]) -> None:
        try:
            self.query_one(f"#{field_id}", Select).set_options([(v, v) for v in values])
        except Exception:
            return

    def _set_select_value(self, field_id: str, value: str) -> None:
        try:
            self.query_one(f"#{field_id}", Select).value = value
        except Exception:
            return

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)
