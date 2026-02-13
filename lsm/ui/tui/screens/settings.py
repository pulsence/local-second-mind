"""
Settings screen for LSM TUI.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch, TabbedContent, TabPane

from lsm.config.loader import load_config_from_file, save_config_to_file
from lsm.config.models import RemoteProviderRef
from lsm.logging import get_logger
from lsm.ui.tui.widgets.settings_base import (
    _field as _shared_field,
    _replace_container_children as _shared_replace_container_children,
    _save_reset_row as _shared_save_reset_row,
    _select_field as _shared_select_field,
    _set_input as _shared_set_input,
    _set_select_options as _shared_set_select_options,
    _set_select_value as _shared_set_select_value,
    _set_switch as _shared_set_switch,
)

logger = get_logger(__name__)


class SettingsScreen(Widget):
    """Settings editor aligned to current config sections."""

    current_mode: str = "grounded"
    """Current selected mode name (for tests and UI state)."""
    _LLM_COMMON_SERVICES = (
        "default",
        "query",
        "decomposition",
        "tagging",
        "ranking",
        "translation",
    )

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
        self._is_refreshing = False
        if getattr(self.app, "current_context", None) == "settings":
            self._refresh_settings()
            tabs = self.query_one("#settings-tabs", TabbedContent)
            self.call_after_refresh(tabs.focus)

    def on_show(self) -> None:
        self.refresh_from_config()

    def refresh_from_config(self) -> None:
        """Refresh all fields from the in-memory config object."""
        self._refresh_settings()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        return

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
            self._field("Persist dir", "settings-ingest-persist-dir"),
            self._field("Collection", "settings-ingest-collection"),
            self._field("Manifest", "settings-ingest-manifest"),
            self._field("Chroma flush interval", "settings-ingest-chroma-flush-interval"),
            self._select_field("Chunking strategy", "settings-ingest-chunking-strategy", [("structure", "structure"), ("fixed", "fixed")]),
            self._field("Chunk size", "settings-ingest-chunk-size"),
            self._field("Chunk overlap", "settings-ingest-chunk-overlap"),
            self._field("Tags per chunk", "settings-ingest-tags-per-chunk"),
            self._field("Translation target", "settings-ingest-translation-target"),
            self._field("Extensions (comma-separated)", "settings-ingest-extensions"),
            self._field("Exclude dirs (comma-separated)", "settings-ingest-exclude-dirs"),
            self._field("Max files", "settings-ingest-max-files"),
            self._field("Max seconds", "settings-ingest-max-seconds"),
            self._field("Override extensions", "settings-ingest-override-extensions", field_type="switch"),
            self._field("Override excludes", "settings-ingest-override-excludes", field_type="switch"),
            self._field("Dry run", "settings-ingest-dry-run", field_type="switch"),
            self._field("Skip errors", "settings-ingest-skip-errors", field_type="switch"),
            self._field("Enable OCR", "settings-ingest-enable-ocr", field_type="switch"),
            self._field("Enable AI tagging", "settings-ingest-enable-ai-tagging", field_type="switch"),
            self._field("Enable language detection", "settings-ingest-enable-language-detection", field_type="switch"),
            self._field("Enable translation", "settings-ingest-enable-translation", field_type="switch"),
            self._field("Enable versioning", "settings-ingest-enable-versioning", field_type="switch"),
            Static("Roots", classes="settings-subsection-title"),
            Vertical(id="settings-ingest-roots-list"),
            Horizontal(Button("Add Root", id="settings-ingest-root-add"), classes="settings-actions"),
            self._save_reset_row("ingest"),
            classes="settings-section",
        )

    def _query_section(self) -> Widget:
        return Container(
            Static("Query Settings", classes="settings-section-title"),
            self._select_field("Mode", "settings-query-mode"),
            self._field("k", "settings-query-k"),
            self._field("Retrieve k", "settings-query-retrieve-k"),
            self._field("k_rerank", "settings-query-k-rerank"),
            self._field("Min relevance", "settings-query-min-relevance"),
            self._field("Local pool", "settings-query-local-pool"),
            self._field("Max per file", "settings-query-max-per-file"),
            self._field("Path contains (comma-separated)", "settings-query-path-contains"),
            self._field("Ext allow (comma-separated)", "settings-query-ext-allow"),
            self._field("Ext deny (comma-separated)", "settings-query-ext-deny"),
            self._select_field("Rerank strategy", "settings-query-rerank-strategy", [("none", "none"), ("lexical", "lexical"), ("llm", "llm"), ("hybrid", "hybrid")]),
            self._select_field("Chat mode", "settings-query-chat-mode", [("single", "single"), ("chat", "chat")]),
            self._field("Query cache TTL", "settings-query-cache-ttl"),
            self._field("Query cache size", "settings-query-cache-size"),
            self._field("No rerank", "settings-query-no-rerank", field_type="switch"),
            self._field("Enable query cache", "settings-query-enable-cache", field_type="switch"),
            self._field("Enable LLM server cache", "settings-query-enable-llm-server-cache", field_type="switch"),
            self._save_reset_row("query"),
            classes="settings-section",
        )

    def _llm_section(self) -> Widget:
        return Container(
            Static("LLM Providers and Services", classes="settings-section-title"),
            Static("Services", classes="settings-subsection-title"),
            Vertical(id="settings-llm-services-list"),
            Horizontal(Button("Add Service", id="settings-llm-service-add"), classes="settings-actions"),
            Static("Providers", classes="settings-subsection-title"),
            Vertical(id="settings-llm-providers-list"),
            Horizontal(Button("Add Provider", id="settings-llm-provider-add"), classes="settings-actions"),
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
            Static("Remote Chains", classes="settings-subsection-title"),
            Vertical(id="settings-remote-chains-list"),
            Horizontal(Button("Add Chain", id="settings-remote-chain-add"), classes="settings-actions"),
            Static("Remote Providers", classes="settings-subsection-title"),
            Vertical(id="settings-remote-providers-list"),
            Horizontal(Button("Add Provider", id="settings-remote-provider-add"), classes="settings-actions"),
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
        return _shared_save_reset_row(section)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if self._handle_structured_button(button_id):
            return
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
        if self._is_refreshing:
            return
        if self._apply_dynamic_text_update(event.input.id or "", event.value):
            return
        self._apply_live_update(event.input.id or "", event.value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if self._is_refreshing:
            return
        if self._apply_dynamic_switch_update(event.switch.id or "", bool(event.value)):
            return
        self._apply_live_switch_update(event.switch.id or "", bool(event.value))

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._is_refreshing:
            return
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
        if getattr(self, "_is_refreshing", False):
            return
        self._is_refreshing = True
        try:
            ingest = getattr(config, "ingest", None)
            query = getattr(config, "query", None)
            vdb = getattr(config, "vectordb", None)
            chats = getattr(config, "chats", None)
            notes = getattr(config, "notes", None)
            modes = getattr(config, "modes", None) or {}

            self._set_input("settings-config-path", str(getattr(config, "config_path", "") or ""))
            self._set_input("settings-global-folder", self._format_optional(getattr(config, "global_folder", None)))
            self._set_input("settings-global-embed-model", str(getattr(config, "embed_model", "")))
            self._set_input("settings-global-device", str(getattr(config, "device", "")))
            self._set_input("settings-global-batch-size", str(getattr(config, "batch_size", "")))
            self._set_input("settings-global-embedding-dimension", self._format_optional(getattr(config, "embedding_dimension", None)))

            if ingest is not None:
                self._refresh_ingest_roots_fields(config)
                self._set_input("settings-ingest-persist-dir", str(getattr(ingest, "persist_dir", "")))
                self._set_input("settings-ingest-collection", str(getattr(ingest, "collection", "")))
                self._set_input("settings-ingest-manifest", str(getattr(ingest, "manifest", "")))
                self._set_input("settings-ingest-chroma-flush-interval", str(getattr(ingest, "chroma_flush_interval", "")))
                self._set_select_value("settings-ingest-chunking-strategy", str(getattr(ingest, "chunking_strategy", "")))
                self._set_input("settings-ingest-chunk-size", str(getattr(ingest, "chunk_size", "")))
                self._set_input("settings-ingest-chunk-overlap", str(getattr(ingest, "chunk_overlap", "")))
                self._set_input("settings-ingest-tags-per-chunk", str(getattr(ingest, "tags_per_chunk", "")))
                self._set_input("settings-ingest-translation-target", str(getattr(ingest, "translation_target", "")))
                self._set_input("settings-ingest-extensions", ", ".join(getattr(ingest, "extensions", None) or []))
                self._set_input("settings-ingest-exclude-dirs", ", ".join(getattr(ingest, "exclude_dirs", None) or []))
                self._set_input("settings-ingest-max-files", self._format_optional(getattr(ingest, "max_files", None)))
                self._set_input("settings-ingest-max-seconds", self._format_optional(getattr(ingest, "max_seconds", None)))
                self._set_switch("settings-ingest-override-extensions", bool(getattr(ingest, "override_extensions", False)))
                self._set_switch("settings-ingest-override-excludes", bool(getattr(ingest, "override_excludes", False)))
                self._set_switch("settings-ingest-dry-run", bool(getattr(ingest, "dry_run", False)))
                self._set_switch("settings-ingest-skip-errors", bool(getattr(ingest, "skip_errors", True)))
                self._set_switch("settings-ingest-enable-ocr", bool(getattr(ingest, "enable_ocr", False)))
                self._set_switch("settings-ingest-enable-ai-tagging", bool(getattr(ingest, "enable_ai_tagging", False)))
                self._set_switch("settings-ingest-enable-language-detection", bool(getattr(ingest, "enable_language_detection", False)))
                self._set_switch("settings-ingest-enable-translation", bool(getattr(ingest, "enable_translation", False)))
                self._set_switch("settings-ingest-enable-versioning", bool(getattr(ingest, "enable_versioning", False)))

            if query is not None:
                self._set_select_options("settings-query-mode", list(modes.keys()))
                self._set_select_value("settings-query-mode", str(getattr(query, "mode", "")))
                self._set_input("settings-query-k", str(getattr(query, "k", "")))
                self._set_input("settings-query-retrieve-k", self._format_optional(getattr(query, "retrieve_k", None)))
                self._set_input("settings-query-k-rerank", str(getattr(query, "k_rerank", "")))
                self._set_input("settings-query-min-relevance", str(getattr(query, "min_relevance", "")))
                self._set_input("settings-query-local-pool", self._format_optional(getattr(query, "local_pool", None)))
                self._set_input("settings-query-max-per-file", str(getattr(query, "max_per_file", "")))
                self._set_input("settings-query-path-contains", ", ".join(getattr(query, "path_contains", None) or []))
                self._set_input("settings-query-ext-allow", ", ".join(getattr(query, "ext_allow", None) or []))
                self._set_input("settings-query-ext-deny", ", ".join(getattr(query, "ext_deny", None) or []))
                self._set_select_value("settings-query-rerank-strategy", str(getattr(query, "rerank_strategy", "")))
                self._set_select_value("settings-query-chat-mode", str(getattr(query, "chat_mode", "")))
                self._set_input("settings-query-cache-ttl", str(getattr(query, "query_cache_ttl", "")))
                self._set_input("settings-query-cache-size", str(getattr(query, "query_cache_size", "")))
                self._set_switch("settings-query-no-rerank", bool(getattr(query, "no_rerank", False)))
                self._set_switch("settings-query-enable-cache", bool(getattr(query, "enable_query_cache", False)))
                self._set_switch("settings-query-enable-llm-server-cache", bool(getattr(query, "enable_llm_server_cache", False)))

            if vdb is not None:
                self._set_select_value("settings-vdb-provider", str(getattr(vdb, "provider", "")))
                self._set_input("settings-vdb-collection", str(getattr(vdb, "collection", "")))
                self._set_input("settings-vdb-persist-dir", str(getattr(vdb, "persist_dir", "")))
                self._set_input("settings-vdb-hnsw-space", str(getattr(vdb, "chroma_hnsw_space", "")))
                self._set_input("settings-vdb-connection-string", self._format_optional(getattr(vdb, "connection_string", None)))
                self._set_input("settings-vdb-host", self._format_optional(getattr(vdb, "host", None)))
                self._set_input("settings-vdb-port", self._format_optional(getattr(vdb, "port", None)))
                self._set_input("settings-vdb-database", self._format_optional(getattr(vdb, "database", None)))
                self._set_input("settings-vdb-user", self._format_optional(getattr(vdb, "user", None)))
                self._set_input("settings-vdb-password", "")
                self._set_input("settings-vdb-index-type", str(getattr(vdb, "index_type", "")))
                self._set_input("settings-vdb-pool-size", str(getattr(vdb, "pool_size", "")))

            self._set_select_options("settings-modes-mode", list(modes.keys()))
            if query is not None:
                mode_name = str(getattr(query, "mode", ""))
                self._set_select_value("settings-modes-mode", mode_name)
                self._update_mode_display(mode_name)

            if chats is not None:
                self._set_switch("settings-chats-enabled", bool(getattr(chats, "enabled", False)))
                self._set_input("settings-chats-dir", str(getattr(chats, "dir", "")))
                self._set_switch("settings-chats-auto-save", bool(getattr(chats, "auto_save", False)))
                self._set_input("settings-chats-format", str(getattr(chats, "format", "")))

            if notes is not None:
                self._set_switch("settings-notes-enabled", bool(getattr(notes, "enabled", False)))
                self._set_input("settings-notes-dir", str(getattr(notes, "dir", "")))
                self._set_input("settings-notes-template", str(getattr(notes, "template", "")))
                self._set_input("settings-notes-filename-format", str(getattr(notes, "filename_format", "")))
                self._set_input("settings-notes-integration", str(getattr(notes, "integration", "")))
                self._set_switch("settings-notes-wikilinks", bool(getattr(notes, "wikilinks", False)))
                self._set_switch("settings-notes-backlinks", bool(getattr(notes, "backlinks", False)))
                self._set_switch("settings-notes-include-tags", bool(getattr(notes, "include_tags", False)))

            self._refresh_llm_structured_fields(config)
            self._refresh_remote_structured_fields(config)
        except Exception as exc:
            self._set_status(f"Failed to refresh settings: {exc}", True)
        finally:
            def _clear_refresh_flag() -> None:
                self._is_refreshing = False

            try:
                self.call_after_refresh(_clear_refresh_flag)
            except Exception:
                _clear_refresh_flag()

    def _apply_live_update(self, field_id: str, value: str) -> None:
        config = getattr(self.app, "config", None)
        if config is None:
            return
        text = (value or "").strip()
        try:
            self._apply_live_update_inner(config, field_id, text)
        except Exception as exc:
            self._set_status(f"Invalid value for {field_id}: {exc}", True)

    def _apply_live_update_inner(self, config: Any, field_id: str, text: str) -> None:
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
        elif field_id == "settings-ingest-manifest":
            config.ingest.manifest = Path(text)
        elif field_id == "settings-ingest-chroma-flush-interval" and text:
            config.ingest.chroma_flush_interval = int(text)
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
        elif field_id == "settings-ingest-extensions":
            config.ingest.extensions = [item.strip() for item in text.split(",") if item.strip()] or None
        elif field_id == "settings-ingest-exclude-dirs":
            config.ingest.exclude_dirs = [item.strip() for item in text.split(",") if item.strip()] or None
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
        elif field_id == "settings-query-retrieve-k":
            config.query.retrieve_k = int(text) if text else None
        elif field_id == "settings-query-k-rerank" and text:
            config.query.k_rerank = int(text)
        elif field_id == "settings-query-min-relevance" and text:
            config.query.min_relevance = float(text)
        elif field_id == "settings-query-local-pool":
            config.query.local_pool = int(text) if text else None
        elif field_id == "settings-query-max-per-file" and text:
            config.query.max_per_file = int(text)
        elif field_id == "settings-query-path-contains":
            config.query.path_contains = [item.strip() for item in text.split(",") if item.strip()] or None
        elif field_id == "settings-query-ext-allow":
            config.query.ext_allow = [item.strip() for item in text.split(",") if item.strip()] or None
        elif field_id == "settings-query-ext-deny":
            config.query.ext_deny = [item.strip() for item in text.split(",") if item.strip()] or None
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
            config.vectordb.port = int(text) if text else None
        elif field_id == "settings-vdb-database":
            config.vectordb.database = text or None
        elif field_id == "settings-vdb-user":
            config.vectordb.user = text or None
        elif field_id == "settings-vdb-password":
            if text:
                config.vectordb.password = text
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
        elif field_id == "settings-ingest-override-extensions":
            config.ingest.override_extensions = value
        elif field_id == "settings-ingest-override-excludes":
            config.ingest.override_excludes = value
        elif field_id == "settings-ingest-dry-run":
            config.ingest.dry_run = value
        elif field_id == "settings-ingest-skip-errors":
            config.ingest.skip_errors = value
        elif field_id == "settings-query-enable-cache":
            config.query.enable_query_cache = value
        elif field_id == "settings-query-enable-llm-server-cache":
            config.query.enable_llm_server_cache = value
        elif field_id == "settings-query-no-rerank":
            config.query.no_rerank = value
            if value:
                config.query.rerank_strategy = "none"
                self._set_select_value("settings-query-rerank-strategy", "none")
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

    def _apply_dynamic_text_update(self, field_id: str, value: str) -> bool:
        config = getattr(self.app, "config", None)
        if config is None:
            return False
        text = (value or "").strip()
        try:
            handled = self._apply_dynamic_text_update_inner(config, field_id, text)
            return handled
        except Exception as exc:
            self._set_status(f"Invalid value for {field_id}: {exc}", True)
            return True

    def _apply_dynamic_text_update_inner(self, config: Any, field_id: str, text: str) -> bool:
        root_match = re.fullmatch(r"settings-ingest-root-(\d+)-([a-z-]+)", field_id)
        if root_match:
            idx = int(root_match.group(1))
            field = root_match.group(2)
            roots = config.ingest.roots or []
            if idx >= len(roots):
                return True
            root = roots[idx]
            if field == "path":
                if not text:
                    raise ValueError("root path cannot be empty")
                root.path = Path(text)
            elif field == "tags":
                tags = [item.strip() for item in text.split(",") if item.strip()]
                root.tags = tags or None
            elif field == "content-type":
                root.content_type = text or None
            return True

        provider_match = re.fullmatch(r"settings-llm-provider-(\d+)-([a-z-]+)", field_id)
        if provider_match:
            idx = int(provider_match.group(1))
            field = provider_match.group(2)
            providers = config.llm.providers or []
            if idx >= len(providers):
                return True
            provider = providers[idx]
            if field == "name":
                if not text:
                    raise ValueError("provider name cannot be empty")
                if provider.provider_name != text:
                    old_name = provider.provider_name
                    provider.provider_name = text
                    for service in config.llm.services.values():
                        if service.provider == old_name:
                            service.provider = text
            elif field == "api-key":
                provider.api_key = text or None
            elif field == "base-url":
                provider.base_url = text or None
            elif field == "endpoint":
                provider.endpoint = text or None
            elif field == "api-version":
                provider.api_version = text or None
            elif field == "deployment-name":
                provider.deployment_name = text or None
            return True

        service_match = re.fullmatch(r"settings-llm-service-(\d+)-([a-z-]+)", field_id)
        if service_match:
            idx = int(service_match.group(1))
            field = service_match.group(2)
            service_names = self._ordered_service_names(config)
            if idx >= len(service_names):
                return True
            service = config.llm.services[service_names[idx]]
            if field == "provider":
                service.provider = text
            elif field == "model":
                service.model = text
            elif field == "temperature":
                service.temperature = float(text) if text else None
            elif field == "max-tokens":
                service.max_tokens = int(text) if text else None
            return True

        remote_provider_match = re.fullmatch(r"settings-remote-provider-(\d+)-([a-z-]+)", field_id)
        if remote_provider_match:
            idx = int(remote_provider_match.group(1))
            field = remote_provider_match.group(2)
            providers = config.remote_providers or []
            if idx >= len(providers):
                return True
            provider = providers[idx]
            if field == "name":
                if not text:
                    raise ValueError("provider name cannot be empty")
                if provider.name != text:
                    old_name = provider.name
                    provider.name = text
                    for mode_config in (config.modes or {}).values():
                        refs = mode_config.source_policy.remote.remote_providers or []
                        for ref_idx, ref in enumerate(refs):
                            if isinstance(ref, RemoteProviderRef):
                                if ref.source == old_name:
                                    ref.source = text
                            elif str(ref) == old_name:
                                # Keep string refs as strings.
                                refs[ref_idx] = text
                    for chain in config.remote_provider_chains or []:
                        for link in chain.links:
                            if link.source == old_name:
                                link.source = text
            elif field == "type":
                if not text:
                    raise ValueError("provider type cannot be empty")
                provider.type = text
            elif field == "weight":
                provider.weight = float(text) if text else 1.0
            elif field == "api-key":
                provider.api_key = text or None
            elif field == "endpoint":
                provider.endpoint = text or None
            elif field == "max-results":
                provider.max_results = int(text) if text else None
            elif field == "language":
                provider.language = text or None
            elif field == "user-agent":
                provider.user_agent = text or None
            elif field == "timeout":
                provider.timeout = int(text) if text else None
            elif field == "min-interval":
                provider.min_interval_seconds = float(text) if text else None
            elif field == "section-limit":
                provider.section_limit = int(text) if text else None
            elif field == "snippet-max-chars":
                provider.snippet_max_chars = int(text) if text else None
            elif field == "include-disambiguation":
                provider.include_disambiguation = self._parse_optional_bool(text)
            elif field == "cache-ttl":
                provider.cache_ttl = int(text) if text else 86400
            return True

        chain_match = re.fullmatch(r"settings-remote-chain-(\d+)-([a-z-]+)", field_id)
        if chain_match:
            idx = int(chain_match.group(1))
            field = chain_match.group(2)
            chains = config.remote_provider_chains or []
            if idx >= len(chains):
                return True
            chain = chains[idx]
            if field == "name":
                if not text:
                    raise ValueError("chain name cannot be empty")
                chain.name = text
            elif field == "agent-description":
                chain.agent_description = text
            return True

        link_match = re.fullmatch(r"settings-remote-chain-(\d+)-link-(\d+)-([a-z-]+)", field_id)
        if link_match:
            chain_idx = int(link_match.group(1))
            link_idx = int(link_match.group(2))
            field = link_match.group(3)
            chains = config.remote_provider_chains or []
            if chain_idx >= len(chains):
                return True
            links = chains[chain_idx].links
            if link_idx >= len(links):
                return True
            link = links[link_idx]
            if field == "source":
                link.source = text
            elif field == "map":
                link.map = [item.strip() for item in text.split(",") if item.strip()] or None
            return True

        return False

    def _apply_dynamic_switch_update(self, field_id: str, value: bool) -> bool:
        config = getattr(self.app, "config", None)
        if config is None:
            return False
        try:
            match = re.fullmatch(r"settings-remote-provider-(\d+)-cache-results", field_id)
            if not match:
                return False
            idx = int(match.group(1))
            providers = config.remote_providers or []
            if idx >= len(providers):
                return True
            providers[idx].cache_results = value
            return True
        except Exception as exc:
            self._set_status(f"Invalid value for {field_id}: {exc}", True)
            return True

    def _handle_structured_button(self, button_id: str) -> bool:
        config = getattr(self.app, "config", None)
        if config is None:
            return False
        try:
            if button_id == "settings-ingest-root-add":
                self._add_ingest_root(config)
                return True
            if button_id == "settings-llm-provider-add":
                self._add_llm_provider(config)
                return True
            if button_id == "settings-llm-service-add":
                self._add_llm_service(config)
                return True
            if button_id == "settings-remote-provider-add":
                self._add_remote_provider(config)
                return True
            if button_id == "settings-remote-chain-add":
                self._add_remote_chain(config)
                return True

            llm_provider_remove = re.fullmatch(r"settings-llm-provider-remove-(\d+)", button_id)
            if llm_provider_remove:
                self._remove_llm_provider(config, int(llm_provider_remove.group(1)))
                return True

            llm_service_remove = re.fullmatch(r"settings-llm-service-remove-(\d+)", button_id)
            if llm_service_remove:
                self._remove_llm_service(config, int(llm_service_remove.group(1)))
                return True

            remote_provider_remove = re.fullmatch(r"settings-remote-provider-remove-(\d+)", button_id)
            if remote_provider_remove:
                self._remove_remote_provider(config, int(remote_provider_remove.group(1)))
                return True

            remote_chain_remove = re.fullmatch(r"settings-remote-chain-remove-(\d+)", button_id)
            if remote_chain_remove:
                self._remove_remote_chain(config, int(remote_chain_remove.group(1)))
                return True

            add_link = re.fullmatch(r"settings-remote-chain-(\d+)-link-add", button_id)
            if add_link:
                self._add_remote_chain_link(config, int(add_link.group(1)))
                return True

            remove_link = re.fullmatch(r"settings-remote-chain-(\d+)-link-remove-(\d+)", button_id)
            if remove_link:
                self._remove_remote_chain_link(config, int(remove_link.group(1)), int(remove_link.group(2)))
                return True

            root_remove = re.fullmatch(r"settings-ingest-root-remove-(\d+)", button_id)
            if root_remove:
                self._remove_ingest_root(config, int(root_remove.group(1)))
                return True
        except Exception as exc:
            self._set_status(str(exc), True)
            return True
        return False

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

    def _refresh_ingest_roots_fields(self, config: Any) -> None:
        if not self._container_exists("#settings-ingest-roots-list"):
            return
        roots = list(config.ingest.roots or [])
        root_widgets: list[Widget] = []
        if not roots:
            root_widgets.append(Static("No ingest roots configured.", classes="settings-label"))
        for idx, root in enumerate(roots):
            tags = ", ".join(getattr(root, "tags", None) or [])
            root_widgets.append(
                Container(
                    Static(f"Root {idx + 1}", classes="settings-subsection-title"),
                    self._field("Path", f"settings-ingest-root-{idx}-path", value=str(getattr(root, "path", ""))),
                    self._field("Tags (comma-separated)", f"settings-ingest-root-{idx}-tags", value=tags),
                    self._field("Content type", f"settings-ingest-root-{idx}-content-type", value=self._format_optional(getattr(root, "content_type", None))),
                    Horizontal(Button("Remove Root", id=f"settings-ingest-root-remove-{idx}"), classes="settings-actions"),
                    classes="settings-subsection",
                )
            )
        self._replace_container_children("#settings-ingest-roots-list", root_widgets)

    def _add_ingest_root(self, config: Any) -> None:
        from lsm.config.models import RootConfig

        config.ingest.roots.append(RootConfig(path=Path(".")))
        self._refresh_ingest_roots_fields(config)
        self._set_status("Added ingest root", False)

    def _remove_ingest_root(self, config: Any, idx: int) -> None:
        roots = config.ingest.roots or []
        if idx >= len(roots):
            return
        if len(roots) <= 1:
            raise ValueError("At least one ingest root is required.")
        roots.pop(idx)
        self._refresh_ingest_roots_fields(config)
        self._set_status("Removed ingest root", False)

    def _refresh_llm_structured_fields(self, config: Any) -> None:
        has_providers_container = self._container_exists("#settings-llm-providers-list")
        has_services_container = self._container_exists("#settings-llm-services-list")
        if not has_providers_container and not has_services_container:
            return

        llm_cfg = getattr(config, "llm", None)
        providers = list(getattr(llm_cfg, "providers", []) or [])
        services = getattr(llm_cfg, "services", {}) or {}
        if has_providers_container:
            provider_widgets: list[Widget] = []
            if not providers:
                provider_widgets.append(Static("No providers configured.", classes="settings-label"))
            for idx, provider in enumerate(providers):
                provider_name = str(getattr(provider, "provider_name", ""))
                api_key = self._format_optional(getattr(provider, "api_key", None))
                base_url = self._format_optional(getattr(provider, "base_url", None))
                endpoint = self._format_optional(getattr(provider, "endpoint", None))
                api_version = self._format_optional(getattr(provider, "api_version", None))
                deployment_name = self._format_optional(getattr(provider, "deployment_name", None))
                provider_widgets.append(
                    Container(
                        Static(f"Provider {idx + 1}", classes="settings-subsection-title"),
                        self._field("Name", f"settings-llm-provider-{idx}-name", value=provider_name),
                        self._field("API key", f"settings-llm-provider-{idx}-api-key", value=api_key),
                        self._field("Base URL", f"settings-llm-provider-{idx}-base-url", value=base_url),
                        self._field("Endpoint", f"settings-llm-provider-{idx}-endpoint", value=endpoint),
                        self._field("API version", f"settings-llm-provider-{idx}-api-version", value=api_version),
                        self._field("Deployment", f"settings-llm-provider-{idx}-deployment-name", value=deployment_name),
                        Horizontal(Button("Remove Provider", id=f"settings-llm-provider-remove-{idx}"), classes="settings-actions"),
                        classes="settings-subsection",
                    )
                )
            self._replace_container_children("#settings-llm-providers-list", provider_widgets)

        service_names = self._ordered_service_names(config)
        if has_services_container:
            service_widgets: list[Widget] = []
            if not service_names:
                service_widgets.append(Static("No services configured.", classes="settings-label"))
            for idx, service_name in enumerate(service_names):
                service = services.get(service_name)
                if service is None:
                    continue
                provider_name = str(getattr(service, "provider", ""))
                model = str(getattr(service, "model", ""))
                temperature = self._format_optional(getattr(service, "temperature", None))
                max_tokens = self._format_optional(getattr(service, "max_tokens", None))
                service_widgets.append(
                    Container(
                        Static(f"Service {idx + 1}", classes="settings-subsection-title"),
                        Static(f"Name: {service_name}", classes="settings-label"),
                        self._field("Provider", f"settings-llm-service-{idx}-provider", value=provider_name),
                        self._field("Model", f"settings-llm-service-{idx}-model", value=model),
                        self._field("Temperature", f"settings-llm-service-{idx}-temperature", value=temperature),
                        self._field("Max tokens", f"settings-llm-service-{idx}-max-tokens", value=max_tokens),
                        Horizontal(Button("Remove Service", id=f"settings-llm-service-remove-{idx}"), classes="settings-actions"),
                        classes="settings-subsection",
                    )
                )
            self._replace_container_children("#settings-llm-services-list", service_widgets)

    def _refresh_remote_structured_fields(self, config: Any) -> None:
        has_providers_container = self._container_exists("#settings-remote-providers-list")
        has_chains_container = self._container_exists("#settings-remote-chains-list")
        if not has_providers_container and not has_chains_container:
            return

        providers = list(getattr(config, "remote_providers", None) or [])
        if has_providers_container:
            provider_widgets: list[Widget] = []
            if not providers:
                provider_widgets.append(Static("No remote providers configured.", classes="settings-label"))
            for idx, provider in enumerate(providers):
                provider_name = str(getattr(provider, "name", ""))
                provider_type = str(getattr(provider, "type", ""))
                weight = str(getattr(provider, "weight", ""))
                api_key = self._format_optional(getattr(provider, "api_key", None))
                endpoint = self._format_optional(getattr(provider, "endpoint", None))
                max_results = self._format_optional(getattr(provider, "max_results", None))
                language = self._format_optional(getattr(provider, "language", None))
                user_agent = self._format_optional(getattr(provider, "user_agent", None))
                timeout = self._format_optional(getattr(provider, "timeout", None))
                min_interval = self._format_optional(getattr(provider, "min_interval_seconds", None))
                section_limit = self._format_optional(getattr(provider, "section_limit", None))
                snippet_max_chars = self._format_optional(getattr(provider, "snippet_max_chars", None))
                include_disambiguation = self._format_optional(getattr(provider, "include_disambiguation", None))
                cache_results = bool(getattr(provider, "cache_results", False))
                cache_ttl = str(getattr(provider, "cache_ttl", ""))
                provider_widgets.append(
                    Container(
                        Static(f"Provider {idx + 1}", classes="settings-subsection-title"),
                        self._field("Name", f"settings-remote-provider-{idx}-name", value=provider_name),
                        self._field("Type", f"settings-remote-provider-{idx}-type", value=provider_type),
                        self._field("Weight", f"settings-remote-provider-{idx}-weight", value=weight),
                        self._field("API key", f"settings-remote-provider-{idx}-api-key", value=api_key),
                        self._field("Endpoint", f"settings-remote-provider-{idx}-endpoint", value=endpoint),
                        self._field("Max results", f"settings-remote-provider-{idx}-max-results", value=max_results),
                        self._field("Language", f"settings-remote-provider-{idx}-language", value=language),
                        self._field("User agent", f"settings-remote-provider-{idx}-user-agent", value=user_agent),
                        self._field("Timeout (s)", f"settings-remote-provider-{idx}-timeout", value=timeout),
                        self._field("Min interval (s)", f"settings-remote-provider-{idx}-min-interval", value=min_interval),
                        self._field("Section limit", f"settings-remote-provider-{idx}-section-limit", value=section_limit),
                        self._field("Snippet max chars", f"settings-remote-provider-{idx}-snippet-max-chars", value=snippet_max_chars),
                        self._field("Include disambiguation", f"settings-remote-provider-{idx}-include-disambiguation", value=include_disambiguation),
                        self._field("Cache results", f"settings-remote-provider-{idx}-cache-results", field_type="switch", value=cache_results),
                        self._field("Cache TTL", f"settings-remote-provider-{idx}-cache-ttl", value=cache_ttl),
                        Horizontal(Button("Remove Provider", id=f"settings-remote-provider-remove-{idx}"), classes="settings-actions"),
                        classes="settings-subsection",
                    )
                )
            self._replace_container_children("#settings-remote-providers-list", provider_widgets)

        chains = list(getattr(config, "remote_provider_chains", None) or [])
        if has_chains_container:
            chain_widgets: list[Widget] = []
            if not chains:
                chain_widgets.append(Static("No remote chains configured.", classes="settings-label"))
            for chain_idx, chain in enumerate(chains):
                links: list[Widget] = []
                for link_idx, link in enumerate(getattr(chain, "links", None) or []):
                    source = str(getattr(link, "source", ""))
                    mapping = ", ".join(getattr(link, "map", None) or [])
                    links.append(
                        Container(
                            Static(f"Link {link_idx + 1}", classes="settings-subsection-title"),
                            self._field("Source", f"settings-remote-chain-{chain_idx}-link-{link_idx}-source", value=source),
                            self._field("Map (comma-separated output:input)", f"settings-remote-chain-{chain_idx}-link-{link_idx}-map", value=mapping),
                            Horizontal(Button("Remove Link", id=f"settings-remote-chain-{chain_idx}-link-remove-{link_idx}"), classes="settings-actions"),
                            classes="settings-subsection",
                        )
                    )
                chain_name = str(getattr(chain, "name", ""))
                chain_description = str(getattr(chain, "agent_description", ""))
                chain_widgets.append(
                    Container(
                        Static(f"Chain {chain_idx + 1}", classes="settings-subsection-title"),
                        self._field("Name", f"settings-remote-chain-{chain_idx}-name", value=chain_name),
                        self._field("Agent description", f"settings-remote-chain-{chain_idx}-agent-description", value=chain_description),
                        *links,
                        Horizontal(Button("Add Link", id=f"settings-remote-chain-{chain_idx}-link-add"), classes="settings-actions"),
                        Horizontal(Button("Remove Chain", id=f"settings-remote-chain-remove-{chain_idx}"), classes="settings-actions"),
                        classes="settings-subsection",
                    )
                )
            self._replace_container_children("#settings-remote-chains-list", chain_widgets)

    def _replace_container_children(self, selector: str, widgets: Sequence[Widget]) -> None:
        _shared_replace_container_children(self, selector, widgets)

    def _container_exists(self, selector: str) -> bool:
        try:
            self.query_one(selector)
            return True
        except Exception:
            return False

    def _ordered_service_names(self, config: Any) -> list[str]:
        llm_cfg = getattr(config, "llm", None)
        services = getattr(llm_cfg, "services", {}) or {}
        names = list(services.keys())
        ordered: list[str] = []
        for service_name in self._LLM_COMMON_SERVICES:
            if service_name in services:
                ordered.append(service_name)
        for service_name in names:
            if service_name not in ordered:
                ordered.append(service_name)
        return ordered

    def _add_llm_provider(self, config: Any) -> None:
        from lsm.config.models import LLMProviderConfig, LLMServiceConfig

        existing = {p.provider_name for p in (config.llm.providers or [])}
        name = self._next_name(existing, "provider")
        config.llm.providers.append(LLMProviderConfig(provider_name=name))
        if not config.llm.services:
            config.llm.services["default"] = LLMServiceConfig(provider=name, model="gpt-5.2")
        self._refresh_llm_structured_fields(config)
        self._set_status(f"Added LLM provider '{name}'", False)

    def _remove_llm_provider(self, config: Any, idx: int) -> None:
        providers = config.llm.providers or []
        if idx >= len(providers):
            return
        if len(providers) <= 1:
            raise ValueError("At least one LLM provider is required.")
        removed = providers.pop(idx).provider_name
        fallback_provider = providers[0].provider_name
        for service in config.llm.services.values():
            if service.provider == removed:
                service.provider = fallback_provider
        self._refresh_llm_structured_fields(config)
        self._set_status(f"Removed LLM provider '{removed}'", False)

    def _add_llm_service(self, config: Any) -> None:
        from lsm.config.models import LLMServiceConfig

        provider_name = config.llm.providers[0].provider_name if config.llm.providers else ""
        model = "gpt-5.2"
        default_service = config.llm.services.get("default")
        if default_service is not None and default_service.model:
            model = default_service.model
        existing = set(config.llm.services.keys())
        name = self._next_name(existing, "service")
        config.llm.services[name] = LLMServiceConfig(
            provider=provider_name,
            model=model,
        )
        self._refresh_llm_structured_fields(config)
        self._set_status(f"Added LLM service '{name}'", False)

    def _remove_llm_service(self, config: Any, idx: int) -> None:
        service_names = self._ordered_service_names(config)
        if idx >= len(service_names):
            return
        if len(config.llm.services) <= 1:
            raise ValueError("At least one LLM service is required.")
        removed = service_names[idx]
        config.llm.services.pop(removed, None)
        self._refresh_llm_structured_fields(config)
        self._set_status(f"Removed LLM service '{removed}'", False)

    def _add_remote_provider(self, config: Any) -> None:
        from lsm.config.models import RemoteProviderConfig

        if config.remote_providers is None:
            config.remote_providers = []
        existing = {p.name for p in config.remote_providers}
        name = self._next_name(existing, "provider")
        config.remote_providers.append(
            RemoteProviderConfig(
                name=name,
                type="web_search",
            )
        )
        self._refresh_remote_structured_fields(config)
        self._set_status(f"Added remote provider '{name}'", False)

    def _remove_remote_provider(self, config: Any, idx: int) -> None:
        providers = config.remote_providers or []
        if idx >= len(providers):
            return
        removed = providers.pop(idx).name
        if not providers:
            config.remote_providers = None

        for mode_config in (config.modes or {}).values():
            refs = mode_config.source_policy.remote.remote_providers
            if not refs:
                continue
            filtered = []
            for ref in refs:
                if isinstance(ref, RemoteProviderRef):
                    if ref.source.lower() != removed.lower():
                        filtered.append(ref)
                elif str(ref).lower() != removed.lower():
                    filtered.append(ref)
            mode_config.source_policy.remote.remote_providers = filtered or None

        if config.remote_provider_chains:
            cleaned = []
            for chain in config.remote_provider_chains:
                chain.links = [link for link in chain.links if link.source.lower() != removed.lower()]
                if chain.links:
                    cleaned.append(chain)
            config.remote_provider_chains = cleaned or None

        self._refresh_remote_structured_fields(config)
        self._set_status(f"Removed remote provider '{removed}'", False)

    def _add_remote_chain(self, config: Any) -> None:
        from lsm.config.models import ChainLink, RemoteProviderChainConfig

        if config.remote_provider_chains is None:
            config.remote_provider_chains = []
        existing = {c.name for c in config.remote_provider_chains}
        name = self._next_name(existing, "chain")
        default_source = ""
        if config.remote_providers:
            default_source = config.remote_providers[0].name
        config.remote_provider_chains.append(
            RemoteProviderChainConfig(
                name=name,
                links=[ChainLink(source=default_source)],
            )
        )
        self._refresh_remote_structured_fields(config)
        self._set_status(f"Added remote chain '{name}'", False)

    def _remove_remote_chain(self, config: Any, idx: int) -> None:
        chains = config.remote_provider_chains or []
        if idx >= len(chains):
            return
        removed = chains.pop(idx).name
        if not chains:
            config.remote_provider_chains = None
        self._refresh_remote_structured_fields(config)
        self._set_status(f"Removed remote chain '{removed}'", False)

    def _add_remote_chain_link(self, config: Any, chain_idx: int) -> None:
        from lsm.config.models import ChainLink

        chains = config.remote_provider_chains or []
        if chain_idx >= len(chains):
            return
        default_source = ""
        if config.remote_providers:
            default_source = config.remote_providers[0].name
        chains[chain_idx].links.append(ChainLink(source=default_source))
        self._refresh_remote_structured_fields(config)
        self._set_status(f"Added link to chain '{chains[chain_idx].name}'", False)

    def _remove_remote_chain_link(self, config: Any, chain_idx: int, link_idx: int) -> None:
        chains = config.remote_provider_chains or []
        if chain_idx >= len(chains):
            return
        links = chains[chain_idx].links
        if link_idx >= len(links):
            return
        if len(links) <= 1:
            raise ValueError("A chain must keep at least one link.")
        links.pop(link_idx)
        self._refresh_remote_structured_fields(config)
        self._set_status(f"Removed link from chain '{chains[chain_idx].name}'", False)

    @staticmethod
    def _next_name(existing_names: set[str], prefix: str) -> str:
        index = 1
        while True:
            candidate = f"{prefix}_{index}"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _parse_optional_bool(value: str) -> Optional[bool]:
        text = (value or "").strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError("expected true/false or blank")

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

    def _field(
        self,
        label: str,
        field_id: str,
        *,
        placeholder: str = "",
        disabled: bool = False,
        field_type: str = "input",
        value: Any = "",
    ) -> Widget:
        return _shared_field(
            label,
            field_id,
            placeholder=placeholder,
            disabled=disabled,
            field_type=field_type,
            value=value,
        )

    def _select_field(self, label: str, field_id: str, options: Optional[list[tuple[str, str]]] = None) -> Widget:
        return _shared_select_field(label, field_id, options)

    def _set_input(self, field_id: str, value: str) -> None:
        _shared_set_input(self, field_id, value)

    def _set_switch(self, field_id: str, value: bool) -> None:
        _shared_set_switch(self, field_id, value)

    def _set_select_options(self, field_id: str, values: list[str]) -> None:
        _shared_set_select_options(self, field_id, values)

    def _set_select_value(self, field_id: str, value: str) -> None:
        _shared_set_select_value(self, field_id, value)

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)
