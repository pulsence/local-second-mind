"""Settings view model and immutable state snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Optional

from lsm.config.loader import build_config_from_raw, clone_config, config_to_raw, save_config_to_file
from lsm.config.models import (
    ChainLink,
    LLMProviderConfig,
    LLMServiceConfig,
    LSMConfig,
    RemoteProviderChainConfig,
    RemoteProviderConfig,
    RemoteProviderRef,
    RootConfig,
)


@dataclass(frozen=True)
class SettingsSnapshot:
    """Immutable snapshot of settings editor state."""

    persisted_config: LSMConfig
    draft_config: LSMConfig
    dirty_fields: frozenset[str]
    dirty_tabs: frozenset[str]
    validation_errors: dict[str, str]


@dataclass(frozen=True)
class SettingsActionResult:
    """Result value for view model actions."""

    handled: bool
    changed_tabs: tuple[str, ...] = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class SettingTableRow:
    """Single key/value row rendered in a settings command-table tab."""

    key: str
    value: str
    state: str


class SettingsViewModel:
    """ViewModel for settings editing with draft/persisted state separation."""

    _TAB_FIELD_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("settings-global", ("settings-config-path", "settings-global-")),
        ("settings-ingest", ("settings-ingest-",)),
        ("settings-query", ("settings-query-",)),
        ("settings-llm", ("settings-llm-",)),
        ("settings-vdb", ("settings-vdb-",)),
        ("settings-modes", ("settings-modes-",)),
        ("settings-remote", ("settings-remote-",)),
        ("settings-chats-notes", ("settings-chats-", "settings-notes-")),
    )

    _TAB_SECTION_KEYS: dict[str, tuple[str, ...]] = {
        "settings-global": ("global",),
        "settings-ingest": ("ingest",),
        "settings-query": ("query",),
        "settings-llm": ("llms",),
        "settings-vdb": ("vectordb",),
        "settings-modes": ("modes", "query"),
        "settings-remote": ("remote_providers", "remote_provider_chains", "modes"),
        "settings-chats-notes": ("chats", "notes"),
    }

    _LLM_COMMON_SERVICES = (
        "default",
        "query",
        "decomposition",
        "tagging",
        "ranking",
        "translation",
    )
    _KEY_TOKEN_RE = re.compile(r"([^[\].]+)|\[(\d+)\]")

    def __init__(
        self,
        config: LSMConfig,
        *,
        clone_func: Callable[[LSMConfig], LSMConfig] = clone_config,
    ) -> None:
        self._clone = clone_func
        self._persisted_config = self._clone(config)
        self._draft_config = self._clone(config)
        self._dirty_fields: set[str] = set()
        self._dirty_tabs: set[str] = set()
        self._validation_errors: dict[str, str] = {}

    @property
    def persisted_config(self) -> LSMConfig:
        """Last saved/loaded settings config snapshot."""
        return self._persisted_config

    @property
    def draft_config(self) -> LSMConfig:
        """Editable draft config snapshot."""
        return self._draft_config

    @property
    def dirty_fields(self) -> frozenset[str]:
        """Field ids that have unsaved draft edits."""
        return frozenset(self._dirty_fields)

    @property
    def dirty_tabs(self) -> frozenset[str]:
        """Tab ids that have unsaved draft edits."""
        return frozenset(self._dirty_tabs)

    @property
    def validation_errors(self) -> dict[str, str]:
        """Current validation errors indexed by field/action id."""
        return dict(self._validation_errors)

    def snapshot(self) -> SettingsSnapshot:
        """Return an immutable copy of current editor state."""
        return SettingsSnapshot(
            persisted_config=self._clone(self._persisted_config),
            draft_config=self._clone(self._draft_config),
            dirty_fields=frozenset(self._dirty_fields),
            dirty_tabs=frozenset(self._dirty_tabs),
            validation_errors=dict(self._validation_errors),
        )

    def load(self, config: LSMConfig) -> SettingsActionResult:
        """Load a new persisted config and reset draft state."""
        self._persisted_config = self._clone(config)
        self._draft_config = self._clone(config)
        self._dirty_fields.clear()
        self._dirty_tabs.clear()
        self._validation_errors.clear()
        return SettingsActionResult(handled=True, changed_tabs=tuple(self._TAB_SECTION_KEYS.keys()))

    def update_field(self, field_id: str, value: Any) -> SettingsActionResult:
        """Apply a single field change to draft config."""
        if not field_id:
            return SettingsActionResult(handled=False)

        tab_id = self._tab_for_field(field_id)
        if tab_id is None:
            return SettingsActionResult(handled=False)

        text = self._as_text(value)
        try:
            handled = self._update_field_inner(field_id, value, text)
        except Exception as exc:
            self._validation_errors[field_id] = str(exc)
            return SettingsActionResult(handled=True, changed_tabs=(tab_id,), error=str(exc))

        if not handled:
            return SettingsActionResult(handled=False)

        self._validation_errors.pop(field_id, None)
        self._dirty_fields.add(field_id)

        changed_tabs = {tab_id}
        if field_id in {"settings-query-mode", "settings-modes-mode"}:
            changed_tabs.update({"settings-query", "settings-modes"})
        if field_id.startswith("settings-remote-provider-") and field_id.endswith("-name"):
            changed_tabs.update({"settings-remote", "settings-modes"})

        self._dirty_tabs.update(changed_tabs)
        self._recompute_dirty_tabs()
        return SettingsActionResult(handled=True, changed_tabs=tuple(sorted(changed_tabs)))

    def add_item(self, action_id: str) -> SettingsActionResult:
        """Apply a structured add-item action to draft config."""
        if not action_id:
            return SettingsActionResult(handled=False)

        tab_id = self._tab_for_field(action_id)
        if tab_id is None:
            return SettingsActionResult(handled=False)

        try:
            handled, changed_tabs = self._add_item_inner(action_id)
        except Exception as exc:
            self._validation_errors[action_id] = str(exc)
            return SettingsActionResult(handled=True, changed_tabs=(tab_id,), error=str(exc))

        if not handled:
            return SettingsActionResult(handled=False)

        self._validation_errors.pop(action_id, None)
        self._dirty_fields.add(action_id)
        changed_tabs = set(changed_tabs) or {tab_id}
        self._dirty_tabs.update(changed_tabs)
        self._recompute_dirty_tabs()
        return SettingsActionResult(handled=True, changed_tabs=tuple(sorted(changed_tabs)))

    def remove_item(self, action_id: str) -> SettingsActionResult:
        """Apply a structured remove-item action to draft config."""
        if not action_id:
            return SettingsActionResult(handled=False)

        tab_id = self._tab_for_field(action_id)
        if tab_id is None:
            return SettingsActionResult(handled=False)

        try:
            handled, changed_tabs = self._remove_item_inner(action_id)
        except Exception as exc:
            self._validation_errors[action_id] = str(exc)
            return SettingsActionResult(handled=True, changed_tabs=(tab_id,), error=str(exc))

        if not handled:
            return SettingsActionResult(handled=False)

        self._validation_errors.pop(action_id, None)
        self._dirty_fields.add(action_id)
        changed_tabs = set(changed_tabs) or {tab_id}
        self._dirty_tabs.update(changed_tabs)
        self._recompute_dirty_tabs()
        return SettingsActionResult(handled=True, changed_tabs=tuple(sorted(changed_tabs)))

    def rename_key(self, kind: str, old_key: str, new_key: str) -> SettingsActionResult:
        """Apply a typed key-rename action."""
        old_name = (old_key or "").strip()
        new_name = (new_key or "").strip()
        if not old_name or not new_name:
            return SettingsActionResult(handled=False)

        try:
            changed_tabs = self._rename_key_inner(kind, old_name, new_name)
        except Exception as exc:
            action_id = f"rename:{kind}:{old_name}"
            self._validation_errors[action_id] = str(exc)
            return SettingsActionResult(handled=True, changed_tabs=(), error=str(exc))

        if not changed_tabs:
            return SettingsActionResult(handled=False)

        self._dirty_fields.add(f"rename:{kind}:{old_name}:{new_name}")
        self._dirty_tabs.update(changed_tabs)
        self._recompute_dirty_tabs()
        return SettingsActionResult(handled=True, changed_tabs=tuple(sorted(changed_tabs)))

    def reset_tab(self, tab_id: str) -> SettingsActionResult:
        """Reset one tab's draft state back to the persisted state."""
        keys = self._TAB_SECTION_KEYS.get(tab_id)
        if not keys:
            return SettingsActionResult(handled=False)

        self._replace_draft_sections(keys)
        self._clear_dirty_for_tab(tab_id)
        self._clear_errors_for_tab(tab_id)

        changed_tabs = {tab_id}
        if tab_id in {"settings-query", "settings-modes"}:
            changed_tabs.update({"settings-query", "settings-modes"})
        if tab_id == "settings-remote":
            changed_tabs.add("settings-modes")

        self._recompute_dirty_tabs()
        return SettingsActionResult(handled=True, changed_tabs=tuple(sorted(changed_tabs)))

    def reset_all(self) -> SettingsActionResult:
        """Reset full draft state back to persisted state."""
        self._draft_config = self._clone(self._persisted_config)
        self._dirty_fields.clear()
        self._dirty_tabs.clear()
        self._validation_errors.clear()
        return SettingsActionResult(handled=True, changed_tabs=tuple(self._TAB_SECTION_KEYS.keys()))

    def save(
        self,
        *,
        saver: Callable[[LSMConfig, Path | str], None] = save_config_to_file,
    ) -> SettingsActionResult:
        """Validate draft state and persist it to disk."""
        cfg_path = self._draft_config.config_path
        if cfg_path is None:
            error = "No config path available for save."
            self._validation_errors["_save"] = error
            return SettingsActionResult(handled=True, error=error)

        try:
            self._draft_config.validate()
            saver(self._draft_config, cfg_path)
        except Exception as exc:
            self._validation_errors["_save"] = str(exc)
            return SettingsActionResult(handled=True, error=str(exc))

        self._persisted_config = self._clone(self._draft_config)
        self._dirty_fields.clear()
        self._dirty_tabs.clear()
        self._validation_errors.clear()
        return SettingsActionResult(handled=True, changed_tabs=tuple(self._TAB_SECTION_KEYS.keys()))

    def table_rows(self, tab_id: str) -> list[SettingTableRow]:
        """Build flattened key/value rows for a settings tab."""
        section_keys = self._TAB_SECTION_KEYS.get(tab_id)
        if not section_keys:
            return []

        draft_raw = config_to_raw(self._draft_config)
        persisted_raw = config_to_raw(self._persisted_config)
        show_section_prefix = len(section_keys) > 1

        rows: list[SettingTableRow] = []
        for section in section_keys:
            draft_value = draft_raw.get(section)
            persisted_value = persisted_raw.get(section)
            prefix = section if show_section_prefix else ""
            self._flatten_rows(prefix, draft_value, persisted_value, rows)

        rows.sort(key=lambda row: row.key)
        return rows

    def set_key(self, tab_id: str, key: str, value_text: str) -> SettingsActionResult:
        """Set one key path in the draft config for a tab."""
        return self._mutate_key_path(
            tab_id=tab_id,
            key=key,
            mutator=lambda draft_raw, persisted_raw, tokens: self._set_key_path_value(
                draft_raw,
                persisted_raw,
                tokens,
                value_text,
            ),
            field_marker=f"cmd:set:{tab_id}:{key}",
        )

    def unset_key(self, tab_id: str, key: str) -> SettingsActionResult:
        """Set one key path to null in the draft config for a tab."""
        return self._mutate_key_path(
            tab_id=tab_id,
            key=key,
            mutator=lambda draft_raw, _persisted_raw, tokens: self._write_path_value(
                draft_raw,
                tokens,
                None,
            ),
            field_marker=f"cmd:unset:{tab_id}:{key}",
        )

    def delete_key(self, tab_id: str, key: str) -> SettingsActionResult:
        """Delete one key path from the draft config for a tab."""
        return self._mutate_key_path(
            tab_id=tab_id,
            key=key,
            mutator=lambda draft_raw, _persisted_raw, tokens: self._delete_path(draft_raw, tokens),
            field_marker=f"cmd:delete:{tab_id}:{key}",
        )

    def reset_key(self, tab_id: str, key: str) -> SettingsActionResult:
        """Reset one key path back to persisted value for a tab."""
        return self._mutate_key_path(
            tab_id=tab_id,
            key=key,
            mutator=self._reset_path_from_persisted,
            field_marker=f"cmd:reset:{tab_id}:{key}",
        )

    def default_key(self, tab_id: str, key: str) -> SettingsActionResult:
        """Reset one key path to default by deleting it from raw draft config."""
        return self._mutate_key_path(
            tab_id=tab_id,
            key=key,
            mutator=lambda draft_raw, _persisted_raw, tokens: self._delete_path(draft_raw, tokens),
            field_marker=f"cmd:default:{tab_id}:{key}",
        )

    def _update_field_inner(self, field_id: str, value: Any, text: str) -> bool:
        cfg = self._draft_config

        root_match = re.fullmatch(r"settings-ingest-root-(\d+)-([a-z-]+)", field_id)
        if root_match:
            idx = int(root_match.group(1))
            field = root_match.group(2)
            roots = cfg.ingest.roots or []
            if idx >= len(roots):
                return True
            root = roots[idx]
            if field == "path":
                if not text:
                    raise ValueError("root path cannot be empty")
                root.path = Path(text)
            elif field == "tags":
                root.tags = self._parse_csv(text)
            elif field == "content-type":
                root.content_type = text or None
            return True

        provider_match = re.fullmatch(r"settings-llm-provider-(\d+)-([a-z-]+)", field_id)
        if provider_match:
            idx = int(provider_match.group(1))
            field = provider_match.group(2)
            providers = cfg.llm.providers or []
            if idx >= len(providers):
                return True
            provider = providers[idx]
            if field == "name":
                if not text:
                    raise ValueError("provider name cannot be empty")
                if provider.provider_name != text:
                    self._rename_llm_provider(provider.provider_name, text)
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
            service_names = self._ordered_service_names()
            if idx >= len(service_names):
                return True
            service = cfg.llm.services[service_names[idx]]
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
            providers = cfg.remote_providers or []
            if idx >= len(providers):
                return True
            provider = providers[idx]
            if field == "name":
                if not text:
                    raise ValueError("provider name cannot be empty")
                if provider.name != text:
                    self._rename_remote_provider(provider.name, text)
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
            elif field == "cache-results":
                provider.cache_results = bool(value)
            elif field == "cache-ttl":
                provider.cache_ttl = int(text) if text else 86400
            return True

        chain_match = re.fullmatch(r"settings-remote-chain-(\d+)-([a-z-]+)", field_id)
        if chain_match:
            idx = int(chain_match.group(1))
            field = chain_match.group(2)
            chains = cfg.remote_provider_chains or []
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
            chains = cfg.remote_provider_chains or []
            if chain_idx >= len(chains):
                return True
            links = chains[chain_idx].links
            if link_idx >= len(links):
                return True
            link = links[link_idx]
            if field == "source":
                link.source = text
            elif field == "map":
                link.map = self._parse_csv(text)
            return True

        if field_id == "settings-global-folder":
            cfg.global_settings.global_folder = Path(text) if text else None
        elif field_id == "settings-global-embed-model":
            cfg.global_settings.embed_model = text
        elif field_id == "settings-global-device":
            cfg.global_settings.device = text
        elif field_id == "settings-global-batch-size" and text:
            cfg.global_settings.batch_size = int(text)
        elif field_id == "settings-global-embedding-dimension":
            cfg.global_settings.embedding_dimension = int(text) if text else None
        elif field_id == "settings-ingest-chunking-strategy":
            cfg.ingest.chunking_strategy = text
        elif field_id == "settings-ingest-chunk-size" and text:
            cfg.ingest.chunk_size = int(text)
        elif field_id == "settings-ingest-chunk-overlap" and text:
            cfg.ingest.chunk_overlap = int(text)
        elif field_id == "settings-ingest-tags-per-chunk" and text:
            cfg.ingest.tags_per_chunk = int(text)
        elif field_id == "settings-ingest-translation-target":
            cfg.ingest.translation_target = text or "en"
        elif field_id == "settings-ingest-extensions":
            cfg.ingest.extensions = self._parse_csv(text)
        elif field_id == "settings-ingest-exclude-dirs":
            cfg.ingest.exclude_dirs = self._parse_csv(text)
        elif field_id == "settings-ingest-max-files":
            cfg.ingest.max_files = int(text) if text else None
        elif field_id == "settings-ingest-max-seconds":
            cfg.ingest.max_seconds = int(text) if text else None
        elif field_id == "settings-ingest-override-extensions":
            cfg.ingest.override_extensions = bool(value)
        elif field_id == "settings-ingest-override-excludes":
            cfg.ingest.override_excludes = bool(value)
        elif field_id == "settings-ingest-dry-run":
            cfg.ingest.dry_run = bool(value)
        elif field_id == "settings-ingest-skip-errors":
            cfg.ingest.skip_errors = bool(value)
        elif field_id == "settings-ingest-enable-ocr":
            cfg.ingest.enable_ocr = bool(value)
        elif field_id == "settings-ingest-enable-ai-tagging":
            cfg.ingest.enable_ai_tagging = bool(value)
        elif field_id == "settings-ingest-enable-language-detection":
            cfg.ingest.enable_language_detection = bool(value)
        elif field_id == "settings-ingest-enable-translation":
            cfg.ingest.enable_translation = bool(value)
        elif field_id == "settings-query-mode":
            cfg.query.mode = text
        elif field_id == "settings-query-k" and text:
            cfg.query.k = int(text)
        elif field_id == "settings-query-retrieve-k":
            cfg.query.retrieve_k = int(text) if text else None
        elif field_id == "settings-query-min-relevance" and text:
            cfg.query.min_relevance = float(text)
        elif field_id == "settings-query-local-pool":
            cfg.query.local_pool = int(text) if text else None
        elif field_id == "settings-query-max-per-file" and text:
            cfg.query.max_per_file = int(text)
        elif field_id == "settings-query-path-contains":
            cfg.query.path_contains = self._parse_csv(text)
        elif field_id == "settings-query-ext-allow":
            cfg.query.ext_allow = self._parse_csv(text)
        elif field_id == "settings-query-ext-deny":
            cfg.query.ext_deny = self._parse_csv(text)
        elif field_id == "settings-query-rerank-strategy":
            cfg.query.rerank_strategy = text
        elif field_id == "settings-query-chat-mode":
            cfg.query.chat_mode = text
        elif field_id == "settings-query-cache-ttl" and text:
            cfg.query.query_cache_ttl = int(text)
        elif field_id == "settings-query-cache-size" and text:
            cfg.query.query_cache_size = int(text)
        elif field_id == "settings-query-no-rerank":
            cfg.query.no_rerank = bool(value)
            if cfg.query.no_rerank:
                cfg.query.rerank_strategy = "none"
        elif field_id == "settings-query-enable-cache":
            cfg.query.enable_query_cache = bool(value)
        elif field_id == "settings-query-enable-llm-server-cache":
            cfg.query.enable_llm_server_cache = bool(value)
        elif field_id == "settings-vdb-provider":
            cfg.vectordb.provider = text
        elif field_id == "settings-vdb-collection":
            cfg.vectordb.collection = text
        elif field_id == "settings-vdb-path":
            cfg.vectordb.path = Path(text)
        elif field_id == "settings-vdb-connection-string":
            cfg.vectordb.connection_string = text or None
        elif field_id == "settings-vdb-host":
            cfg.vectordb.host = text or None
        elif field_id == "settings-vdb-port":
            cfg.vectordb.port = int(text) if text else None
        elif field_id == "settings-vdb-database":
            cfg.vectordb.database = text or None
        elif field_id == "settings-vdb-user":
            cfg.vectordb.user = text or None
        elif field_id == "settings-vdb-password":
            if text:
                cfg.vectordb.password = text
        elif field_id == "settings-vdb-index-type":
            cfg.vectordb.index_type = text or "hnsw"
        elif field_id == "settings-vdb-pool-size" and text:
            cfg.vectordb.pool_size = int(text)
        elif field_id == "settings-modes-mode":
            cfg.query.mode = text
        elif field_id == "settings-notes-enabled":
            cfg.notes.enabled = bool(value)
        elif field_id == "settings-notes-dir":
            cfg.notes.dir = text or "notes"
        elif field_id == "settings-notes-template":
            cfg.notes.template = text or "default"
        elif field_id == "settings-notes-filename-format":
            cfg.notes.filename_format = text or "timestamp"
        elif field_id == "settings-notes-integration":
            cfg.notes.integration = text or "none"
        elif field_id == "settings-notes-wikilinks":
            cfg.notes.wikilinks = bool(value)
        elif field_id == "settings-notes-backlinks":
            cfg.notes.backlinks = bool(value)
        elif field_id == "settings-notes-include-tags":
            cfg.notes.include_tags = bool(value)
        elif field_id == "settings-chats-enabled":
            cfg.chats.enabled = bool(value)
        elif field_id == "settings-chats-dir":
            cfg.chats.dir = text or "Chats"
        elif field_id == "settings-chats-auto-save":
            cfg.chats.auto_save = bool(value)
        elif field_id == "settings-chats-format":
            cfg.chats.format = text or "markdown"
        else:
            return False

        return True

    def _mutate_key_path(
        self,
        *,
        tab_id: str,
        key: str,
        mutator: Callable[[dict[str, Any], dict[str, Any], list[Any]], bool],
        field_marker: str,
    ) -> SettingsActionResult:
        key_text = (key or "").strip()
        if not key_text:
            return SettingsActionResult(handled=True, error="key is required")

        section_keys = self._TAB_SECTION_KEYS.get(tab_id)
        if not section_keys:
            return SettingsActionResult(handled=False)

        try:
            abs_tokens = self._resolve_key_tokens(tab_id, key_text)
        except Exception as exc:
            return SettingsActionResult(handled=True, changed_tabs=(tab_id,), error=str(exc))

        draft_raw = config_to_raw(self._draft_config)
        persisted_raw = config_to_raw(self._persisted_config)

        try:
            changed = mutator(draft_raw, persisted_raw, abs_tokens)
            if not changed:
                return SettingsActionResult(handled=True, changed_tabs=(tab_id,))
            path = self._draft_config.config_path or self._persisted_config.config_path or Path("config.json")
            self._draft_config = build_config_from_raw(draft_raw, path)
        except Exception as exc:
            return SettingsActionResult(handled=True, changed_tabs=(tab_id,), error=str(exc))

        self._dirty_fields.add(field_marker)
        self._dirty_tabs.add(tab_id)
        self._recompute_dirty_tabs()
        changed_tabs = tuple(sorted(set(self._dirty_tabs) | {tab_id}))
        return SettingsActionResult(handled=True, changed_tabs=changed_tabs)

    def _resolve_key_tokens(self, tab_id: str, key_text: str) -> list[Any]:
        rel_tokens = self._parse_key_tokens(key_text)
        if not rel_tokens:
            raise ValueError("invalid key path")

        section_keys = self._TAB_SECTION_KEYS.get(tab_id) or ()
        if len(section_keys) == 1:
            return [section_keys[0], *rel_tokens]

        first = rel_tokens[0]
        if not isinstance(first, str) or first not in section_keys:
            allowed = ", ".join(section_keys)
            raise ValueError(f"key must start with one of: {allowed}")
        return rel_tokens

    def _parse_key_tokens(self, key_text: str) -> list[Any]:
        tokens: list[Any] = []
        for match in self._KEY_TOKEN_RE.finditer(key_text):
            dict_key = match.group(1)
            list_index = match.group(2)
            if dict_key is not None:
                tokens.append(dict_key)
            elif list_index is not None:
                tokens.append(int(list_index))
        return tokens

    def _set_key_path_value(
        self,
        draft_raw: dict[str, Any],
        persisted_raw: dict[str, Any],
        tokens: list[Any],
        value_text: str,
    ) -> bool:
        current_exists, current_value = self._read_path(draft_raw, tokens)
        if not current_exists:
            current_exists, current_value = self._read_path(persisted_raw, tokens)

        parsed = self._parse_value(value_text, current_value if current_exists else None)
        return self._write_path_value(draft_raw, tokens, parsed)

    def _reset_path_from_persisted(
        self,
        draft_raw: dict[str, Any],
        persisted_raw: dict[str, Any],
        tokens: list[Any],
    ) -> bool:
        exists, value = self._read_path(persisted_raw, tokens)
        if exists:
            return self._write_path_value(draft_raw, tokens, deepcopy(value))
        return self._delete_path(draft_raw, tokens)

    def _write_path_value(self, root: Any, tokens: list[Any], value: Any) -> bool:
        if not tokens:
            return False

        node = root
        for idx, token in enumerate(tokens[:-1]):
            next_token = tokens[idx + 1]
            if isinstance(token, str):
                if not isinstance(node, dict):
                    raise ValueError("invalid path (expected object)")
                if token not in node or node[token] is None:
                    node[token] = [] if isinstance(next_token, int) else {}
                node = node[token]
                continue

            if not isinstance(node, list):
                raise ValueError("invalid path (expected list)")
            if token < 0 or token >= len(node):
                raise ValueError("list index out of range")
            if node[token] is None:
                node[token] = [] if isinstance(next_token, int) else {}
            node = node[token]

        last = tokens[-1]
        if isinstance(last, str):
            if not isinstance(node, dict):
                raise ValueError("invalid path (expected object)")
            node[last] = value
            return True

        if not isinstance(node, list):
            raise ValueError("invalid path (expected list)")
        if last < 0 or last >= len(node):
            raise ValueError("list index out of range")
        node[last] = value
        return True

    def _delete_path(self, root: Any, tokens: list[Any]) -> bool:
        if not tokens:
            return False

        node = root
        for token in tokens[:-1]:
            if isinstance(token, str):
                if not isinstance(node, dict) or token not in node:
                    raise ValueError("key not found")
                node = node[token]
                continue
            if not isinstance(node, list) or token < 0 or token >= len(node):
                raise ValueError("list index out of range")
            node = node[token]

        last = tokens[-1]
        if isinstance(last, str):
            if not isinstance(node, dict) or last not in node:
                raise ValueError("key not found")
            del node[last]
            return True

        if not isinstance(node, list) or last < 0 or last >= len(node):
            raise ValueError("list index out of range")
        node.pop(last)
        return True

    def _read_path(self, root: Any, tokens: list[Any]) -> tuple[bool, Any]:
        node = root
        for token in tokens:
            if isinstance(token, str):
                if not isinstance(node, dict) or token not in node:
                    return False, None
                node = node[token]
                continue

            if not isinstance(node, list) or token < 0 or token >= len(node):
                return False, None
            node = node[token]
        return True, node

    def _parse_value(self, value_text: str, current_value: Any) -> Any:
        text = (value_text or "").strip()
        if text == "":
            return ""

        if isinstance(current_value, bool):
            lowered = text.lower()
            if lowered in {"1", "true", "yes", "on", "y"}:
                return True
            if lowered in {"0", "false", "no", "off", "n"}:
                return False
            raise ValueError("expected boolean value")

        if isinstance(current_value, int) and not isinstance(current_value, bool):
            return int(text)

        if isinstance(current_value, float):
            return float(text)

        if isinstance(current_value, list):
            if text.startswith("["):
                value = json.loads(text)
                if not isinstance(value, list):
                    raise ValueError("expected JSON array")
                return value
            return [item.strip() for item in text.split(",") if item.strip()]

        lowered = text.lower()
        if lowered == "null":
            return None
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.\d+", text):
            return float(text)
        if text.startswith("[") or text.startswith("{"):
            return json.loads(text)
        return text

    def _flatten_rows(
        self,
        key: str,
        draft_value: Any,
        persisted_value: Any,
        rows: list[SettingTableRow],
    ) -> None:
        if isinstance(draft_value, dict):
            if not draft_value and key:
                rows.append(
                    SettingTableRow(
                        key=key,
                        value="{}",
                        state="dirty" if draft_value != persisted_value else "",
                    )
                )
                return

            for child_key in sorted(draft_value.keys()):
                child_path = f"{key}.{child_key}" if key else str(child_key)
                child_persisted = (
                    persisted_value.get(child_key)
                    if isinstance(persisted_value, dict)
                    else None
                )
                self._flatten_rows(child_path, draft_value[child_key], child_persisted, rows)
            return

        if isinstance(draft_value, list):
            if not draft_value and key:
                rows.append(
                    SettingTableRow(
                        key=key,
                        value="[]",
                        state="dirty" if draft_value != persisted_value else "",
                    )
                )
                return

            for index, child_value in enumerate(draft_value):
                child_path = f"{key}[{index}]"
                child_persisted = (
                    persisted_value[index]
                    if isinstance(persisted_value, list) and index < len(persisted_value)
                    else None
                )
                self._flatten_rows(child_path, child_value, child_persisted, rows)
            return

        if not key:
            return

        rows.append(
            SettingTableRow(
                key=key,
                value=self._format_row_value(draft_value),
                state="dirty" if draft_value != persisted_value else "",
            )
        )

    @staticmethod
    def _format_row_value(value: Any) -> str:
        if value is None:
            return "<null>"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True)
        return str(value)

    def _add_item_inner(self, action_id: str) -> tuple[bool, set[str]]:
        cfg = self._draft_config
        if action_id == "settings-ingest-root-add":
            cfg.ingest.roots.append(RootConfig(path=Path(".")))
            return True, {"settings-ingest"}

        if action_id == "settings-llm-provider-add":
            existing = {p.provider_name for p in (cfg.llm.providers or [])}
            name = self._next_name(existing, "provider")
            cfg.llm.providers.append(LLMProviderConfig(provider_name=name))
            if not cfg.llm.services:
                cfg.llm.services["default"] = LLMServiceConfig(provider=name, model="gpt-5.2")
            return True, {"settings-llm"}

        if action_id == "settings-llm-service-add":
            provider_name = cfg.llm.providers[0].provider_name if cfg.llm.providers else ""
            model = "gpt-5.2"
            default_service = cfg.llm.services.get("default")
            if default_service is not None and default_service.model:
                model = default_service.model
            existing = set(cfg.llm.services.keys())
            name = self._next_name(existing, "service")
            cfg.llm.services[name] = LLMServiceConfig(provider=provider_name, model=model)
            return True, {"settings-llm"}

        if action_id == "settings-remote-provider-add":
            if cfg.remote_providers is None:
                cfg.remote_providers = []
            existing = {p.name for p in cfg.remote_providers}
            name = self._next_name(existing, "provider")
            cfg.remote_providers.append(RemoteProviderConfig(name=name, type="web_search"))
            return True, {"settings-remote"}

        if action_id == "settings-remote-chain-add":
            if cfg.remote_provider_chains is None:
                cfg.remote_provider_chains = []
            existing = {c.name for c in cfg.remote_provider_chains}
            name = self._next_name(existing, "chain")
            default_source = cfg.remote_providers[0].name if cfg.remote_providers else ""
            cfg.remote_provider_chains.append(
                RemoteProviderChainConfig(name=name, links=[ChainLink(source=default_source)])
            )
            return True, {"settings-remote"}

        add_link = re.fullmatch(r"settings-remote-chain-(\d+)-link-add", action_id)
        if add_link:
            chain_idx = int(add_link.group(1))
            chains = cfg.remote_provider_chains or []
            if chain_idx >= len(chains):
                return True, {"settings-remote"}
            default_source = cfg.remote_providers[0].name if cfg.remote_providers else ""
            chains[chain_idx].links.append(ChainLink(source=default_source))
            return True, {"settings-remote"}

        return False, set()

    def _remove_item_inner(self, action_id: str) -> tuple[bool, set[str]]:
        cfg = self._draft_config

        root_remove = re.fullmatch(r"settings-ingest-root-remove-(\d+)", action_id)
        if root_remove:
            idx = int(root_remove.group(1))
            roots = cfg.ingest.roots or []
            if idx >= len(roots):
                return True, {"settings-ingest"}
            if len(roots) <= 1:
                raise ValueError("At least one ingest root is required.")
            roots.pop(idx)
            return True, {"settings-ingest"}

        llm_provider_remove = re.fullmatch(r"settings-llm-provider-remove-(\d+)", action_id)
        if llm_provider_remove:
            idx = int(llm_provider_remove.group(1))
            providers = cfg.llm.providers or []
            if idx >= len(providers):
                return True, {"settings-llm"}
            if len(providers) <= 1:
                raise ValueError("At least one LLM provider is required.")
            removed = providers.pop(idx).provider_name
            fallback_provider = providers[0].provider_name
            for service in cfg.llm.services.values():
                if service.provider == removed:
                    service.provider = fallback_provider
            return True, {"settings-llm"}

        llm_service_remove = re.fullmatch(r"settings-llm-service-remove-(\d+)", action_id)
        if llm_service_remove:
            idx = int(llm_service_remove.group(1))
            service_names = self._ordered_service_names()
            if idx >= len(service_names):
                return True, {"settings-llm"}
            if len(cfg.llm.services) <= 1:
                raise ValueError("At least one LLM service is required.")
            removed = service_names[idx]
            cfg.llm.services.pop(removed, None)
            return True, {"settings-llm"}

        remote_provider_remove = re.fullmatch(r"settings-remote-provider-remove-(\d+)", action_id)
        if remote_provider_remove:
            idx = int(remote_provider_remove.group(1))
            providers = cfg.remote_providers or []
            if idx >= len(providers):
                return True, {"settings-remote"}

            removed = providers.pop(idx).name
            if not providers:
                cfg.remote_providers = None

            for mode_config in (cfg.modes or {}).values():
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

            if cfg.remote_provider_chains:
                cleaned = []
                for chain in cfg.remote_provider_chains:
                    chain.links = [link for link in chain.links if link.source.lower() != removed.lower()]
                    if chain.links:
                        cleaned.append(chain)
                cfg.remote_provider_chains = cleaned or None

            return True, {"settings-remote", "settings-modes"}

        remote_chain_remove = re.fullmatch(r"settings-remote-chain-remove-(\d+)", action_id)
        if remote_chain_remove:
            idx = int(remote_chain_remove.group(1))
            chains = cfg.remote_provider_chains or []
            if idx >= len(chains):
                return True, {"settings-remote"}
            chains.pop(idx)
            if not chains:
                cfg.remote_provider_chains = None
            return True, {"settings-remote"}

        remove_link = re.fullmatch(r"settings-remote-chain-(\d+)-link-remove-(\d+)", action_id)
        if remove_link:
            chain_idx = int(remove_link.group(1))
            link_idx = int(remove_link.group(2))
            chains = cfg.remote_provider_chains or []
            if chain_idx >= len(chains):
                return True, {"settings-remote"}
            links = chains[chain_idx].links
            if link_idx >= len(links):
                return True, {"settings-remote"}
            if len(links) <= 1:
                raise ValueError("A chain must keep at least one link.")
            links.pop(link_idx)
            return True, {"settings-remote"}

        return False, set()

    def _rename_key_inner(self, kind: str, old_key: str, new_key: str) -> set[str]:
        if kind == "llm_provider":
            self._rename_llm_provider(old_key, new_key)
            return {"settings-llm"}
        if kind == "llm_service":
            services = self._draft_config.llm.services
            if old_key not in services:
                return set()
            if new_key in services and new_key != old_key:
                raise ValueError(f"service '{new_key}' already exists")
            services[new_key] = services.pop(old_key)
            return {"settings-llm"}
        if kind == "remote_provider":
            self._rename_remote_provider(old_key, new_key)
            return {"settings-remote", "settings-modes"}
        return set()

    def _rename_llm_provider(self, old_name: str, new_name: str) -> None:
        providers = self._draft_config.llm.providers or []
        target = next((p for p in providers if p.provider_name == old_name), None)
        if target is None:
            raise ValueError(f"provider '{old_name}' not found")
        if any(p.provider_name == new_name for p in providers if p is not target):
            raise ValueError(f"provider '{new_name}' already exists")

        target.provider_name = new_name
        for service in self._draft_config.llm.services.values():
            if service.provider == old_name:
                service.provider = new_name

    def _rename_remote_provider(self, old_name: str, new_name: str) -> None:
        providers = self._draft_config.remote_providers or []
        target = next((p for p in providers if p.name == old_name), None)
        if target is None:
            raise ValueError(f"remote provider '{old_name}' not found")
        if any(p.name == new_name for p in providers if p is not target):
            raise ValueError(f"remote provider '{new_name}' already exists")

        target.name = new_name

        for mode_config in (self._draft_config.modes or {}).values():
            refs = mode_config.source_policy.remote.remote_providers or []
            for ref_idx, ref in enumerate(refs):
                if isinstance(ref, RemoteProviderRef):
                    if ref.source == old_name:
                        ref.source = new_name
                elif str(ref) == old_name:
                    refs[ref_idx] = new_name

        for chain in self._draft_config.remote_provider_chains or []:
            for link in chain.links:
                if link.source == old_name:
                    link.source = new_name

    def _replace_draft_sections(self, section_keys: tuple[str, ...]) -> None:
        persisted_raw = config_to_raw(self._persisted_config)
        draft_raw = config_to_raw(self._draft_config)
        for key in section_keys:
            draft_raw[key] = deepcopy(persisted_raw.get(key))
        path = self._draft_config.config_path or self._persisted_config.config_path or Path("config.json")
        self._draft_config = build_config_from_raw(draft_raw, path)

    def _recompute_dirty_tabs(self) -> None:
        persisted_raw = config_to_raw(self._persisted_config)
        draft_raw = config_to_raw(self._draft_config)

        computed_dirty: set[str] = set()
        for tab_id, keys in self._TAB_SECTION_KEYS.items():
            if any(persisted_raw.get(key) != draft_raw.get(key) for key in keys):
                computed_dirty.add(tab_id)

        self._dirty_tabs = computed_dirty
        if not self._dirty_tabs:
            self._dirty_fields.clear()
        else:
            self._dirty_fields = {
                field_id
                for field_id in self._dirty_fields
                if self._tab_for_field(field_id) in self._dirty_tabs
            }

    def _clear_dirty_for_tab(self, tab_id: str) -> None:
        prefixes = dict(self._TAB_FIELD_PREFIXES).get(tab_id, ())
        self._dirty_fields = {
            field_id
            for field_id in self._dirty_fields
            if not any(field_id.startswith(prefix) for prefix in prefixes)
        }

    def _clear_errors_for_tab(self, tab_id: str) -> None:
        prefixes = dict(self._TAB_FIELD_PREFIXES).get(tab_id, ())
        self._validation_errors = {
            key: value
            for key, value in self._validation_errors.items()
            if not any(key.startswith(prefix) for prefix in prefixes)
        }

    def _ordered_service_names(self) -> list[str]:
        services = self._draft_config.llm.services or {}
        names = list(services.keys())
        ordered: list[str] = []
        for service_name in self._LLM_COMMON_SERVICES:
            if service_name in services:
                ordered.append(service_name)
        for service_name in names:
            if service_name not in ordered:
                ordered.append(service_name)
        return ordered

    def _tab_for_field(self, field_id: str) -> Optional[str]:
        for tab_id, prefixes in self._TAB_FIELD_PREFIXES:
            for prefix in prefixes:
                if field_id.startswith(prefix):
                    return tab_id
        return None

    @staticmethod
    def _next_name(existing_names: set[str], prefix: str) -> str:
        index = 1
        while True:
            candidate = f"{prefix}_{index}"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _parse_csv(text: str) -> Optional[list[str]]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        return values or None

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

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
