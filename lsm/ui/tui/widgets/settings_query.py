"""Query settings tab widget."""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class QuerySettingsTab(BaseSettingsTab):
    """Settings view for query configuration fields."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Query Settings", classes="settings-section-title"),
            self._select_field("Mode", "settings-query-mode"),
            self._field("k", "settings-query-k"),
            self._field("Retrieve k", "settings-query-retrieve-k"),
            self._field("Min relevance", "settings-query-min-relevance"),
            self._select_field(
                "Retrieval profile",
                "settings-query-retrieval-profile",
                [
                    ("dense_only", "dense_only"),
                    ("hybrid_rrf", "hybrid_rrf"),
                    ("llm_rerank", "llm_rerank"),
                ],
            ),
            self._field("k dense", "settings-query-k-dense"),
            self._field("k sparse", "settings-query-k-sparse"),
            self._field("RRF dense weight", "settings-query-rrf-dense-weight"),
            self._field("RRF sparse weight", "settings-query-rrf-sparse-weight"),
            self._field("Path contains (comma-separated)", "settings-query-path-contains"),
            self._field("Ext allow (comma-separated)", "settings-query-ext-allow"),
            self._field("Ext deny (comma-separated)", "settings-query-ext-deny"),
            self._select_field(
                "Chat mode",
                "settings-query-chat-mode",
                [("single", "single"), ("chat", "chat")],
            ),
            self._field("Query cache TTL", "settings-query-cache-ttl"),
            self._field("Query cache size", "settings-query-cache-size"),
            self._field("Enable query cache", "settings-query-enable-cache", field_type="switch"),
            self._field(
                "Enable LLM server cache",
                "settings-query-enable-llm-server-cache",
                field_type="switch",
            ),
            self._save_reset_row("query"),
            classes="settings-section",
        )

    def refresh_fields(self, config: Any) -> None:
        query = getattr(config, "query", None)
        if query is None:
            return

        modes = getattr(config, "modes", None) or {}
        self._set_select_options("settings-query-mode", list(modes.keys()))
        self._set_select_value("settings-query-mode", str(getattr(query, "mode", "")))
        self._set_input("settings-query-k", str(getattr(query, "k", "")))
        self._set_input(
            "settings-query-retrieve-k",
            self._format_optional(getattr(query, "retrieve_k", None)),
        )
        self._set_input("settings-query-min-relevance", str(getattr(query, "min_relevance", "")))
        self._set_select_value(
            "settings-query-retrieval-profile",
            str(getattr(query, "retrieval_profile", "hybrid_rrf")),
        )
        self._set_input("settings-query-k-dense", str(getattr(query, "k_dense", 100)))
        self._set_input("settings-query-k-sparse", str(getattr(query, "k_sparse", 100)))
        self._set_input(
            "settings-query-rrf-dense-weight", str(getattr(query, "rrf_dense_weight", 0.7))
        )
        self._set_input(
            "settings-query-rrf-sparse-weight", str(getattr(query, "rrf_sparse_weight", 0.3))
        )
        self._set_input(
            "settings-query-path-contains",
            ", ".join(getattr(query, "path_contains", None) or []),
        )
        self._set_input(
            "settings-query-ext-allow",
            ", ".join(getattr(query, "ext_allow", None) or []),
        )
        self._set_input(
            "settings-query-ext-deny",
            ", ".join(getattr(query, "ext_deny", None) or []),
        )
        self._set_select_value("settings-query-chat-mode", str(getattr(query, "chat_mode", "")))
        self._set_input("settings-query-cache-ttl", str(getattr(query, "query_cache_ttl", "")))
        self._set_input("settings-query-cache-size", str(getattr(query, "query_cache_size", "")))
        self._set_switch(
            "settings-query-enable-cache",
            bool(getattr(query, "enable_query_cache", False)),
        )
        self._set_switch(
            "settings-query-enable-llm-server-cache",
            bool(getattr(query, "enable_llm_server_cache", False)),
        )

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        query = getattr(config, "query", None)
        if query is None:
            return False

        text = str(value or "").strip()
        if field_id == "settings-query-mode":
            query.mode = text
            return True
        if field_id == "settings-query-k" and text:
            query.k = int(text)
            return True
        if field_id == "settings-query-retrieve-k":
            query.retrieve_k = int(text) if text else None
            return True
        if field_id == "settings-query-min-relevance" and text:
            query.min_relevance = float(text)
            return True
        if field_id == "settings-query-retrieval-profile":
            query.retrieval_profile = text
            return True
        if field_id == "settings-query-k-dense" and text:
            query.k_dense = int(text)
            return True
        if field_id == "settings-query-k-sparse" and text:
            query.k_sparse = int(text)
            return True
        if field_id == "settings-query-rrf-dense-weight" and text:
            query.rrf_dense_weight = float(text)
            return True
        if field_id == "settings-query-rrf-sparse-weight" and text:
            query.rrf_sparse_weight = float(text)
            return True
        if field_id == "settings-query-path-contains":
            query.path_contains = self._parse_csv(text)
            return True
        if field_id == "settings-query-ext-allow":
            query.ext_allow = self._parse_csv(text)
            return True
        if field_id == "settings-query-ext-deny":
            query.ext_deny = self._parse_csv(text)
            return True
        if field_id == "settings-query-chat-mode":
            query.chat_mode = text
            return True
        if field_id == "settings-query-cache-ttl" and text:
            query.query_cache_ttl = int(text)
            return True
        if field_id == "settings-query-cache-size" and text:
            query.query_cache_size = int(text)
            return True
        if field_id == "settings-query-enable-cache":
            query.enable_query_cache = bool(value)
            return True
        if field_id == "settings-query-enable-llm-server-cache":
            query.enable_llm_server_cache = bool(value)
            return True

        return False

    @staticmethod
    def _parse_csv(text: str) -> Optional[list[str]]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        return values or None

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)
