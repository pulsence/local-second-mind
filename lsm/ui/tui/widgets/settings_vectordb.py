"""Vector database settings tab widget."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static

from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class VectorDBSettingsTab(BaseSettingsTab):
    """Settings view for vector database configuration fields."""

    _SQLITE_FIELDS = (
        "settings-vdb-path",
    )

    _POSTGRES_FIELDS = (
        "settings-vdb-connection-string",
        "settings-vdb-host",
        "settings-vdb-port",
        "settings-vdb-database",
        "settings-vdb-user",
        "settings-vdb-password",
        "settings-vdb-index-type",
        "settings-vdb-pool-size",
    )

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Vector DB", classes="settings-section-title"),
            self._select_field(
                "Provider",
                "settings-vdb-provider",
                [("sqlite", "sqlite"), ("postgresql", "postgresql")],
            ),
            self._field("Collection", "settings-vdb-collection"),
            self._field("Path", "settings-vdb-path"),
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

    def refresh_fields(self, config: Any) -> None:
        vdb = getattr(config, "vectordb", None)
        if vdb is None:
            return

        provider = str(getattr(vdb, "provider", ""))
        self._set_select_value("settings-vdb-provider", provider)
        self._set_input("settings-vdb-collection", str(getattr(vdb, "collection", "")))
        self._set_input("settings-vdb-path", str(getattr(vdb, "path", "")))
        self._set_input(
            "settings-vdb-connection-string",
            self._format_optional(getattr(vdb, "connection_string", None)),
        )
        self._set_input("settings-vdb-host", self._format_optional(getattr(vdb, "host", None)))
        self._set_input("settings-vdb-port", self._format_optional(getattr(vdb, "port", None)))
        self._set_input("settings-vdb-database", self._format_optional(getattr(vdb, "database", None)))
        self._set_input("settings-vdb-user", self._format_optional(getattr(vdb, "user", None)))
        self._set_input("settings-vdb-password", "")
        self._set_input("settings-vdb-index-type", str(getattr(vdb, "index_type", "")))
        self._set_input("settings-vdb-pool-size", str(getattr(vdb, "pool_size", "")))

        self._update_provider_field_visibility(provider)

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        vdb = getattr(config, "vectordb", None)
        if vdb is None:
            return False

        text = str(value or "").strip()
        if field_id == "settings-vdb-provider":
            vdb.provider = text
            self._update_provider_field_visibility(text)
            return True
        if field_id == "settings-vdb-collection":
            vdb.collection = text
            return True
        if field_id == "settings-vdb-path":
            vdb.path = Path(text)
            return True
        if field_id == "settings-vdb-connection-string":
            vdb.connection_string = text or None
            return True
        if field_id == "settings-vdb-host":
            vdb.host = text or None
            return True
        if field_id == "settings-vdb-port":
            vdb.port = int(text) if text else None
            return True
        if field_id == "settings-vdb-database":
            vdb.database = text or None
            return True
        if field_id == "settings-vdb-user":
            vdb.user = text or None
            return True
        if field_id == "settings-vdb-password":
            if text:
                vdb.password = text
            return True
        if field_id == "settings-vdb-index-type":
            vdb.index_type = text or "hnsw"
            return True
        if field_id == "settings-vdb-pool-size" and text:
            vdb.pool_size = int(text)
            return True
        return False

    def _update_provider_field_visibility(self, provider: str) -> None:
        provider_name = (provider or "").strip().lower()
        if provider_name == "sqlite":
            self._set_fields_visible(self._SQLITE_FIELDS, True)
            self._set_fields_visible(self._POSTGRES_FIELDS, False)
        elif provider_name == "postgresql":
            self._set_fields_visible(self._SQLITE_FIELDS, False)
            self._set_fields_visible(self._POSTGRES_FIELDS, True)
        else:
            self._set_fields_visible(self._SQLITE_FIELDS, True)
            self._set_fields_visible(self._POSTGRES_FIELDS, True)

    def _set_fields_visible(self, field_ids: tuple[str, ...], visible: bool) -> None:
        for field_id in field_ids:
            try:
                field_widget = self.query_one(f"#{field_id}")
            except Exception:
                continue

            row_widget = getattr(field_widget, "parent", None)
            target = row_widget if row_widget is not None else field_widget
            try:
                target.display = visible
            except Exception:
                continue

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)
