"""Global settings tab widget."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Static

from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class GlobalSettingsTab(BaseSettingsTab):
    """Settings view for global configuration fields."""

    def compose(self) -> ComposeResult:
        yield Container(
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

    def refresh_fields(self, config: Any) -> None:
        global_settings = getattr(config, "global_settings", None)
        if global_settings is None:
            return

        self._set_input("settings-config-path", str(getattr(config, "config_path", "") or ""))
        self._set_input(
            "settings-global-folder",
            self._format_optional(getattr(global_settings, "global_folder", None)),
        )
        self._set_input("settings-global-embed-model", str(getattr(global_settings, "embed_model", "")))
        self._set_input("settings-global-device", str(getattr(global_settings, "device", "")))
        self._set_input("settings-global-batch-size", str(getattr(global_settings, "batch_size", "")))
        self._set_input(
            "settings-global-embedding-dimension",
            self._format_optional(getattr(global_settings, "embedding_dimension", None)),
        )

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        global_settings = getattr(config, "global_settings", None)
        if global_settings is None:
            return False

        text = str(value or "").strip()
        if field_id == "settings-global-folder":
            global_settings.global_folder = Path(text) if text else None
            return True
        if field_id == "settings-global-embed-model":
            global_settings.embed_model = text
            return True
        if field_id == "settings-global-device":
            global_settings.device = text
            return True
        if field_id == "settings-global-batch-size" and text:
            global_settings.batch_size = int(text)
            return True
        if field_id == "settings-global-embedding-dimension":
            global_settings.embedding_dimension = int(text) if text else None
            return True
        return False

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)
