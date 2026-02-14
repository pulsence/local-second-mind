"""Chats and notes settings tab widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class ChatsNotesSettingsTab(BaseSettingsTab):
    """Settings view for chats and notes configuration fields."""

    def compose(self) -> ComposeResult:
        yield Container(
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

    def refresh_fields(self, config: Any) -> None:
        chats = getattr(config, "chats", None)
        notes = getattr(config, "notes", None)

        if chats is not None:
            self._set_switch("settings-chats-enabled", bool(getattr(chats, "enabled", False)))
            self._set_input("settings-chats-dir", str(getattr(chats, "dir", "")))
            self._set_switch("settings-chats-auto-save", bool(getattr(chats, "auto_save", False)))
            self._set_input("settings-chats-format", str(getattr(chats, "format", "")))

        if notes is not None:
            self._set_switch("settings-notes-enabled", bool(getattr(notes, "enabled", False)))
            self._set_input("settings-notes-dir", str(getattr(notes, "dir", "")))
            self._set_input("settings-notes-template", str(getattr(notes, "template", "")))
            self._set_input(
                "settings-notes-filename-format",
                str(getattr(notes, "filename_format", "")),
            )
            self._set_input("settings-notes-integration", str(getattr(notes, "integration", "")))
            self._set_switch("settings-notes-wikilinks", bool(getattr(notes, "wikilinks", False)))
            self._set_switch("settings-notes-backlinks", bool(getattr(notes, "backlinks", False)))
            self._set_switch(
                "settings-notes-include-tags",
                bool(getattr(notes, "include_tags", False)),
            )

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        chats = getattr(config, "chats", None)
        notes = getattr(config, "notes", None)
        text = str(value or "").strip()

        if field_id == "settings-chats-enabled" and chats is not None:
            chats.enabled = bool(value)
            return True
        if field_id == "settings-chats-dir" and chats is not None:
            chats.dir = text or "Chats"
            return True
        if field_id == "settings-chats-auto-save" and chats is not None:
            chats.auto_save = bool(value)
            return True
        if field_id == "settings-chats-format" and chats is not None:
            chats.format = text or "markdown"
            return True

        if field_id == "settings-notes-enabled" and notes is not None:
            notes.enabled = bool(value)
            return True
        if field_id == "settings-notes-dir" and notes is not None:
            notes.dir = text or "notes"
            return True
        if field_id == "settings-notes-template" and notes is not None:
            notes.template = text or "default"
            return True
        if field_id == "settings-notes-filename-format" and notes is not None:
            notes.filename_format = text or "timestamp"
            return True
        if field_id == "settings-notes-integration" and notes is not None:
            notes.integration = text or "none"
            return True
        if field_id == "settings-notes-wikilinks" and notes is not None:
            notes.wikilinks = bool(value)
            return True
        if field_id == "settings-notes-backlinks" and notes is not None:
            notes.backlinks = bool(value)
            return True
        if field_id == "settings-notes-include-tags" and notes is not None:
            notes.include_tags = bool(value)
            return True

        return False
