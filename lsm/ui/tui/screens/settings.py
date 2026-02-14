"""Settings screen controller for LSM TUI."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch, TabbedContent, TabPane

from lsm.config.loader import load_config_from_file, save_config_to_file
from lsm.logging import get_logger
from lsm.ui.tui.widgets.settings_base import BaseSettingsTab
from lsm.ui.tui.widgets.settings_chats_notes import ChatsNotesSettingsTab
from lsm.ui.tui.widgets.settings_global import GlobalSettingsTab
from lsm.ui.tui.widgets.settings_ingest import IngestSettingsTab
from lsm.ui.tui.widgets.settings_llm import LLMSettingsTab
from lsm.ui.tui.widgets.settings_modes import ModesSettingsTab
from lsm.ui.tui.widgets.settings_query import QuerySettingsTab
from lsm.ui.tui.widgets.settings_remote import RemoteSettingsTab
from lsm.ui.tui.widgets.settings_vectordb import VectorDBSettingsTab

logger = get_logger(__name__)


class SettingsScreen(Widget):
    """Controller for the multi-tab settings editor."""

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

    _TAB_LAYOUT: tuple[tuple[str, str], ...] = (
        ("settings-global", "Global (^O)"),
        ("settings-ingest", "Ingest (^G)"),
        ("settings-query", "Query (^Q)"),
        ("settings-llm", "LLM (^L)"),
        ("settings-vdb", "Vector DB (^B)"),
        ("settings-modes", "Modes (^D)"),
        ("settings-remote", "Remote (^R)"),
        ("settings-chats-notes", "Chats/Notes (^N)"),
    )

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._is_refreshing = False
        self._tabs: dict[str, BaseSettingsTab] = {
            "settings-global": GlobalSettingsTab(id="settings-tab-global"),
            "settings-ingest": IngestSettingsTab(id="settings-tab-ingest"),
            "settings-query": QuerySettingsTab(id="settings-tab-query"),
            "settings-llm": LLMSettingsTab(id="settings-tab-llm"),
            "settings-vdb": VectorDBSettingsTab(id="settings-tab-vdb"),
            "settings-modes": ModesSettingsTab(id="settings-tab-modes"),
            "settings-remote": RemoteSettingsTab(id="settings-tab-remote"),
            "settings-chats-notes": ChatsNotesSettingsTab(id="settings-tab-chats-notes"),
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-layout"):
            yield Static("", id="settings-status")
            with TabbedContent(id="settings-tabs", initial="settings-global"):
                for tab_id, title in self._TAB_LAYOUT:
                    with TabPane(title, id=tab_id):
                        with ScrollableContainer(classes="settings-scroll"):
                            yield self._tabs[tab_id]

    def on_mount(self) -> None:
        self._is_refreshing = False
        if getattr(self.app, "current_context", None) == "settings":
            self.refresh_from_config()
            try:
                tabs = self.query_one("#settings-tabs", TabbedContent)
                self.call_after_refresh(tabs.focus)
            except Exception:
                return

    def on_show(self) -> None:
        self.refresh_from_config()

    def refresh_from_config(self) -> None:
        """Refresh all settings tabs from the in-memory config object."""
        config = getattr(self.app, "config", None)
        if config is None:
            return
        if self._is_refreshing:
            return

        self._is_refreshing = True
        try:
            self._refresh_tabs()
            self.current_mode = str(getattr(getattr(config, "query", None), "mode", "") or "")
        except Exception as exc:
            self._set_status(f"Failed to refresh settings: {exc}", True)
        finally:
            def _clear_refresh_flag() -> None:
                self._is_refreshing = False

            try:
                self.call_after_refresh(_clear_refresh_flag)
            except Exception:
                _clear_refresh_flag()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        return

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._is_refreshing:
            return
        self._apply_update(event.input.id or "", event.value, event.input)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if self._is_refreshing:
            return
        self._apply_update(event.switch.id or "", bool(event.value), event.switch)

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._is_refreshing:
            return
        field_id = event.select.id or ""
        value = str(event.value) if event.value is not None else ""
        self._apply_update(field_id, value, event.select)

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
            return

        self._route_button(button_id, event.button)

    def _apply_update(self, field_id: str, value: Any, source: Optional[Widget] = None) -> bool:
        config = getattr(self.app, "config", None)
        if config is None or not field_id:
            return False

        tab = self._resolve_tab(source, field_id)
        if tab is None:
            return False

        try:
            handled = bool(tab.apply_update(field_id, value, config))
        except Exception as exc:
            self._set_status(f"Invalid value for {field_id}: {exc}", True)
            return True

        if handled:
            self._handle_cross_tab_sync(field_id)
        return handled

    def _route_button(self, button_id: str, source: Optional[Widget] = None) -> bool:
        config = getattr(self.app, "config", None)
        if config is None or not button_id:
            return False

        tab = self._resolve_tab(source, button_id)
        if tab is None:
            return False

        try:
            handled = bool(tab.handle_button(button_id, config))
        except Exception as exc:
            self._set_status(str(exc), True)
            return True

        if handled and button_id.startswith("settings-remote-provider-"):
            self._refresh_tabs(["settings-modes"])
        return handled

    def _resolve_tab(self, source: Optional[Widget], field_or_button_id: str) -> Optional[BaseSettingsTab]:
        tab = self._tab_for_widget(source)
        if tab is not None:
            return tab
        tab_id = self._tab_id_for_field(field_or_button_id)
        if tab_id is None:
            return None
        return self._tabs.get(tab_id)

    def _tab_for_widget(self, widget: Optional[Widget]) -> Optional[BaseSettingsTab]:
        node = widget
        while node is not None:
            if isinstance(node, BaseSettingsTab):
                return node
            node = getattr(node, "parent", None)
        return None

    def _tab_id_for_field(self, field_id: str) -> Optional[str]:
        for tab_id, prefixes in self._TAB_FIELD_PREFIXES:
            for prefix in prefixes:
                if field_id.startswith(prefix):
                    return tab_id
        return None

    def _handle_cross_tab_sync(self, field_id: str) -> None:
        if field_id in {"settings-query-mode", "settings-modes-mode"}:
            self._sync_mode_tabs()
            return
        if field_id.startswith("settings-remote-provider-") and field_id.endswith("-name"):
            self._refresh_tabs(["settings-modes"])

    def _sync_mode_tabs(self) -> None:
        config = getattr(self.app, "config", None)
        if config is None:
            return
        self.current_mode = str(getattr(getattr(config, "query", None), "mode", "") or "")
        self._refresh_tabs(["settings-query", "settings-modes"])

    def _refresh_tabs(self, tab_ids: Optional[Sequence[str]] = None) -> None:
        config = getattr(self.app, "config", None)
        if config is None:
            return

        targets = tab_ids or tuple(self._tabs.keys())
        for tab_id in targets:
            tab = self._tabs.get(tab_id)
            if tab is not None:
                tab.guarded_refresh_fields(config)

    # Backward-compatible wrappers used by existing tests and transition code.
    def _refresh_settings(self) -> None:
        self.refresh_from_config()

    def _apply_live_update(self, field_id: str, value: str) -> None:
        self._apply_update(field_id, value)

    def _apply_live_switch_update(self, field_id: str, value: bool) -> None:
        self._apply_update(field_id, value)

    def _apply_dynamic_text_update(self, field_id: str, value: str) -> bool:
        return self._apply_update(field_id, value)

    def _apply_dynamic_switch_update(self, field_id: str, value: bool) -> bool:
        return self._apply_update(field_id, value)

    def _handle_structured_button(self, button_id: str) -> bool:
        return self._route_button(button_id)

    def _save_config(self) -> None:
        cfg = getattr(self.app, "config", None)
        if cfg and getattr(cfg, "config_path", None):
            save_config_to_file(cfg, cfg.config_path)

    def _reset_config(self) -> None:
        cfg = getattr(self.app, "config", None)
        if not cfg or not getattr(cfg, "config_path", None):
            self._set_status("No config path available for reset.", True)
            return
        self.app.config = load_config_from_file(cfg.config_path)
        self.refresh_from_config()
        self._set_status("Configuration reloaded from disk.", False)

    def _activate_tab(self, tab_id: str) -> None:
        self.query_one("#settings-tabs", TabbedContent).active = tab_id

    def action_settings_tab_1(self) -> None:
        self._activate_tab("settings-global")

    def action_settings_tab_2(self) -> None:
        self._activate_tab("settings-ingest")

    def action_settings_tab_3(self) -> None:
        self._activate_tab("settings-query")

    def action_settings_tab_4(self) -> None:
        self._activate_tab("settings-llm")

    def action_settings_tab_5(self) -> None:
        self._activate_tab("settings-vdb")

    def action_settings_tab_6(self) -> None:
        self._activate_tab("settings-modes")

    def action_settings_tab_7(self) -> None:
        self._activate_tab("settings-remote")

    def action_settings_tab_8(self) -> None:
        self._activate_tab("settings-chats-notes")

    def _set_status(self, message: str, error: bool) -> None:
        try:
            widget = self.query_one("#settings-status", Static)
            widget.update(f"[red]{message}[/red]" if error else message)
        except Exception:
            logger.warning(message)
