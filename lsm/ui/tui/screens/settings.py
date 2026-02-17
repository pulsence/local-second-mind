"""Command-table settings screen for LSM TUI."""

from __future__ import annotations

import shlex
from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, Input, Static, TabbedContent, TabPane

from lsm.logging import get_logger
from lsm.ui.tui.screens.base import ManagedScreenMixin
from lsm.ui.tui.state import SettingTableRow, SettingsActionResult, SettingsViewModel

logger = get_logger(__name__)


class SettingsScreen(ManagedScreenMixin, Widget):
    """Controller for the command-table settings editor."""

    current_mode: str = "grounded"
    """Current selected mode name (for tests and UI state)."""

    class UnsavedChangesBlocking(Message):
        """Posted when the user attempts to leave settings with unsaved changes.

        The app should show a confirmation prompt and call
        ``force_discard_and_leave()`` if the user confirms.
        """

        def __init__(self, target_context: str) -> None:
            super().__init__()
            self.target_context = target_context

    BINDINGS = [
        Binding("f2", "local_settings_tab_1", "Global", show=True),
        Binding("f3", "local_settings_tab_2", "Ingest", show=True),
        Binding("f4", "local_settings_tab_3", "Query", show=True),
        Binding("f5", "local_settings_tab_4", "LLM", show=True),
        Binding("f6", "local_settings_tab_5", "Vector DB", show=True),
        Binding("f7", "local_settings_tab_6", "Modes", show=True),
        Binding("f8", "local_settings_tab_7", "Remote", show=True),
        Binding("f9", "local_settings_tab_8", "Chats/Notes", show=True),
    ]

    _TAB_LAYOUT: tuple[tuple[str, str], ...] = (
        ("settings-global", "Global (F2)"),
        ("settings-ingest", "Ingest (F3)"),
        ("settings-query", "Query (F4)"),
        ("settings-llm", "LLM (F5)"),
        ("settings-vdb", "Vector DB (F6)"),
        ("settings-modes", "Modes (F7)"),
        ("settings-remote", "Remote (F8)"),
        ("settings-chats-notes", "Chats/Notes (F9)"),
    )

    _TAB_CLEAN_LABELS: dict[str, str] = {tab_id: title for tab_id, title in _TAB_LAYOUT}

    _TAB_HELP: dict[str, str] = {
        "settings-global": "Commands: set <key> <value> | unset <key> | delete <key> | reset <key> | default <key> | save | discard",
        "settings-ingest": "Commands: set roots[0].path <value> | set chunk_size 1200 | delete exclude_dirs | save | discard",
        "settings-query": "Commands: set mode insight | set k 12 | reset mode | default min_relevance | save | discard",
        "settings-llm": "Commands: set providers[0].provider_name openai | set services.query.model gpt-5.2 | save | discard",
        "settings-vdb": "Commands: set provider postgresql | set connection_string <value> | unset password | save | discard",
        "settings-modes": "Commands: set query.mode grounded | set modes.grounded.source_policy.remote.enabled true | save | discard",
        "settings-remote": "Commands: set remote_providers[0].name brave | delete remote_provider_chains[0].links[1] | save | discard",
        "settings-chats-notes": "Commands: set chats.enabled true | set notes.dir notes | default notes.template | save | discard",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._view_model: Optional[SettingsViewModel] = None
        self._source_config: Any = None
        self._stale_tabs: set[str] = {tab_id for tab_id, _ in self._TAB_LAYOUT}
        self._last_dirty_tabs: frozenset[str] = frozenset()

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-layout"):
            yield Static("", id="settings-status")
            with TabbedContent(id="settings-tabs", initial="settings-global"):
                for tab_id, title in self._TAB_LAYOUT:
                    with TabPane(title, id=tab_id):
                        with Vertical(classes="settings-tab-panel"):
                            yield DataTable(id=self._table_id(tab_id), classes="settings-kv-table")
                            yield Input(
                                placeholder="set <key> <value> | unset <key> | delete <key> | reset <key> | default <key> | save",
                                id=self._command_id(tab_id),
                                classes="settings-command-input",
                            )
                            yield Static(self._TAB_HELP[tab_id], classes="settings-command-help")

    def on_mount(self) -> None:
        for tab_id, _ in self._TAB_LAYOUT:
            self._configure_table(tab_id)

        if getattr(self.app, "current_context", None) == "settings":
            self.refresh_from_config()
            try:
                tabs = self.query_one("#settings-tabs", TabbedContent)
                self.call_after_refresh(tabs.focus)
                self.call_after_refresh(lambda: self._focus_command_input(self._active_tab_id()))
            except Exception:
                return

    def on_show(self) -> None:
        self.refresh_from_config()

    def on_unmount(self) -> None:
        self._cancel_managed_workers(reason="settings-unmount")
        self._cancel_managed_timers(reason="settings-unmount")

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        tabbed = getattr(event, "tabbed_content", None)
        if getattr(tabbed, "id", None) != "settings-tabs":
            return

        tab_id = self._normalize_tab_id(str(getattr(getattr(event, "tab", None), "id", "") or "").strip())
        if tab_id not in {tid for tid, _ in self._TAB_LAYOUT}:
            tab_id = self._active_tab_id()

        if tab_id in self._stale_tabs:
            self._refresh_tab_from_view_model(tab_id)
        self._focus_command_input(tab_id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        tab_id = self._tab_id_from_command_input_id(event.input.id or "")
        if tab_id is None:
            return

        command = (event.value or "").strip()
        event.input.value = ""
        if not command:
            return

        self._execute_command(tab_id, command)

    def refresh_from_config(self) -> None:
        """Refresh current active tab from settings view model state."""
        view_model = self._ensure_view_model()
        if view_model is None:
            return

        self.current_mode = str(getattr(view_model.draft_config.query, "mode", "") or "")
        self._mark_tabs_stale([tab_id for tab_id, _ in self._TAB_LAYOUT])
        active = self._active_tab_id()
        self._refresh_tab_from_view_model(active)
        self._focus_command_input(active)

    def _execute_command(self, tab_id: str, command: str) -> None:
        view_model = self._ensure_view_model()
        if view_model is None:
            self._set_status("No config available.", True)
            return

        try:
            tokens = shlex.split(command)
        except Exception as exc:
            self._set_status(f"Command parse error: {exc}", True)
            return

        if not tokens:
            return

        verb = tokens[0].lower()
        result: Optional[SettingsActionResult] = None

        if verb == "save":
            result = view_model.save()
            if result.error is None:
                self.app.config = view_model.persisted_config
                self._source_config = self.app.config
                self._set_status("Configuration saved.", False)
                self._notify_event("Configuration saved successfully.", severity="info")
            else:
                self._set_status(f"Save failed: {result.error}", True)
        elif verb == "set":
            if len(tokens) < 3:
                self._set_status("Usage: set <key> <value>", True)
                return
            key = tokens[1]
            value = " ".join(tokens[2:])
            result = view_model.set_key(tab_id, key, value)
            self._set_status(f"Set {key}", result.error is not None)
        elif verb == "unset":
            if len(tokens) != 2:
                self._set_status("Usage: unset <key>", True)
                return
            key = tokens[1]
            result = view_model.unset_key(tab_id, key)
            self._set_status(f"Unset {key}", result.error is not None)
        elif verb == "delete":
            if len(tokens) != 2:
                self._set_status("Usage: delete <key>", True)
                return
            key = tokens[1]
            result = view_model.delete_key(tab_id, key)
            self._set_status(f"Deleted {key}", result.error is not None)
        elif verb == "reset":
            if len(tokens) == 1:
                result = view_model.reset_tab(tab_id)
                self._set_status("Tab reset to persisted values.", result.error is not None)
            elif len(tokens) == 2:
                key = tokens[1]
                result = view_model.reset_key(tab_id, key)
                self._set_status(f"Reset {key}", result.error is not None)
            else:
                self._set_status("Usage: reset <key>", True)
                return
        elif verb == "default":
            if len(tokens) != 2:
                self._set_status("Usage: default <key>", True)
                return
            key = tokens[1]
            result = view_model.default_key(tab_id, key)
            self._set_status(f"Defaulted {key}", result.error is not None)
        elif verb == "discard":
            if len(tokens) == 1:
                result = view_model.reset_all()
                self._set_status("All unsaved changes discarded.", result.error is not None)
            elif len(tokens) == 2 and tokens[1] == "tab":
                result = view_model.reset_tab(tab_id)
                self._set_status("Tab changes discarded.", result.error is not None)
            else:
                self._set_status("Usage: discard | discard tab", True)
                return
        else:
            self._set_status(f"Unknown command: {verb}", True)
            return

        if result is None:
            return
        if result.error:
            self._set_status(result.error, True)

        self._apply_action_result(tab_id, result)

    def _apply_action_result(self, active_tab_id: str, result: SettingsActionResult) -> None:
        if not result.handled:
            return

        self._mark_tabs_stale(result.changed_tabs)
        self.current_mode = str(getattr(self._view_model.draft_config.query, "mode", "") or "") if self._view_model else self.current_mode

        if active_tab_id in self._stale_tabs:
            self._refresh_tab_from_view_model(active_tab_id)

        self._sync_dirty_indicators()

    def _ensure_view_model(self) -> Optional[SettingsViewModel]:
        config = getattr(self.app, "config", None)
        if config is None:
            return None

        if self._view_model is None:
            self._view_model = SettingsViewModel(config)
            self._source_config = config
            self._mark_tabs_stale([tab_id for tab_id, _ in self._TAB_LAYOUT])
            return self._view_model

        if config is not self._source_config:
            self._view_model.load(config)
            self._source_config = config
            self._mark_tabs_stale([tab_id for tab_id, _ in self._TAB_LAYOUT])

        return self._view_model

    def _refresh_tab_from_view_model(self, tab_id: str) -> None:
        view_model = self._view_model
        if view_model is None:
            return

        rows = view_model.table_rows(tab_id)
        table = self.query_one(f"#{self._table_id(tab_id)}", DataTable)
        table.clear()
        if not rows:
            table.add_row("(no settings)", "", "")
        else:
            for row in rows:
                table.add_row(row.key, row.value, row.state)

        self._stale_tabs.discard(tab_id)

    def _configure_table(self, tab_id: str) -> None:
        table = self.query_one(f"#{self._table_id(tab_id)}", DataTable)
        if table.columns:
            return
        table.add_columns("Key", "Value", "State")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def _mark_tabs_stale(self, tab_ids: list[str] | tuple[str, ...]) -> None:
        for tab_id in tab_ids:
            if tab_id in {tid for tid, _ in self._TAB_LAYOUT}:
                self._stale_tabs.add(tab_id)

    def _active_tab_id(self) -> str:
        try:
            tabs = self.query_one("#settings-tabs", TabbedContent)
            active = self._normalize_tab_id(str(getattr(tabs, "active", "") or "").strip())
            if active in {tab_id for tab_id, _ in self._TAB_LAYOUT}:
                return active
        except Exception:
            pass
        return "settings-global"

    @staticmethod
    def _normalize_tab_id(value: str) -> str:
        text = (value or "").strip()
        return text[:-4] if text.endswith("-tab") else text

    @staticmethod
    def _table_id(tab_id: str) -> str:
        return f"{tab_id}-table"

    @staticmethod
    def _command_id(tab_id: str) -> str:
        return f"{tab_id}-command"

    def _tab_id_from_command_input_id(self, input_id: str) -> Optional[str]:
        key = (input_id or "").strip()
        if not key.endswith("-command"):
            return None
        candidate = key[: -len("-command")]
        if candidate in {tab_id for tab_id, _ in self._TAB_LAYOUT}:
            return candidate
        return None

    def _activate_tab(self, tab_id: str) -> None:
        self.query_one("#settings-tabs", TabbedContent).active = tab_id
        self.call_after_refresh(lambda: self._focus_command_input(tab_id))

    def _focus_command_input(self, tab_id: str) -> None:
        try:
            command = self.query_one(f"#{self._command_id(tab_id)}", Input)
            command.focus()
        except Exception:
            return

    def action_local_settings_tab_1(self) -> None:
        self._activate_tab("settings-global")

    def action_local_settings_tab_2(self) -> None:
        self._activate_tab("settings-ingest")

    def action_local_settings_tab_3(self) -> None:
        self._activate_tab("settings-query")

    def action_local_settings_tab_4(self) -> None:
        self._activate_tab("settings-llm")

    def action_local_settings_tab_5(self) -> None:
        self._activate_tab("settings-vdb")

    def action_local_settings_tab_6(self) -> None:
        self._activate_tab("settings-modes")

    def action_local_settings_tab_7(self) -> None:
        self._activate_tab("settings-remote")

    def action_local_settings_tab_8(self) -> None:
        self._activate_tab("settings-chats-notes")

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

    def _notify_event(
        self,
        message: str,
        *,
        severity: str = "info",
        timeout: Optional[float] = None,
    ) -> None:
        notify = getattr(self.app, "notify_event", None)
        if not callable(notify):
            return
        try:
            kwargs: dict[str, Any] = {"severity": severity}
            if timeout is not None:
                kwargs["timeout"] = timeout
            notify(message, **kwargs)
        except Exception:
            logger.exception("Failed to emit settings notification")

    # ------------------------------------------------------------------
    # Dirty-state tracking and unsaved-change guards
    # ------------------------------------------------------------------

    @property
    def has_unsaved_changes(self) -> bool:
        """True when the draft config diverges from the persisted config."""
        if self._view_model is None:
            return False
        return bool(self._view_model.dirty_tabs)

    @property
    def dirty_tab_ids(self) -> frozenset[str]:
        """Tab ids that currently have unsaved edits."""
        if self._view_model is None:
            return frozenset()
        return self._view_model.dirty_tabs

    def force_discard_and_leave(self) -> None:
        """Discard all unsaved changes and reset dirty state."""
        if self._view_model is not None:
            self._view_model.reset_all()
        self._sync_dirty_indicators()

    def _sync_dirty_indicators(self) -> None:
        """Update tab labels and status area to reflect current dirty state."""
        dirty = self.dirty_tab_ids
        if dirty == self._last_dirty_tabs:
            return
        self._last_dirty_tabs = dirty

        # Update tab pane titles with dirty markers
        for tab_id, clean_title in self._TAB_CLEAN_LABELS.items():
            label = f"* {clean_title}" if tab_id in dirty else clean_title
            try:
                tabs = self.query_one("#settings-tabs", TabbedContent)
                tab_widget = tabs.get_tab(f"{tab_id}-tab")
                if tab_widget is not None:
                    tab_widget.label = label
            except Exception:
                continue

        # Update status area with aggregate dirty summary
        if dirty:
            count = len(dirty)
            noun = "tab has" if count == 1 else "tabs have"
            self._set_status(f"{count} {noun} unsaved changes. Use 'save' to persist or 'discard' to reset.", False)
        else:
            self._set_status("", False)
