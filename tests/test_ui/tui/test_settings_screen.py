from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from lsm.ui.tui.screens.settings import SettingsScreen
from lsm.ui.tui.state import SettingTableRow, SettingsActionResult


class _Tabs:
    def __init__(self) -> None:
        self.id = "settings-tabs"
        self.active = "settings-global"
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _StaticWidget:
    def __init__(self) -> None:
        self.last = ""

    def update(self, value: str) -> None:
        self.last = value


class _Table:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.rows: list[tuple[str, str, str]] = []
        self.cursor_type = ""
        self.zebra_stripes = False

    def add_columns(self, *columns: str) -> None:
        self.columns.extend(columns)

    def add_row(self, key: str, value: str, state: str) -> None:
        self.rows.append((key, value, state))

    def clear(self) -> None:
        self.rows = []


class _CommandInput:
    def __init__(self) -> None:
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _FakeViewModel:
    def __init__(self, config: Any) -> None:
        self.persisted_config = config
        self.draft_config = config
        self.table_rows_calls: list[str] = []
        self.set_calls: list[tuple[str, str, str]] = []
        self.unset_calls: list[tuple[str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.reset_key_calls: list[tuple[str, str]] = []
        self.default_key_calls: list[tuple[str, str]] = []
        self.reset_tab_calls: list[str] = []
        self.save_calls = 0
        self.load_calls: list[Any] = []

        self.next_result = SettingsActionResult(handled=True, changed_tabs=("settings-global",))
        self.next_save_result = SettingsActionResult(handled=True, changed_tabs=("settings-global",))

    def table_rows(self, tab_id: str) -> list[SettingTableRow]:
        self.table_rows_calls.append(tab_id)
        return [SettingTableRow(key="sample", value=tab_id, state="")]

    def load(self, config: Any) -> SettingsActionResult:
        self.load_calls.append(config)
        self.persisted_config = config
        self.draft_config = config
        return SettingsActionResult(handled=True, changed_tabs=("settings-global",))

    def set_key(self, tab_id: str, key: str, value: str) -> SettingsActionResult:
        self.set_calls.append((tab_id, key, value))
        return self.next_result

    def unset_key(self, tab_id: str, key: str) -> SettingsActionResult:
        self.unset_calls.append((tab_id, key))
        return self.next_result

    def delete_key(self, tab_id: str, key: str) -> SettingsActionResult:
        self.delete_calls.append((tab_id, key))
        return self.next_result

    def reset_key(self, tab_id: str, key: str) -> SettingsActionResult:
        self.reset_key_calls.append((tab_id, key))
        return self.next_result

    def default_key(self, tab_id: str, key: str) -> SettingsActionResult:
        self.default_key_calls.append((tab_id, key))
        return self.next_result

    def reset_tab(self, tab_id: str) -> SettingsActionResult:
        self.reset_tab_calls.append(tab_id)
        return self.next_result

    def save(self) -> SettingsActionResult:
        self.save_calls += 1
        return self.next_save_result


class _TestableSettingsScreen(SettingsScreen):
    def __init__(self, app):
        super().__init__()
        self._test_app = app
        self.widgets: dict[str, Any] = {}

    @property
    def app(self):  # type: ignore[override]
        return self._test_app

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        key = selector if isinstance(selector, str) else selector
        if key in self.widgets:
            return self.widgets[key]
        raise KeyError(key)

    def call_after_refresh(self, fn):  # type: ignore[override]
        fn()


def _config() -> Any:
    return SimpleNamespace(
        config_path=Path("config.json"),
        query=SimpleNamespace(mode="grounded"),
    )


def _screen(context: str = "settings") -> tuple[_TestableSettingsScreen, _FakeViewModel]:
    app = SimpleNamespace(config=_config(), current_context=context)
    screen = _TestableSettingsScreen(app)
    vm = _FakeViewModel(app.config)
    screen._view_model = vm
    screen._source_config = app.config

    screen.widgets["#settings-status"] = _StaticWidget()
    screen.widgets["#settings-tabs"] = _Tabs()
    for tab_id, _ in screen._TAB_LAYOUT:
        screen.widgets[f"#{tab_id}-table"] = _Table()
        screen.widgets[f"#{tab_id}-command"] = _CommandInput()

    return screen, vm


def test_settings_bindings_use_function_keys_and_avoid_app_conflicts() -> None:
    from lsm.ui.tui.app import LSMApp

    keys = [binding.key for binding in SettingsScreen.BINDINGS]
    assert keys == ["f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]

    app_keys = {binding.key for binding in LSMApp.BINDINGS}
    assert app_keys.isdisjoint(keys)


def test_refresh_from_config_populates_only_active_table() -> None:
    screen, vm = _screen()

    screen.refresh_from_config()

    assert vm.table_rows_calls == ["settings-global"]
    assert screen.widgets["#settings-global-table"].rows
    assert screen.widgets["#settings-query-table"].rows == []


def test_tab_activation_refreshes_stale_selected_tab() -> None:
    screen, vm = _screen()
    tabs = screen.widgets["#settings-tabs"]

    tabs.active = "settings-llm"
    event = SimpleNamespace(tabbed_content=tabs, tab=SimpleNamespace(id="settings-llm-tab"))

    screen.on_tabbed_content_tab_activated(event)

    assert vm.table_rows_calls == ["settings-llm"]
    assert screen.widgets["#settings-llm-table"].rows


def test_tab_activation_focuses_active_command_input() -> None:
    screen, _ = _screen()
    tabs = screen.widgets["#settings-tabs"]

    tabs.active = "settings-query"
    event = SimpleNamespace(tabbed_content=tabs, tab=SimpleNamespace(id="settings-query-tab"))

    screen.on_tabbed_content_tab_activated(event)

    assert screen.widgets["#settings-query-command"].focused is True


def test_set_command_routes_to_view_model() -> None:
    screen, vm = _screen()

    event = SimpleNamespace(
        input=SimpleNamespace(id="settings-global-command", value="set global_folder /tmp"),
        value="set global_folder /tmp",
    )
    screen.on_input_submitted(event)

    assert vm.set_calls == [("settings-global", "global_folder", "/tmp")]
    assert event.input.value == ""


def test_save_command_updates_app_config() -> None:
    screen, vm = _screen()
    persisted = _config()
    persisted.query.mode = "insight"
    vm.persisted_config = persisted

    event = SimpleNamespace(
        input=SimpleNamespace(id="settings-query-command", value="save"),
        value="save",
    )
    screen.on_input_submitted(event)

    assert vm.save_calls == 1
    assert screen.app.config is persisted
    status = screen.widgets["#settings-status"].last
    assert "Configuration saved" in status


def test_reset_without_key_routes_to_tab_reset() -> None:
    screen, vm = _screen()

    event = SimpleNamespace(
        input=SimpleNamespace(id="settings-ingest-command", value="reset"),
        value="reset",
    )
    screen.on_input_submitted(event)

    assert vm.reset_tab_calls == ["settings-ingest"]


def test_invalid_command_sets_error_status() -> None:
    screen, _ = _screen()

    event = SimpleNamespace(
        input=SimpleNamespace(id="settings-global-command", value="wat"),
        value="wat",
    )
    screen.on_input_submitted(event)

    assert "Unknown command" in screen.widgets["#settings-status"].last


def test_command_error_is_reported() -> None:
    screen, vm = _screen()
    vm.next_result = SettingsActionResult(handled=True, changed_tabs=("settings-global",), error="bad key")

    event = SimpleNamespace(
        input=SimpleNamespace(id="settings-global-command", value="set nope x"),
        value="set nope x",
    )
    screen.on_input_submitted(event)

    assert "bad key" in screen.widgets["#settings-status"].last


def test_refresh_loads_external_config_object() -> None:
    screen, vm = _screen()
    new_config = _config()
    new_config.query.mode = "hybrid"
    screen.app.config = new_config

    screen.refresh_from_config()

    assert vm.load_calls == [new_config]
