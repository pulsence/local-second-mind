from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from lsm.ui.tui.screens.settings import SettingsScreen
from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class _Tabs:
    def __init__(self) -> None:
        self.active = ""
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _StaticWidget:
    def __init__(self) -> None:
        self.last = ""

    def update(self, value: str) -> None:
        self.last = value


class _FakeTab(BaseSettingsTab):
    def __init__(self, name: str) -> None:
        super().__init__(id=f"tab-{name}")
        self.tab_name = name
        self.refresh_calls = 0
        self.apply_calls: list[tuple[str, Any]] = []
        self.button_calls: list[str] = []
        self.next_apply_handled = True
        self.next_button_handled = True
        self.apply_error: Optional[Exception] = None
        self.button_error: Optional[Exception] = None
        self.apply_hook = None

    def refresh_fields(self, config: Any) -> None:
        self.refresh_calls += 1

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        self.apply_calls.append((field_id, value))
        if self.apply_error is not None:
            raise self.apply_error
        if callable(self.apply_hook):
            self.apply_hook(field_id, value, config)
        return self.next_apply_handled

    def handle_button(self, button_id: str, config: Any) -> bool:
        self.button_calls.append(button_id)
        if self.button_error is not None:
            raise self.button_error
        return self.next_button_handled


class _TestableSettingsScreen(SettingsScreen):
    def __init__(self, app):
        super().__init__()
        self._test_app = app
        self.widgets = {}

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


class _DeferredRefreshSettingsScreen(_TestableSettingsScreen):
    def __init__(self, app):
        super().__init__(app)
        self.after_refresh = None

    def call_after_refresh(self, fn):  # type: ignore[override]
        self.after_refresh = fn


def _config() -> Any:
    return SimpleNamespace(
        config_path=Path("config.json"),
        query=SimpleNamespace(mode="grounded"),
        ingest=SimpleNamespace(),
        llm=SimpleNamespace(),
        vectordb=SimpleNamespace(),
        modes={},
        remote_providers=[],
        remote_provider_chains=[],
        notes=SimpleNamespace(),
        chats=SimpleNamespace(),
    )


def _screen(context: str = "settings") -> _TestableSettingsScreen:
    app = SimpleNamespace(config=_config(), current_context=context)
    screen = _TestableSettingsScreen(app)
    screen._tabs = {
        "settings-global": _FakeTab("global"),
        "settings-ingest": _FakeTab("ingest"),
        "settings-query": _FakeTab("query"),
        "settings-llm": _FakeTab("llm"),
        "settings-vdb": _FakeTab("vdb"),
        "settings-modes": _FakeTab("modes"),
        "settings-remote": _FakeTab("remote"),
        "settings-chats-notes": _FakeTab("chats-notes"),
    }
    return screen


def test_focus_and_tab_actions() -> None:
    screen = _screen()
    tabs = _Tabs()
    screen.widgets["#settings-tabs"] = tabs

    screen.on_mount()
    assert tabs.focused is True

    screen.action_settings_tab_1()
    assert tabs.active == "settings-global"
    screen.action_settings_tab_8()
    assert tabs.active == "settings-chats-notes"


def test_refresh_from_config_delegates_to_all_tabs() -> None:
    screen = _screen()

    screen.refresh_from_config()

    for tab in screen._tabs.values():
        assert tab.refresh_calls == 1


def test_on_show_refreshes_values_even_if_context_is_stale() -> None:
    screen = _screen(context="query")

    screen.on_show()

    for tab in screen._tabs.values():
        assert tab.refresh_calls == 1


def test_refresh_keeps_guard_until_after_refresh_callback_runs() -> None:
    app = SimpleNamespace(config=_config(), current_context="settings")
    screen = _DeferredRefreshSettingsScreen(app)
    screen._tabs = {
        "settings-global": _FakeTab("global"),
        "settings-ingest": _FakeTab("ingest"),
        "settings-query": _FakeTab("query"),
        "settings-llm": _FakeTab("llm"),
        "settings-vdb": _FakeTab("vdb"),
        "settings-modes": _FakeTab("modes"),
        "settings-remote": _FakeTab("remote"),
        "settings-chats-notes": _FakeTab("chats-notes"),
    }

    screen.refresh_from_config()

    assert screen._is_refreshing is True
    assert callable(screen.after_refresh)

    screen.after_refresh()
    assert screen._is_refreshing is False


def test_input_update_routes_to_query_tab_by_field_prefix() -> None:
    screen = _screen()

    event = SimpleNamespace(input=SimpleNamespace(id="settings-query-k"), value="20")
    screen.on_input_changed(event)

    query_tab = screen._tabs["settings-query"]
    assert query_tab.apply_calls == [("settings-query-k", "20")]


def test_switch_update_routes_to_chats_notes_tab() -> None:
    screen = _screen()

    event = SimpleNamespace(switch=SimpleNamespace(id="settings-notes-enabled"), value=True)
    screen.on_switch_changed(event)

    chats_notes_tab = screen._tabs["settings-chats-notes"]
    assert chats_notes_tab.apply_calls == [("settings-notes-enabled", True)]


def test_select_update_routes_to_vectordb_tab() -> None:
    screen = _screen()

    event = SimpleNamespace(select=SimpleNamespace(id="settings-vdb-provider"), value="postgresql")
    screen.on_select_changed(event)

    vdb_tab = screen._tabs["settings-vdb"]
    assert vdb_tab.apply_calls == [("settings-vdb-provider", "postgresql")]


def test_mode_change_syncs_query_and_modes_tabs() -> None:
    screen = _screen()
    cfg = screen.app.config

    query_tab = screen._tabs["settings-query"]

    def _apply_mode(field_id: str, value: Any, config: Any) -> None:
        if field_id == "settings-query-mode":
            config.query.mode = str(value)

    query_tab.apply_hook = _apply_mode

    event = SimpleNamespace(select=SimpleNamespace(id="settings-query-mode"), value="insight")
    screen.on_select_changed(event)

    assert cfg.query.mode == "insight"
    assert screen.current_mode == "insight"
    assert screen._tabs["settings-query"].refresh_calls == 1
    assert screen._tabs["settings-modes"].refresh_calls == 1


def test_update_uses_source_tab_when_available() -> None:
    screen = _screen()
    source = SimpleNamespace(parent=screen._tabs["settings-remote"])

    handled = screen._apply_update("settings-query-k", "14", source)

    assert handled is True
    assert screen._tabs["settings-remote"].apply_calls == [("settings-query-k", "14")]
    assert screen._tabs["settings-query"].apply_calls == []


def test_save_and_reset_buttons(monkeypatch) -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    saved = {"called": False}
    reset = {"called": False}
    monkeypatch.setattr(screen, "_save_config", lambda: saved.__setitem__("called", True))
    monkeypatch.setattr(screen, "_reset_config", lambda: reset.__setitem__("called", True))

    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="settings-save-global")))
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="settings-reset-global")))

    assert saved["called"] is True
    assert reset["called"] is True


def test_button_routes_to_remote_tab_handler() -> None:
    screen = _screen()

    handled = screen._handle_structured_button("settings-remote-chain-add")

    assert handled is True
    assert screen._tabs["settings-remote"].button_calls == ["settings-remote-chain-add"]


def test_invalid_field_update_sets_error_status() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    query_tab = screen._tabs["settings-query"]
    query_tab.apply_error = ValueError("bad value")

    screen._apply_live_update("settings-query-k", "oops")

    assert "Invalid value for settings-query-k" in status.last


def test_button_handler_exception_sets_error_status() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    remote_tab = screen._tabs["settings-remote"]
    remote_tab.button_error = ValueError("cannot remove")

    handled = screen._handle_structured_button("settings-remote-provider-remove-0")

    assert handled is True
    assert "cannot remove" in status.last


def test_reset_config_reloads_from_disk_and_refreshes(monkeypatch) -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    new_config = _config()
    new_config.query.mode = "hybrid"

    import lsm.ui.tui.screens.settings as settings_module

    monkeypatch.setattr(settings_module, "load_config_from_file", lambda _path: new_config)

    screen._reset_config()

    assert screen.app.config.query.mode == "hybrid"
    assert screen._tabs["settings-global"].refresh_calls == 1
    assert "Configuration reloaded from disk" in status.last
