from __future__ import annotations

from typing import Any

from textual.containers import Horizontal
from textual.widgets import Button, Input, Select, Switch

from lsm.ui.tui.widgets.settings_base import (
    BaseSettingsTab,
    _field,
    _replace_container_children,
    _save_reset_row,
    _select_field,
    _set_input,
    _set_select_options,
    _set_select_value,
    _set_switch,
)


class _InputWidget:
    def __init__(self) -> None:
        self.value = ""


class _SwitchWidget:
    def __init__(self) -> None:
        self.value = False


class _SelectWidget:
    def __init__(self) -> None:
        self.value = ""
        self.options: list[tuple[str, str]] = []

    def set_options(self, options: list[tuple[str, str]]) -> None:
        self.options = options


class _ContainerWidget:
    def __init__(self) -> None:
        self.children: list[Any] = ["existing"]

    def remove_children(self) -> None:
        self.children = []

    def mount(self, widget: Any) -> None:
        self.children.append(widget)


class _Owner:
    def __init__(self) -> None:
        self.widgets: dict[str, Any] = {}

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        key = selector if isinstance(selector, str) else selector
        if key in self.widgets:
            return self.widgets[key]
        raise KeyError(key)


class _Controller:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.parent = None

    def _set_status(self, message: str, error: bool) -> None:
        self.calls.append((message, error))


class _TestSettingsTab(BaseSettingsTab):
    def __init__(self) -> None:
        super().__init__()
        self.widgets: dict[str, Any] = {}
        self.posted_messages: list[Any] = []
        self.refresh_calls = 0

    def refresh_fields(self, config: Any) -> None:
        self.refresh_calls += 1
        if config.get("reentrant"):
            self.guarded_refresh_fields({"reentrant": False})

    def apply_update(self, field_id: str, value: Any, config: Any) -> None:
        return None

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        key = selector if isinstance(selector, str) else selector
        if key in self.widgets:
            return self.widgets[key]
        raise KeyError(key)

    def post_message(self, message):  # type: ignore[override]
        self.posted_messages.append(message)


def test_field_and_select_helpers_create_expected_rows() -> None:
    input_row = _field("Name", "settings-name")
    switch_row = _field("Enabled", "settings-enabled", field_type="switch", value=True, disabled=True)
    select_row = _select_field("Mode", "settings-mode", [("grounded", "grounded")])

    assert isinstance(input_row, Horizontal)
    assert isinstance(switch_row, Horizontal)
    assert isinstance(select_row, Horizontal)

    input_children = getattr(input_row, "_pending_children", [])
    switch_children = getattr(switch_row, "_pending_children", [])
    select_children = getattr(select_row, "_pending_children", [])

    assert isinstance(input_children[1], Input)
    assert input_children[1].id == "settings-name"

    assert isinstance(switch_children[1], Switch)
    assert switch_children[1].id == "settings-enabled"
    assert switch_children[1].value is True
    assert switch_children[1].disabled is True

    assert isinstance(select_children[1], Select)
    assert select_children[1].id == "settings-mode"


def test_setter_helpers_and_container_replacement() -> None:
    owner = _Owner()
    input_widget = _InputWidget()
    switch_widget = _SwitchWidget()
    select_widget = _SelectWidget()
    container = _ContainerWidget()

    owner.widgets["#settings-text"] = input_widget
    owner.widgets["#settings-switch"] = switch_widget
    owner.widgets["#settings-select"] = select_widget
    owner.widgets["#settings-container"] = container

    _set_input(owner, "settings-text", "hello")
    _set_switch(owner, "settings-switch", True)
    _set_select_options(owner, "settings-select", ["a", "b"])
    _set_select_value(owner, "settings-select", "b")
    _replace_container_children(owner, "#settings-container", ["x", "y"])

    assert input_widget.value == "hello"
    assert switch_widget.value is True
    assert select_widget.options == [("a", "a"), ("b", "b")]
    assert select_widget.value == "b"
    assert container.children == ["x", "y"]

    _set_input(owner, "missing-text", "ignored")
    _set_switch(owner, "missing-switch", False)
    _set_select_options(owner, "missing-select", ["x"])
    _set_select_value(owner, "missing-select", "x")
    _replace_container_children(owner, "#missing-container", ["x"])


def test_save_reset_row_contains_expected_buttons() -> None:
    row = _save_reset_row("global")
    children = getattr(row, "_pending_children", [])

    assert isinstance(row, Horizontal)
    assert isinstance(children[0], Button)
    assert isinstance(children[1], Button)
    assert children[0].id == "settings-save-global"
    assert children[1].id == "settings-reset-global"


def test_base_tab_guarded_refresh_prevents_reentrant_refresh() -> None:
    tab = _TestSettingsTab()
    tab.guarded_refresh_fields({"reentrant": True})

    assert tab.refresh_calls == 1
    assert tab._is_refreshing is False


def test_base_tab_post_status_prefers_parent_controller() -> None:
    tab = _TestSettingsTab()
    controller = _Controller()
    tab._parent = controller

    tab.post_status("Invalid field", True)
    assert controller.calls == [("Invalid field", True)]
    assert tab.posted_messages == []

    tab._parent = None
    tab.post_status("Fallback", False)
    assert len(tab.posted_messages) == 1
    assert isinstance(tab.posted_messages[0], BaseSettingsTab.StatusUpdate)
    assert tab.posted_messages[0].message == "Fallback"
    assert tab.posted_messages[0].error is False
