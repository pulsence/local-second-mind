"""
Shared settings tab utilities for the LSM TUI.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static, Switch


def _field(
    label: str,
    field_id: str,
    *,
    placeholder: str = "",
    disabled: bool = False,
    field_type: str = "input",
    value: Any = "",
) -> Widget:
    """Create a labeled Input or Switch field row."""
    label_widget = Static(label, classes="settings-label")
    if field_type == "switch":
        field_widget: Widget = Switch(value=bool(value), id=field_id)
    else:
        field_widget = Input(value="" if value is None else str(value), placeholder=placeholder, id=field_id)
    field_widget.disabled = disabled
    return Horizontal(label_widget, field_widget, classes="settings-field")


def _select_field(
    label: str,
    field_id: str,
    options: Optional[list[tuple[str, str]]] = None,
) -> Widget:
    """Create a labeled Select field row."""
    return Horizontal(
        Static(label, classes="settings-label"),
        Select(options or [], id=field_id),
        classes="settings-field",
    )


def _set_input(owner: Widget, field_id: str, value: str) -> None:
    """Set an Input widget value if the widget exists."""
    try:
        owner.query_one(f"#{field_id}", Input).value = value or ""
    except Exception:
        return


def _set_switch(owner: Widget, field_id: str, value: bool) -> None:
    """Set a Switch widget value if the widget exists."""
    try:
        owner.query_one(f"#{field_id}", Switch).value = bool(value)
    except Exception:
        return


def _set_select_options(owner: Widget, field_id: str, values: list[str]) -> None:
    """Replace Select options if the widget exists."""
    try:
        owner.query_one(f"#{field_id}", Select).set_options([(v, v) for v in values])
    except Exception:
        return


def _set_select_value(owner: Widget, field_id: str, value: str) -> None:
    """Set a Select value if the widget exists."""
    try:
        owner.query_one(f"#{field_id}", Select).value = value
    except Exception:
        return


def _save_reset_row(section: str) -> Widget:
    """Create Save/Reset action row for a settings section."""
    return Horizontal(
        Button("Save", id=f"settings-save-{section}"),
        Button("Reset", id=f"settings-reset-{section}"),
        classes="settings-actions",
    )


def _replace_container_children(owner: Widget, selector: str, widgets: Sequence[Widget]) -> None:
    """Replace all children in a container matched by selector."""
    try:
        container = owner.query_one(selector)
        if hasattr(container, "remove_children"):
            container.remove_children()
        for widget in widgets:
            container.mount(widget)
    except Exception:
        return


class BaseSettingsTab(Widget):
    """Base widget for settings tab views."""

    class StatusUpdate(Message):
        """Status update emitted when no controller status handler is available."""

        def __init__(self, message: str, error: bool = False) -> None:
            super().__init__()
            self.message = message
            self.error = error

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._is_refreshing = False

    def refresh_fields(self, config: Any) -> None:
        """Refresh tab widgets from config values."""
        raise NotImplementedError("Settings tab must implement refresh_fields().")

    def apply_update(self, field_id: str, value: Any, config: Any) -> None:
        """Apply a single field update to the config object."""
        raise NotImplementedError("Settings tab must implement apply_update().")

    def guarded_refresh_fields(self, config: Any) -> None:
        """Run refresh with `_is_refreshing` guard enabled."""
        if self._is_refreshing:
            return
        self._is_refreshing = True
        try:
            self.refresh_fields(config)
        finally:
            self._is_refreshing = False

    def post_status(self, message: str, error: bool = False) -> None:
        """Post status through the parent controller, or emit a message."""
        controller = getattr(self, "parent", None)
        while controller is not None:
            set_status = getattr(controller, "_set_status", None)
            if callable(set_status):
                set_status(message, error)
                return
            controller = getattr(controller, "parent", None)
        self.post_message(self.StatusUpdate(message, error))

    def _field(
        self,
        label: str,
        field_id: str,
        *,
        placeholder: str = "",
        disabled: bool = False,
        field_type: str = "input",
        value: Any = "",
    ) -> Widget:
        return _field(
            label,
            field_id,
            placeholder=placeholder,
            disabled=disabled,
            field_type=field_type,
            value=value,
        )

    def _select_field(
        self,
        label: str,
        field_id: str,
        options: Optional[list[tuple[str, str]]] = None,
    ) -> Widget:
        return _select_field(label, field_id, options=options)

    def _set_input(self, field_id: str, value: str) -> None:
        _set_input(self, field_id, value)

    def _set_switch(self, field_id: str, value: bool) -> None:
        _set_switch(self, field_id, value)

    def _set_select_options(self, field_id: str, values: list[str]) -> None:
        _set_select_options(self, field_id, values)

    def _set_select_value(self, field_id: str, value: str) -> None:
        _set_select_value(self, field_id, value)

    def _save_reset_row(self, section: str) -> Widget:
        return _save_reset_row(section)

    def _replace_container_children(self, selector: str, widgets: Sequence[Widget]) -> None:
        _replace_container_children(self, selector, widgets)
