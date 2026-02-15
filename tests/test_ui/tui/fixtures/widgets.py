"""Reusable fake widget doubles for TUI screen tests.

These lightweight doubles mimic Textual widget interfaces without requiring
a real Textual app mount.
"""

from __future__ import annotations

from typing import Any


class FakeStatic:
    """Fake Static widget that records ``update()`` calls."""

    def __init__(self, initial: str = "") -> None:
        self.last = initial

    def update(self, message: str) -> None:
        self.last = message


class FakeInput:
    """Fake Input widget with value and focus tracking."""

    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class FakeSelect:
    """Fake Select widget with options and focus tracking."""

    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.options: list[tuple[str, str]] = []
        self.focused = False

    def set_options(self, options: list[tuple[str, str]]) -> None:
        self.options = list(options)

    def focus(self) -> None:
        self.focused = True


class FakeRichLog:
    """Fake RichLog widget that collects written lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ended = False

    def write(self, message: Any) -> None:
        self.lines.append(str(message))

    def scroll_end(self) -> None:
        self.ended = True


class FakeButton:
    """Fake Button widget with ID and label tracking."""

    def __init__(self, label: str = "", widget_id: str = "") -> None:
        self.label = label
        self.id = widget_id
