"""
TUI Widget modules for LSM.

Contains custom widgets:
- ResultsPanel: Scrollable results display with citation expansion
- CommandInput: Input with history and autocomplete
- StatusBar: Mode indicator, provider status, cost counter
"""

from __future__ import annotations

__all__ = [
    "ResultsPanel",
    "CommandInput",
    "StatusBar",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "ResultsPanel":
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        return ResultsPanel
    elif name == "CommandInput":
        from lsm.gui.shell.tui.widgets.input import CommandInput
        return CommandInput
    elif name == "StatusBar":
        from lsm.gui.shell.tui.widgets.status import StatusBar
        return StatusBar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
