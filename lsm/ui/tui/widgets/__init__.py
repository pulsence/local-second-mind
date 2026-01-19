"""
TUI Widget modules for LSM.

Contains custom widgets:
- ResultsPanel: Scrollable results display with citation expansion
- CommandInput: Input with history and autocomplete
- StatusBar: Mode indicator, provider status, cost counter

And messages:
- CitationSelected: Fired when a citation is selected
- CitationExpanded: Fired when a citation is expanded
- CommandSubmitted: Fired when a command is submitted
"""

from __future__ import annotations

__all__ = [
    "ResultsPanel",
    "ResultItem",
    "CitationSelected",
    "CitationExpanded",
    "CommandInput",
    "CommandSubmitted",
    "StatusBar",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "ResultsPanel":
        from lsm.ui.tui.widgets.results import ResultsPanel
        return ResultsPanel
    elif name == "ResultItem":
        from lsm.ui.tui.widgets.results import ResultItem
        return ResultItem
    elif name == "CitationSelected":
        from lsm.ui.tui.widgets.results import CitationSelected
        return CitationSelected
    elif name == "CitationExpanded":
        from lsm.ui.tui.widgets.results import CitationExpanded
        return CitationExpanded
    elif name == "CommandInput":
        from lsm.ui.tui.widgets.input import CommandInput
        return CommandInput
    elif name == "CommandSubmitted":
        from lsm.ui.tui.widgets.input import CommandSubmitted
        return CommandSubmitted
    elif name == "StatusBar":
        from lsm.ui.tui.widgets.status import StatusBar
        return StatusBar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
