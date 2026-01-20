"""
TUI Screen modules for LSM.

Contains screen classes for different views:
- MainScreen: Main tabbed interface
- IngestScreen: Document ingestion management
- QueryScreen: Query interface
- SettingsScreen: Configuration panel
"""

from __future__ import annotations

__all__ = [
    "MainScreen",
    "IngestScreen",
    "QueryScreen",
    "SettingsScreen",
    "RemoteScreen",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "MainScreen":
        from lsm.ui.tui.screens.main import MainScreen
        return MainScreen
    elif name == "IngestScreen":
        from lsm.ui.tui.screens.ingest import IngestScreen
        return IngestScreen
    elif name == "QueryScreen":
        from lsm.ui.tui.screens.query import QueryScreen
        return QueryScreen
    elif name == "SettingsScreen":
        from lsm.ui.tui.screens.settings import SettingsScreen
        return SettingsScreen
    elif name == "RemoteScreen":
        from lsm.ui.tui.screens.remote import RemoteScreen
        return RemoteScreen
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
