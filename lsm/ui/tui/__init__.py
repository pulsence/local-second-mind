"""
TUI (Text User Interface) module for LSM.

Provides a rich terminal interface built with Textual.
"""

from __future__ import annotations

__all__ = [
    "LSMApp",
    "run_tui",
]


def __getattr__(name: str):
    if name in __all__:
        from lsm.ui.tui.app import LSMApp, run_tui
        return {"LSMApp": LSMApp, "run_tui": run_tui}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
