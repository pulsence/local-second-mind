"""
TUI (Text User Interface) module for LSM.

Provides a rich terminal interface built with Textual.
"""

from __future__ import annotations

__all__ = [
    "LSMApp",
    "run_tui",
]

from lsm.ui.tui.app import LSMApp, run_tui
