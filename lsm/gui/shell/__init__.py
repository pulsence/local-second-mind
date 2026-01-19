"""LSM Shell module - CLI helpers and TUI interface.

This module contains:
- logging: CLI logging utilities
- ingest: Ingest command handlers and CLI helpers
- query: Query command handlers
- commands: Command handlers for CLI invocation
- tui: Textual-based TUI interface
"""

from __future__ import annotations

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "configure_logging_from_args",
    # TUI
    "LSMApp",
    "run_tui",
]

# Import logging utilities (these are used everywhere)
from lsm.gui.shell.logging import get_logger, setup_logging, configure_logging_from_args

# Lazy import for TUI to avoid loading textual unless needed
def __getattr__(name: str):
    if name in ("LSMApp", "run_tui"):
        from lsm.gui.shell.tui import LSMApp, run_tui
        if name == "LSMApp":
            return LSMApp
        return run_tui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
