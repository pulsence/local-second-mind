"""LSM Shell module - CLI and REPL interfaces.

This module contains:
- unified: Main unified shell combining ingest and query modes
- logging: CLI logging utilities
- ingest: Ingest-mode REPL and CLI
- query: Query-mode REPL and CLI
- commands: Command handlers for CLI invocation
- tui: Textual-based TUI interface
"""

from __future__ import annotations

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "configure_logging_from_args",
    # Unified shell
    "run_unified_shell",
    "UnifiedShell",
    # TUI
    "LSMApp",
    "run_tui",
]

# Import logging utilities (these are used everywhere)
from lsm.gui.shell.logging import get_logger, setup_logging, configure_logging_from_args

# Import main entry points for convenience
from lsm.gui.shell.unified import run_unified_shell, UnifiedShell

# Lazy import for TUI to avoid loading textual unless needed
def __getattr__(name: str):
    if name in ("LSMApp", "run_tui"):
        from lsm.gui.shell.tui import LSMApp, run_tui
        if name == "LSMApp":
            return LSMApp
        return run_tui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
