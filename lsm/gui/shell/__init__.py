"""LSM Shell module - CLI and REPL interfaces.

This module contains:
- unified: Main unified shell combining ingest and query modes
- logging: CLI logging utilities
- ingest: Ingest-mode REPL and CLI
- query: Query-mode REPL and CLI
- commands: Command handlers for CLI invocation
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
]

# Import logging utilities (these are used everywhere)
from lsm.gui.shell.logging import get_logger, setup_logging, configure_logging_from_args

# Import main entry points for convenience
from lsm.gui.shell.unified import run_unified_shell, UnifiedShell
