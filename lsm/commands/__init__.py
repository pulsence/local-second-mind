"""
Command entrypoints for the unified CLI.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.commands.
All functionality has been moved to lsm.gui.shell.commands.

This module re-exports from lsm.gui.shell.commands for backward compatibility.
"""

from lsm.gui.shell.commands import run_ingest, run_query

__all__ = [
    "run_ingest",
    "run_query",
]
