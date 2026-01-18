"""LSM Query Shell module - query mode REPL and CLI.

This module contains:
- cli: Command-line interface for query operations

Note: REPL functionality remains in lsm.query.repl for now to avoid
circular imports. This module provides the CLI entry point.
"""

from __future__ import annotations

__all__ = [
    "run_query_cli",
]

from lsm.gui.shell.query.cli import run_query_cli
