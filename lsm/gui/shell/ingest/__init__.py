"""LSM Ingest Shell module - ingest mode REPL and CLI.

This module contains:
- cli: Command-line interface for ingest operations

Note: REPL functionality remains in lsm.ingest.repl for now to avoid
circular imports. This module provides the CLI entry point.
"""

from __future__ import annotations

__all__ = [
    "main",
]

from lsm.gui.shell.ingest.cli import main
