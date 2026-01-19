"""LSM Shell Commands module - CLI command handlers.

This module contains command handlers for CLI invocation:
- ingest: Ingest command handler
- query: Query command handler
"""

from __future__ import annotations

__all__ = [
    "run_ingest",
    "run_query",
]

from lsm.ui.shell.commands.ingest import run_ingest
from lsm.ui.shell.commands.query import run_query
