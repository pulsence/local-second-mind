"""
Query command handler.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.commands.query.
This module re-exports from lsm.gui.shell.commands for backward compatibility.
"""

from __future__ import annotations

from lsm.gui.shell.commands.query import run_query, run_single_shot_query

__all__ = ["run_query", "run_single_shot_query"]
