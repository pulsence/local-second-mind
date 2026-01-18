"""
Ingest command handler.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.commands.ingest.
This module re-exports from lsm.gui.shell.commands for backward compatibility.
"""

from __future__ import annotations

from lsm.gui.shell.commands.ingest import run_ingest

__all__ = ["run_ingest"]
