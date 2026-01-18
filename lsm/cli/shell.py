"""
Unified interactive shell for LSM.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.unified.
All functionality has been moved to lsm.gui.shell.unified.

This module re-exports from lsm.gui.shell.unified for backward compatibility.
"""

from __future__ import annotations

# Re-export from new location
from lsm.gui.shell.unified import UnifiedShell, run_unified_shell, ContextType
from lsm.gui.shell.logging import get_logger

__all__ = [
    "UnifiedShell",
    "run_unified_shell",
    "ContextType",
]
