"""
CLI utilities for Local Second Mind.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.
All functionality has been moved to lsm.gui.shell.

This module re-exports from lsm.gui.shell for backward compatibility.
"""

import warnings

# Re-export from new location for backward compatibility
from lsm.gui.shell.logging import (
    setup_logging,
    get_logger,
    configure_logging_from_args,
)

from lsm.gui.shell.unified import run_unified_shell

__all__ = [
    "setup_logging",
    "get_logger",
    "configure_logging_from_args",
    "run_unified_shell",
]


def _deprecated_warning():
    warnings.warn(
        "lsm.cli is deprecated, use lsm.gui.shell instead",
        DeprecationWarning,
        stacklevel=3,
    )
