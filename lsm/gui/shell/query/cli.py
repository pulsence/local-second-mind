"""
CLI entrypoint for query module.

Provides clean initialization and entry to the query REPL.
"""

from __future__ import annotations

from lsm.config.models import LSMConfig
from lsm.gui.shell.logging import get_logger

logger = get_logger(__name__)


def run_query_cli(config: LSMConfig) -> int:
    """
    Run the query CLI with given configuration.

    Note: interactive query is now TUI-only; this returns a non-zero exit code.
    """
    logger.info("Query CLI deprecated; use TUI")
    print("Interactive query is now TUI-only. Run `lsm` to launch the TUI.")
    return 2
