"""
Command handler for ingest command.
"""

from __future__ import annotations

from lsm.gui.shell.ingest.cli import main as ingest_main


def run_ingest(args) -> int:
    """Run ingest command with optional interactive mode."""
    interactive = getattr(args, 'interactive', False)
    skip_errors = getattr(args, "skip_errors", None)
    return ingest_main(args.config, interactive=interactive, skip_errors=skip_errors)
