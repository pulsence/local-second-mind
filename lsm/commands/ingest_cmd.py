from __future__ import annotations

from lsm.ingest.cli import main as ingest_main


def run_ingest(args) -> int:
    """Run ingest command with optional interactive mode."""
    interactive = getattr(args, 'interactive', False)
    return ingest_main(args.config, interactive=interactive)
