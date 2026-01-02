from __future__ import annotations

from lsm.ingest.cli import main as ingest_main


def run_ingest(args) -> int:
    return ingest_main(args.config)
