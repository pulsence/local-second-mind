"""CLI glue for `lsm ingest`.

We import the legacy implementation lazily so that `python -m lsm --help`
works even if optional heavy dependencies are not yet installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestArgs:
    config: str


def run_ingest(args: IngestArgs) -> int:
    from lsm.ingest import legacy_ingest  # lazy import

    # legacy_ingest.main accepts argv list
    return int(legacy_ingest.main(["--config", args.config]) or 0)
