"""CLI glue for `lsm query`.

We import the legacy implementation lazily so that the CLI parser can load
without requiring query-time dependencies.

The legacy query module parses arguments directly from sys.argv,
so we temporarily inject the argv we want.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class QueryArgs:
    config: str


def run_query(args: QueryArgs) -> int:
    from lsm.query import legacy_query  # lazy import

    old_argv = sys.argv[:]
    try:
        sys.argv = ["lsm", "--config", args.config]
        legacy_query.main()
        return 0
    finally:
        sys.argv = old_argv
