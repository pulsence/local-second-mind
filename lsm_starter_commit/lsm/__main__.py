"""Unified CLI entrypoint.

Usage:
  python -m lsm ingest --config config.yaml
  python -m lsm query  --config config.yaml

This starter refactor keeps your existing ingest/query implementations intact
by importing them as legacy modules and wrapping their CLIs.
"""

from __future__ import annotations

import argparse

from lsm.commands.ingest_cmd import run_ingest
from lsm.commands.query_cmd import run_query


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lsm", description="Local Second Mind CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest/update local files into Chroma")
    p_ing.add_argument("--config", default="config.json", help="Path to YAML/JSON config")
    p_ing.set_defaults(func=run_ingest)

    p_q = sub.add_parser("query", help="Query local knowledge base (interactive)")
    p_q.add_argument("--config", default="config.json", help="Path to YAML/JSON config")
    p_q.set_defaults(func=run_query)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
