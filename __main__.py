# __main__.py
import argparse
import sys
from types import SimpleNamespace

from lsm.commands.ingest_cmd import run_ingest
from lsm.commands.query_cmd import run_query


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lsm",
        description="Local Second Mind CLI",
    )

    p.add_argument(
        "tool",
        nargs="?",
        choices=["ingest", "query"],
        help="Tool to run (ingest or query). If omitted, you'll be prompted.",
    )
    p.add_argument(
        "--config",
        default="config.json",
        help="Path to unified YAML/JSON config (default: config.json)",
    )

    return p

def interactive_select_command() -> str:
    print("\nLocal Second Mind\n")
    print("Select a command:")
    print("  1) ingest  – Ingest/update local files")
    print("  2) query   – Query the knowledge base\n")

    choice = input("Enter choice [1/2]: ").strip()
    if choice == "1":
        return "ingest"
    if choice == "2":
        return "query"

    print("Invalid selec1tion.")
    raise SystemExit(2)

def dispatch(tool: str, config_path: str) -> int:
    # Keep the same signature your command modules expect: args.config
    args = SimpleNamespace(config=config_path)

    if tool == "ingest":
        return int(run_ingest(args) or 0)
    if tool == "query":
        return int(run_query(args) or 0)

    # Should never happen due to argparse choices
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # If tool is missing (e.g., user provided only --config), prompt for it.
    if not args.tool:
        tool = interactive_select_command()
        return dispatch(tool, args.config)

    return dispatch(args.tool, args.config)

if __name__ == "__main__":
    raise SystemExit(main())
