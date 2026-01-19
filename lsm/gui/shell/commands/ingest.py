"""
Command handler for ingest command.
"""

from __future__ import annotations

from lsm.gui.shell.ingest.cli import run_build_cli, run_tag_cli, run_wipe_cli


def run_ingest(args) -> int:
    """Run ingest command with build/tag/wipe subcommands."""
    command = getattr(args, "ingest_command", None)
    if command == "build":
        return run_build_cli(
            args.config,
            force=getattr(args, "force", False),
            skip_errors=getattr(args, "skip_errors", None),
            dry_run=getattr(args, "dry_run", None),
        )
    if command == "tag":
        return run_tag_cli(args.config, max_chunks=getattr(args, "max", None))
    if command == "wipe":
        return run_wipe_cli(args.config, confirm=getattr(args, "confirm", False))

    print("Missing ingest subcommand. Use `lsm ingest --help` for options.")
    return 2
