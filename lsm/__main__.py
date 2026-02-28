"""
Main CLI entry point for Local Second Mind.

Provides unified command-line interface with subcommands for ingest and query.
"""

import argparse
import sys
from pathlib import Path

from lsm.logging import configure_logging_from_args, get_logger

DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parent.parent / "config.json")

def build_parser() -> argparse.ArgumentParser:
    """
    Build the main argument parser with subcommands.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="lsm",
        description="Local Second Mind - Local-first RAG for personal knowledge management",
        epilog="Use 'lsm <command> --help' for more information on a specific command.",
    )

    # Global flags (available to all commands)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (overrides --verbose)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to file",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        required=False,  # Allow running without command for interactive mode
    )

    # -------------------------------------------------------------------------
    # Ingest subcommand
    # -------------------------------------------------------------------------
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the knowledge base",
        description="Scan directories, parse documents, and build vector database.",
    )
    ingest_subparsers = ingest_parser.add_subparsers(
        dest="ingest_command",
        title="ingest commands",
        required=True,
    )
    build_parser = ingest_subparsers.add_parser(
        "build",
        help="Build or update the vector database",
    )
    build_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Simulate ingest without writing to database",
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest all files (clears manifest)",
    )
    build_parser.add_argument(
        "--force-reingest-changed-config",
        action="store_true",
        help="Allow selective re-ingest when schema-affecting config changed",
    )
    build_parser.add_argument(
        "--force-file-pattern",
        type=str,
        help="Only ingest files matching this glob pattern",
    )
    build_parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=None,
        help="Continue ingest when parsing errors occur",
    )

    tag_parser = ingest_subparsers.add_parser(
        "tag",
        help="Run AI tagging on untagged chunks",
    )
    tag_parser.add_argument(
        "--max",
        type=int,
        help="Maximum number of chunks to tag",
    )

    wipe_parser = ingest_subparsers.add_parser(
        "wipe",
        help="Delete all chunks in the collection",
    )
    wipe_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive wipe",
    )

    # -------------------------------------------------------------------------
    # DB maintenance subcommand
    # -------------------------------------------------------------------------
    db_parser = subparsers.add_parser(
        "db",
        help="Database maintenance commands",
        description="Prune old chunk versions and run DB maintenance tasks.",
    )
    db_subparsers = db_parser.add_subparsers(
        dest="db_command",
        title="db commands",
        required=True,
    )

    prune_parser = db_subparsers.add_parser(
        "prune",
        help="Prune non-current chunk versions",
    )
    prune_parser.add_argument(
        "--max-versions",
        type=int,
        help="Keep at most N non-current versions per source file",
    )
    prune_parser.add_argument(
        "--older-than-days",
        type=int,
        help="Prune only versions older than N days",
    )

    complete_parser = db_subparsers.add_parser(
        "complete",
        help="Run selective completion re-ingest for changed configuration",
    )
    complete_parser.add_argument(
        "--force-file-pattern",
        type=str,
        help="Restrict completion ingest to files matching this glob pattern",
    )

    return parser




def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for LSM CLI.

    Args:
        argv: Command-line arguments (for testing)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on arguments
    configure_logging_from_args(
        verbose=args.verbose,
        log_level=args.log_level,
        log_file=str(args.log_file) if args.log_file else None,
    )

    logger = get_logger(__name__)
    logger.debug(f"Parsed arguments: {args}")

    # Load configuration
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        print(f"Error: Configuration file not found: {cfg_path}")
        print("Create a config.json file or use --config to specify a different path.")
        return 1

    logger.info(f"Loading configuration from: {cfg_path}")
    from lsm.config import load_config_from_file
    config = load_config_from_file(cfg_path)

    # TUI interface if no command specified
    if not args.command:
        logger.info("Starting TUI interface")
        from lsm.ui.tui.app import run_tui
        return run_tui(config)

    # Dispatch to appropriate command (single-shot mode)
    try:
        if args.command == "ingest":
            logger.info("Starting ingest command")
            from lsm.ui.shell.cli import run_ingest
            return run_ingest(args)
        if args.command == "db":
            logger.info("Starting db command")
            from lsm.ui.shell.cli import run_db
            return run_db(args)

        else:
            # Should not reach here with required=True in subparsers
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user.")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.exception("Unhandled exception in main")
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
