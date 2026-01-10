"""
Main CLI entry point for Local Second Mind.

Provides unified command-line interface with subcommands for ingest and query.
"""

import argparse
import sys
from pathlib import Path

from lsm.cli.logging import configure_logging_from_args, get_logger
from lsm.commands.ingest_cmd import run_ingest
from lsm.commands.query_cmd import run_query


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
        default="config.json",
        help="Path to configuration file (default: config.json)",
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
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate ingest without writing to database",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest all files (ignore manifest)",
    )

    # -------------------------------------------------------------------------
    # Query subcommand
    # -------------------------------------------------------------------------
    query_parser = subparsers.add_parser(
        "query",
        help="Query the knowledge base",
        description="Interactively search and ask questions about your documents.",
    )
    query_parser.add_argument(
        "--mode",
        choices=["grounded", "insight"],
        help="Query mode: grounded (strict citations) or insight (thematic analysis)",
    )
    query_parser.add_argument(
        "--model",
        help="Override LLM model from config",
    )
    query_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip LLM reranking step",
    )
    query_parser.add_argument(
        "-k",
        type=int,
        help="Number of chunks to retrieve",
    )

    return parser


def interactive_select_command() -> str:
    """
    Prompt user to select a command interactively.

    Returns:
        Command name ('ingest' or 'query')

    Raises:
        SystemExit: If user provides invalid choice
    """
    print("\n╔═══════════════════════════════════════╗")
    print("║      Local Second Mind (LSM)          ║")
    print("╚═══════════════════════════════════════╝\n")
    print("Select a command:")
    print("  1) ingest  – Ingest/update local documents")
    print("  2) query   – Query the knowledge base")
    print("  0) exit    – Exit\n")

    while True:
        choice = input("Enter choice [1/2/0]: ").strip()

        if choice == "1":
            return "ingest"
        elif choice == "2":
            return "query"
        elif choice == "0":
            print("Goodbye!")
            raise SystemExit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")


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

    # Interactive mode if no command specified
    if not args.command:
        command = interactive_select_command()
        # Create a new namespace with the selected command
        args.command = command

    # Dispatch to appropriate command
    try:
        if args.command == "ingest":
            logger.info("Starting ingest command")
            return run_ingest(args)

        elif args.command == "query":
            logger.info("Starting query command")
            return run_query(args)

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
