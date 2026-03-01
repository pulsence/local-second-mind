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

    # -------------------------------------------------------------------------
    # Cache maintenance subcommand
    # -------------------------------------------------------------------------
    cache_parser = subparsers.add_parser(
        "cache",
        help="Cache maintenance commands",
        description="Clear reranker cache.",
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command",
        title="cache commands",
        required=True,
    )
    cache_clear_parser = cache_subparsers.add_parser("clear", help="Clear reranker cache")
    cache_clear_parser.add_argument(
        "--reranker",
        action="store_true",
        help="Clear the lsm_reranker_cache table",
    )

    # -------------------------------------------------------------------------
    # Migration subcommand
    # -------------------------------------------------------------------------
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate vector/state data across backends",
        description="Run explicit migration between supported backend/state formats.",
    )
    migrate_parser.add_argument(
        "--from",
        dest="migration_source",
        required=True,
        choices=["chroma", "sqlite", "postgresql", "v0.7"],
        help="Migration source backend/state",
    )
    migrate_parser.add_argument(
        "--to",
        dest="migration_target",
        required=True,
        choices=["sqlite", "postgresql", "v0.8"],
        help="Migration target backend/state",
    )
    migrate_parser.add_argument(
        "--source-path",
        type=str,
        help="Source sqlite/chroma path override",
    )
    migrate_parser.add_argument(
        "--source-collection",
        type=str,
        help="Source collection override",
    )
    migrate_parser.add_argument(
        "--source-connection-string",
        type=str,
        help="Source PostgreSQL connection string override",
    )
    migrate_parser.add_argument(
        "--source-dir",
        type=str,
        help="Legacy v0.7 state directory (manifest.json, memories.db, schedules.json)",
    )
    migrate_parser.add_argument(
        "--target-path",
        type=str,
        help="Target sqlite path override",
    )
    migrate_parser.add_argument(
        "--target-collection",
        type=str,
        help="Target collection override",
    )
    migrate_parser.add_argument(
        "--target-connection-string",
        type=str,
        help="Target PostgreSQL connection string override",
    )
    migrate_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Vector migration batch size",
    )

    # -------------------------------------------------------------------------
    # Eval subcommand
    # -------------------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate retrieval quality",
        description="Run retrieval evaluation against a test dataset.",
    )
    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_command",
        title="eval commands",
        required=True,
    )

    eval_retrieval_parser = eval_subparsers.add_parser(
        "retrieval",
        help="Run retrieval evaluation",
    )
    eval_retrieval_parser.add_argument(
        "--profile",
        type=str,
        default="dense_only",
        help="Retrieval profile to evaluate (default: dense_only)",
    )
    eval_retrieval_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset directory (default: bundled dataset)",
    )
    eval_retrieval_parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Name of baseline to compare against",
    )

    eval_save_parser = eval_subparsers.add_parser(
        "save-baseline",
        help="Save evaluation results as a named baseline",
    )
    eval_save_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Baseline name",
    )
    eval_save_parser.add_argument(
        "--profile",
        type=str,
        default="dense_only",
        help="Retrieval profile to evaluate and save",
    )
    eval_save_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset directory",
    )

    eval_list_parser = eval_subparsers.add_parser(
        "list-baselines",
        help="List saved evaluation baselines",
    )

    # -------------------------------------------------------------------------
    # Cluster subcommand
    # -------------------------------------------------------------------------
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster management commands",
        description="Build and manage embedding clusters for cluster-filtered retrieval.",
    )
    cluster_subparsers = cluster_parser.add_subparsers(
        dest="cluster_command",
        title="cluster commands",
        required=True,
    )

    cluster_build_parser = cluster_subparsers.add_parser(
        "build",
        help="Build cluster assignments for all current embeddings",
    )
    cluster_build_parser.add_argument(
        "--algorithm",
        choices=["kmeans", "hdbscan"],
        default=None,
        help="Clustering algorithm (default: from config or 'kmeans')",
    )
    cluster_build_parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters for k-means (default: from config or 50)",
    )

    cluster_visualize_parser = cluster_subparsers.add_parser(
        "visualize",
        help="Export a UMAP HTML plot of cluster distributions",
    )
    cluster_visualize_parser.add_argument(
        "--output",
        default="clusters.html",
        help="Output HTML file path (default: clusters.html)",
    )

    # --- graph command ---
    graph_parser = subparsers.add_parser(
        "graph",
        help="Knowledge graph commands",
        description="Build and manage the knowledge graph.",
    )
    graph_subparsers = graph_parser.add_subparsers(
        dest="graph_command",
        title="graph commands",
        required=True,
    )

    graph_build_links_parser = graph_subparsers.add_parser(
        "build-links",
        help="Build thematic links between chunks using embedding similarity",
    )
    graph_build_links_parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for creating edges (default: 0.8)",
    )
    graph_build_links_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for embedding comparison (default: 500)",
    )

    # --- finetune command ---
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune embedding models on corpus data",
    )
    finetune_subparsers = finetune_parser.add_subparsers(
        dest="finetune_command",
        title="finetune commands",
        required=True,
    )

    finetune_train_parser = finetune_subparsers.add_parser(
        "train",
        help="Extract training pairs and fine-tune embedding model",
    )
    finetune_train_parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model to fine-tune (default: all-MiniLM-L6-v2)",
    )
    finetune_train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    finetune_train_parser.add_argument(
        "--output",
        default="./models/finetuned",
        help="Output directory for fine-tuned model",
    )
    finetune_train_parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of training pairs to use",
    )

    finetune_list_parser = finetune_subparsers.add_parser(
        "list",
        help="List registered fine-tuned models",
    )

    finetune_activate_parser = finetune_subparsers.add_parser(
        "activate",
        help="Set a fine-tuned model as active",
    )
    finetune_activate_parser.add_argument(
        "model_id",
        help="Model ID to activate",
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

    # Database health check (skip for migrate command — it handles its own state)
    if args.command and args.command != "migrate":
        try:
            from lsm.db.health import check_db_health

            health = check_db_health(config)
            if health.status != "ok":
                if health.blocking:
                    print(f"Database issue: {health.details}")
                    if health.suggested_action:
                        print(f"Suggested action: {health.suggested_action}")
                    return 1
                else:
                    logger.warning(f"Database advisory: {health.details}")
        except Exception:
            logger.debug("Startup DB health check failed", exc_info=True)

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
        if args.command == "cache":
            logger.info("Starting cache command")
            from lsm.ui.shell.cli import run_cache
            return run_cache(args, config)
        if args.command == "migrate":
            logger.info("Starting migrate command")
            from lsm.ui.shell.cli import run_migrate
            return run_migrate(args)
        if args.command == "eval":
            logger.info("Starting eval command")
            from lsm.eval.cli import run_eval
            return run_eval(args, config)
        if args.command == "cluster":
            logger.info("Starting cluster command")
            from lsm.ui.shell.cli import run_cluster
            return run_cluster(args, config)
        if args.command == "finetune":
            logger.info("Starting finetune command")
            from lsm.ui.shell.cli import run_finetune
            return run_finetune(args, config)
        if args.command == "graph":
            logger.info("Starting graph command")
            from lsm.ui.shell.cli import run_graph
            return run_graph(args, config)

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
