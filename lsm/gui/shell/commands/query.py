"""
CLI glue for `lsm query`.

Routes to the new modular query implementation.
Supports both interactive REPL and single-shot query modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsm.config import load_config_from_file
from lsm.config.models import LSMConfig
from lsm.gui.shell.logging import get_logger

logger = get_logger(__name__)


def run_query(args: Any) -> int:
    """
    Run the query command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    # Lazy import to avoid loading dependencies if not needed
    from lsm.query import run_query_cli

    # Load configuration
    cfg_path = Path(args.config).expanduser().resolve()

    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(f"Query config not found: {cfg_path}")

    logger.info(f"Loading configuration from: {cfg_path}")
    config = load_config_from_file(cfg_path)

    # Apply CLI overrides to config
    if hasattr(args, 'mode') and args.mode:
        config.query.mode = args.mode
        logger.info(f"Overriding query mode to: {args.mode}")

    if hasattr(args, 'model') and args.model:
        config.llm.override_feature_model("query", args.model)
        logger.info(f"Overriding LLM model to: {args.model}")

    if hasattr(args, 'no_rerank') and args.no_rerank:
        config.query.no_rerank = True
        logger.info("Disabling reranking")

    if hasattr(args, 'k') and args.k:
        config.query.k = args.k
        logger.info(f"Overriding retrieval depth to: {args.k}")

    logger.debug(f"Query configuration:")
    logger.debug(f"  Collection: {config.collection}")
    query_config = config.llm.get_query_config()
    logger.debug(f"  LLM: {query_config.provider}/{query_config.model}")
    logger.debug(f"  Retrieval: k={config.query.k}")
    logger.debug(f"  Mode: {config.query.mode}")

    # Check if single-shot mode (question provided and not interactive)
    is_interactive = getattr(args, 'interactive', False)
    question = getattr(args, 'question', None)

    if not is_interactive and question:
        # Single-shot query mode
        logger.info(f"Running single-shot query: {question}")
        return run_single_shot_query(config, question)
    else:
        # Interactive REPL mode
        logger.info("Starting interactive query REPL")
        return run_query_cli(config)


def run_single_shot_query(config: LSMConfig, question: str) -> int:
    """
    Run a single query and exit.

    Args:
        config: LSM configuration
        question: Question to ask

    Returns:
        Exit code (0 for success)
    """
    from lsm.query.retrieval import init_embedder
    from lsm.gui.shell.query.repl import run_query_turn
    from lsm.query.session import SessionState
    from lsm.vectordb import create_vectordb_provider

    try:
        # Initialize embedder
        logger.info(f"Initializing embedder: {config.embed_model}")
        embedder = init_embedder(config.embed_model, device=config.device)

        # Initialize vector DB provider
        if config.vectordb.provider == "chromadb":
            persist_dir = Path(config.persist_dir)
            if not persist_dir.exists():
                print(f"Error: ChromaDB directory not found: {persist_dir}")
                print("Run 'lsm ingest' first to create the database.")
                return 1

        logger.info(f"Initializing vector DB provider: {config.vectordb.provider}")
        provider = create_vectordb_provider(config.vectordb)

        # Check collection has data
        count = provider.count()
        if count == 0:
            print(f"Warning: Collection '{config.collection}' is empty.")
            print("Run 'lsm ingest' to populate the database.")
            return 1

        logger.info(f"Collection ready with {count} chunks")

        # Initialize session state
        query_config = config.llm.get_query_config()
        state = SessionState(
            path_contains=config.query.path_contains,
            ext_allow=config.query.ext_allow,
            ext_deny=config.query.ext_deny,
            model=query_config.model,
        )

        # Run single query
        run_query_turn(question, config, state, embedder, provider)

        return 0

    except Exception as e:
        logger.error(f"Single-shot query failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
