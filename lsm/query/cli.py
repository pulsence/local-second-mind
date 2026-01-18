"""
CLI entrypoint for query module.

Provides clean initialization and entry to the query REPL.
"""

from __future__ import annotations

from pathlib import Path

from lsm.config.models import LSMConfig
from lsm.gui.shell.logging import get_logger
from .retrieval import init_embedder
from lsm.gui.shell.query.repl import run_repl
from lsm.vectordb import create_vectordb_provider

logger = get_logger(__name__)


def run_query_cli(config: LSMConfig) -> int:
    """
    Run the query CLI with given configuration.

    Args:
        config: LSM configuration object

    Returns:
        Exit code (0 for success)

    Example:
        >>> from lsm.config import load_config_from_file
        >>> config = load_config_from_file("config.json")
        >>> run_query_cli(config)
    """
    logger.info("Starting query CLI")

    # Initialize embedder
    logger.info(f"Initializing embedder: {config.embed_model}")
    embedder = init_embedder(config.embed_model, device=config.device)

    # Initialize vector DB provider
    logger.info(f"Initializing vector DB provider: {config.vectordb.provider}")
    if config.vectordb.provider == "chromadb":
        persist_dir = Path(config.persist_dir)
        if not persist_dir.exists():
            logger.error(f"Persist directory does not exist: {persist_dir}")
            print(f"Error: ChromaDB directory not found: {persist_dir}")
            print("Run 'lsm ingest' first to create the database.")
            return 1

    provider = create_vectordb_provider(config.vectordb)

    # Check collection has data
    count = provider.count()
    if count == 0:
        logger.warning("Vector DB collection is empty")
        print(f"Warning: Collection '{config.collection}' is empty.")
        print("Run 'lsm ingest' to populate the database.")
        return 1

    logger.info(f"Collection ready with {count} chunks")

    # Start REPL
    try:
        run_repl(config, embedder, provider)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except Exception as e:
        logger.error(f"Query CLI failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1

    logger.info("Query CLI exited successfully")
    return 0
