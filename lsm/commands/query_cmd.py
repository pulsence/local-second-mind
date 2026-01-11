"""
CLI glue for `lsm query`.

Routes to the new modular query implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsm.config import load_config_from_file
from lsm.cli.logging import get_logger

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
        config.llm.model = args.model
        logger.info(f"Overriding LLM model to: {args.model}")

    if hasattr(args, 'no_rerank') and args.no_rerank:
        config.query.no_rerank = True
        logger.info("Disabling reranking")

    if hasattr(args, 'k') and args.k:
        config.query.k = args.k
        logger.info(f"Overriding retrieval depth to: {args.k}")

    logger.debug(f"Query configuration:")
    logger.debug(f"  Collection: {config.collection}")
    logger.debug(f"  LLM: {config.llm.provider}/{config.llm.model}")
    logger.debug(f"  Retrieval: k={config.query.k}")
    logger.debug(f"  Mode: {config.query.mode}")

    # Run query CLI with new modular system
    return run_query_cli(config)
