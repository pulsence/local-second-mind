"""
CLI glue for `lsm query`.

Routes to the query REPL implementation using new config system.
No more sys.argv manipulation!
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
    from lsm.query import legacy_query

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
        config.query.rerank_strategy = "none"
        logger.info("Disabling reranking")

    if hasattr(args, 'k') and args.k:
        config.query.k = args.k
        logger.info(f"Overriding retrieval depth to: {args.k}")

    logger.debug(f"Query configuration:")
    logger.debug(f"  Collection: {config.collection}")
    logger.debug(f"  LLM: {config.llm.provider}/{config.llm.model}")
    logger.debug(f"  Retrieval: k={config.query.k}")

    # Convert config to legacy dict format for now
    # TODO: Eventually refactor legacy_query to use LSMConfig directly
    from lsm.config.loader import normalize_config_legacy

    raw_dict = {
        "roots": [str(r) for r in config.ingest.roots],
        "persist_dir": str(config.persist_dir),
        "collection": config.collection,
        "embed_model": config.embed_model,
        "device": config.device,
        "batch_size": config.batch_size,
        "openai": {"api_key": config.llm.api_key},
        "query": {
            "k": config.query.k,
            "k_rerank": config.query.k_rerank,
            "no_rerank": config.query.no_rerank,
            "max_per_file": config.query.max_per_file,
            "local_pool": config.query.local_pool,
            "model": config.llm.model,
            "min_relevance": config.query.min_relevance,
            "path_contains": config.query.path_contains,
            "ext_allow": config.query.ext_allow,
            "ext_deny": config.query.ext_deny,
            "retrieve_k": config.query.retrieve_k,
        }
    }

    cfg_dict = normalize_config_legacy(raw_dict, cfg_path)

    # Build runtime components
    embedder = legacy_query.build_embedder(cfg_dict["embed_model"], device=cfg_dict["device"])
    col = legacy_query.chroma_collection(cfg_dict["persist_dir"], cfg_dict["collection"])
    oa = legacy_query.openai_client(cfg_dict.get("openai_api_key"))
    rt = legacy_query.Runtime(embedder=embedder, col=col, oa=oa, cfg=cfg_dict)

    # Session state
    state = legacy_query.SessionState(
        path_contains=cfg_dict.get("path_contains"),
        ext_allow=cfg_dict.get("ext_allow"),
        ext_deny=cfg_dict.get("ext_deny"),
        model=cfg_dict.get("model")
    )

    # Start REPL
    legacy_query.print_banner()

    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return 0

            if not line:
                continue

            try:
                if legacy_query.handle_command(line, state, rt):
                    continue
            except SystemExit:
                print("Exiting.")
                return 0

            legacy_query.run_query_turn(line, rt, state)

    except Exception as e:
        logger.exception("Error in query REPL")
        print(f"\nError: {e}")
        return 1
