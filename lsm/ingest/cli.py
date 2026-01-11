"""
CLI entry point for the ingest command.

Loads configuration and orchestrates the ingest pipeline.
"""

from __future__ import annotations

from pathlib import Path

from lsm.config import load_config_from_file
from lsm.cli.logging import get_logger
from lsm.ingest.pipeline import ingest

logger = get_logger(__name__)


def main(config_path: str | Path) -> int:
    """
    Run the ingest pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Exit code (0 for success)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Convert to Path and resolve
    cfg_path = Path(config_path).expanduser().resolve()

    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(
            f"Ingest config not found: {cfg_path}\n"
            f"Either create it or pass --config explicitly."
        )

    # Load and validate configuration
    logger.info(f"Loading configuration from: {cfg_path}")
    config = load_config_from_file(cfg_path)

    logger.info(f"Ingest configuration:")
    logger.info(f"  Roots: {config.ingest.roots}")
    logger.info(f"  Collection: {config.collection}")
    logger.info(f"  Embed model: {config.embed_model}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Extensions: {len(config.ingest.exts)} types")
    logger.info(f"  Dry run: {config.ingest.dry_run}")

    # Run ingest pipeline
    ingest(
        roots=config.ingest.roots,
        persist_dir=config.persist_dir,
        chroma_flush_interval=config.ingest.chroma_flush_interval,
        collection_name=config.collection,
        embed_model_name=config.embed_model,
        device=config.device,
        batch_size=config.batch_size,
        manifest_path=config.ingest.manifest,
        exts=config.ingest.exts,
        exclude_dirs=config.ingest.exclude_set,
        dry_run=config.ingest.dry_run,
        enable_ocr=config.ingest.enable_ocr,
        chunk_size=config.ingest.chunk_size,
        chunk_overlap=config.ingest.chunk_overlap,
    )

    logger.info("Ingest completed successfully")
    return 0
