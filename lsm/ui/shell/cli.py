"""
CLI entry point for ingest single-shot commands.

Provides build, tag, and wipe command runners for non-interactive usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lsm.config import load_config_from_file
from lsm.config.models import LSMConfig
from lsm.logging import get_logger
from lsm.ingest.api import run_ingest, wipe_collection
from lsm.ingest.tagging import tag_chunks
from lsm.vectordb import create_vectordb_provider

logger = get_logger(__name__)


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


def _load_config(config_path: str | Path) -> LSMConfig:
    """Load and validate configuration."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(
            f"Ingest config not found: {cfg_path}\n"
            f"Either create it or pass --config explicitly."
        )

    logger.info(f"Loading configuration from: {cfg_path}")
    return load_config_from_file(cfg_path)


def run_build_cli(
    config_path: str | Path,
    force: bool = False,
    skip_errors: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> int:
    """
    Run the ingest build pipeline.

    Args:
        config_path: Path to configuration file
        force: If True, clears manifest to re-ingest all files
        skip_errors: Override skip_errors config
        dry_run: Override dry_run config

    Returns:
        Exit code (0 for success)
    """
    config = _load_config(config_path)

    if skip_errors is not None:
        config.ingest.skip_errors = skip_errors
    if dry_run is not None:
        config.ingest.dry_run = dry_run

    def progress(event: str, current: int, total: int, message: str) -> None:
        if total > 0:
            print(f"[{event}] {current}/{total} {message}")
        else:
            print(f"[{event}] {message}")

    run_ingest(
        config,
        force=force,
        progress_callback=progress,
    )

    logger.info("Ingest completed successfully")
    return 0


def run_tag_cli(
    config_path: str | Path,
    max_chunks: Optional[int] = None,
) -> int:
    """
    Run AI tagging on untagged chunks.

    Args:
        config_path: Path to configuration file
        max_chunks: Optional limit on number of chunks to tag

    Returns:
        Exit code (0 for success)
    """
    config = _load_config(config_path)
    provider = create_vectordb_provider(config.vectordb)
    tagging_config = config.llm.get_tagging_config()

    print("\nStarting AI tagging...")
    print(f"Using model: {tagging_config.model}")
    print(f"Provider: {tagging_config.provider}")
    if max_chunks:
        print(f"Max chunks to tag: {max_chunks}")

    tagged, failed = tag_chunks(
        collection=provider,
        llm_config=tagging_config,
        num_tags=3,
        batch_size=100,
        max_chunks=max_chunks,
        dry_run=False,
    )

    print("\nTagging complete")
    print(f"Successfully tagged: {tagged} chunks")
    print(f"Failed: {failed} chunks")

    return 0


def run_wipe_cli(
    config_path: str | Path,
    confirm: bool = False,
) -> int:
    """
    Wipe the vector DB collection.

    Args:
        config_path: Path to configuration file
        confirm: Require explicit confirmation

    Returns:
        Exit code (0 for success)
    """
    if not confirm:
        print("Refusing to wipe without confirmation. Use --confirm to proceed.")
        return 2

    config = _load_config(config_path)
    try:
        deleted = wipe_collection(config)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"\nDeleted {deleted:,} chunks from collection '{config.collection}'.")
    print("Collection cleared successfully.")
    return 0
