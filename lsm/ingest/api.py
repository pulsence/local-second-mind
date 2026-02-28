"""Clean, UI-agnostic public API for ingest operations."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from lsm.config.models import LSMConfig
from lsm.ingest.stats import (
    get_collection_info as _get_collection_info,
    get_collection_stats as _get_collection_stats,
)
from lsm.vectordb import create_vectordb_provider


@dataclass
class IngestResult:
    total_files: int
    completed_files: int
    skipped_files: int
    chunks_added: int
    elapsed_seconds: float
    errors: list[Dict[str, Any]]


@dataclass
class CollectionInfo:
    name: str
    chunk_count: int
    provider: str


@dataclass
class CollectionStats:
    chunk_count: int
    unique_files: int
    file_types: Dict[str, int]
    top_files: list[Dict[str, Any]]


def _runtime_artifact_dir(config: LSMConfig) -> Path:
    vdb_path = Path(config.vectordb.path)
    if str(vdb_path).lower().endswith(".db"):
        return vdb_path.parent
    return vdb_path


def _build_stats_cache(
    config: LSMConfig,
    connection: Optional[sqlite3.Connection] = None,
):
    from lsm.ingest.stats_cache import StatsCache

    cache_key = f"{config.vectordb.collection}:collection_stats"
    if config.vectordb.provider == "sqlite":
        if isinstance(connection, sqlite3.Connection):
            return StatsCache(connection=connection, cache_key=cache_key)
        return StatsCache(db_path=config.vectordb.path, cache_key=cache_key)
    return StatsCache(_runtime_artifact_dir(config) / "stats_cache.json", cache_key=cache_key)


def get_collection_info(config: LSMConfig) -> CollectionInfo:
    """Return collection info in structured form."""
    provider = create_vectordb_provider(config.vectordb)
    info = _get_collection_info(provider)
    return CollectionInfo(
        name=info.get("name", config.vectordb.collection),
        chunk_count=int(info.get("count", 0)),
        provider=info.get("provider", provider.name),
    )


def get_collection_stats(
    config: LSMConfig,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> CollectionStats:
    """Return collection stats in structured form."""
    provider = create_vectordb_provider(config.vectordb)
    count = provider.count()

    def report_progress(analyzed: int) -> None:
        if progress_callback:
            progress_callback(analyzed, count)

    runtime_dir = _runtime_artifact_dir(config)
    stats_cache = _build_stats_cache(
        config,
        connection=getattr(provider, "connection", None),
    )
    stats = _get_collection_stats(
        provider,
        limit=None,
        error_report_path=runtime_dir / "ingest_error_report.json",
        progress_callback=report_progress,
        stats_cache=stats_cache,
    )

    top_files = [
        {"source_path": path, "chunk_count": chunks}
        for path, chunks in (stats.get("top_files") or {}).items()
    ]
    return CollectionStats(
        chunk_count=int(stats.get("total_chunks", count)),
        unique_files=int(stats.get("unique_files", 0)),
        file_types=dict(stats.get("file_types", {})),
        top_files=top_files,
    )


def run_ingest(
    config: LSMConfig,
    force: bool = False,
    force_reingest_changed_config: bool = False,
    force_file_pattern: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> IngestResult:
    """Run ingest pipeline and return structured result."""
    from lsm.ingest.pipeline import ingest

    # Resolve translation LLM config if translation is enabled
    translation_llm = None
    if config.ingest.enable_translation:
        try:
            translation_llm = config.llm.resolve_service("translation")
        except Exception:
            translation_llm = config.llm.resolve_service("default")

    result = ingest(
        roots=config.ingest.roots,
        chroma_flush_interval=None,
        embed_model_name=config.embed_model,
        device=config.device,
        batch_size=config.batch_size,
        manifest_path=None,
        exts=config.ingest.exts,
        exclude_dirs=config.ingest.exclude_set,
        vectordb_config=config.vectordb,
        dry_run=config.ingest.dry_run,
        enable_ocr=config.ingest.enable_ocr,
        skip_errors=config.ingest.skip_errors,
        chunk_size=config.ingest.chunk_size,
        chunk_overlap=config.ingest.chunk_overlap,
        chunking_strategy=config.ingest.chunking_strategy,
        max_heading_depth=config.ingest.max_heading_depth,
        intelligent_heading_depth=config.ingest.intelligent_heading_depth,
        progress_callback=progress_callback,
        enable_language_detection=config.ingest.enable_language_detection,
        enable_translation=config.ingest.enable_translation,
        translation_target=config.ingest.translation_target,
        translation_llm_config=translation_llm,
        embedding_dimension=config.embedding_dimension,
        max_files=config.ingest.max_files,
        max_seconds=config.ingest.max_seconds,
        enable_versioning=True,
        force_reingest=force,
        force_reingest_changed_config=force_reingest_changed_config,
        force_file_pattern=force_file_pattern,
    )

    # Invalidate stats cache after ingest
    _build_stats_cache(config).invalidate()

    return IngestResult(
        total_files=int(result.get("total_files", 0)),
        completed_files=int(result.get("completed_files", 0)),
        skipped_files=int(result.get("skipped_files", 0)),
        chunks_added=int(result.get("chunks_added", 0)),
        elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
        errors=list(result.get("errors", [])),
    )


def wipe_collection(config: LSMConfig) -> int:
    """Delete all chunks from the configured collection; return deleted count."""
    provider = create_vectordb_provider(config.vectordb)
    return provider.delete_all()
