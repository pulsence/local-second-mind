"""Clean, UI-agnostic public API for ingest operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from lsm.config.models import LSMConfig
from lsm.ingest.stats import (
    get_collection_info as _get_collection_info,
    get_collection_stats as _get_collection_stats,
)
from lsm.vectordb import create_vectordb_provider
from lsm.vectordb.utils import require_chroma_collection


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


def get_collection_info(config: LSMConfig) -> CollectionInfo:
    """Return collection info in structured form."""
    provider = create_vectordb_provider(config.vectordb)
    if getattr(provider, "name", "") != "chromadb":
        stats = provider.get_stats()
        return CollectionInfo(
            name=config.vectordb.collection,
            chunk_count=provider.count(),
            provider=stats.get("provider", provider.name),
        )

    collection = require_chroma_collection(provider, "get_collection_info")
    info = _get_collection_info(collection)
    return CollectionInfo(
        name=info.get("name", config.vectordb.collection),
        chunk_count=int(info.get("count", 0)),
        provider="chromadb",
    )


def get_collection_stats(
    config: LSMConfig,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> CollectionStats:
    """Return collection stats in structured form."""
    provider = create_vectordb_provider(config.vectordb)
    count = provider.count()
    if getattr(provider, "name", "") != "chromadb":
        return CollectionStats(
            chunk_count=count,
            unique_files=0,
            file_types={},
            top_files=[],
        )

    collection = require_chroma_collection(provider, "get_collection_stats")
    def report_progress(analyzed: int) -> None:
        if progress_callback:
            progress_callback(analyzed, count)

    stats = _get_collection_stats(
        collection,
        limit=None,
        error_report_path=config.ingest.manifest.parent / "ingest_error_report.json",
        progress_callback=report_progress,
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
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> IngestResult:
    """Run ingest pipeline and return structured result."""
    from lsm.ingest.pipeline import ingest

    if force and config.ingest.manifest.exists():
        config.ingest.manifest.unlink()

    result = ingest(
        roots=config.ingest.roots,
        chroma_flush_interval=config.ingest.chroma_flush_interval,
        embed_model_name=config.embed_model,
        device=config.device,
        batch_size=config.batch_size,
        manifest_path=config.ingest.manifest,
        exts=config.ingest.exts,
        exclude_dirs=config.ingest.exclude_set,
        vectordb_config=config.vectordb,
        dry_run=config.ingest.dry_run,
        enable_ocr=config.ingest.enable_ocr,
        skip_errors=config.ingest.skip_errors,
        chunk_size=config.ingest.chunk_size,
        chunk_overlap=config.ingest.chunk_overlap,
        chunking_strategy=config.ingest.chunking_strategy,
        progress_callback=progress_callback,
    )

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
    collection = require_chroma_collection(provider, "/wipe")
    results = collection.get(include=[])
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
