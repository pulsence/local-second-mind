"""
Collection statistics and introspection utilities for ingest pipeline.

Provides functions to inspect ChromaDB collections, analyze metadata,
and generate reports about the ingested knowledge base.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Iterator, Callable
from collections import Counter, defaultdict
from datetime import datetime

import chromadb
from chromadb.api.models.Collection import Collection
from lsm.vectordb.utils import require_chroma_collection


def get_collection_info(collection: Collection) -> Dict[str, Any]:
    """
    Get basic information about a collection.

    Args:
        collection: ChromaDB collection

    Returns:
        Dictionary with collection information
    """
    collection = require_chroma_collection(collection, "get_collection_info")
    count = collection.count()

    info = {
        "name": collection.name,
        "count": count,
        "id": collection.id,
    }

    # Try to get metadata if available
    try:
        metadata = collection.metadata
        if metadata:
            info["metadata"] = metadata
    except Exception:
        pass

    return info


def iter_collection_metadatas(
    collection: Collection,
    where: Optional[Dict[str, Any]] = None,
    batch_size: int = 5000,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over collection metadata in batches.

    Args:
        collection: ChromaDB collection
        where: Optional filter clause
        batch_size: Number of records per batch

    Yields:
        Metadata dictionaries
    """
    collection = require_chroma_collection(collection, "iter_collection_metadatas")
    offset = 0
    while True:
        try:
            results = collection.get(
                where=where if where else None,
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
        except TypeError:
            if offset != 0:
                break
            results = collection.get(
                where=where if where else None,
                limit=batch_size,
                include=["metadatas"],
            )
        metadatas = results.get("metadatas", [])
        if not metadatas:
            break
        for meta in metadatas:
            yield meta
        if len(metadatas) < batch_size:
            break
        offset += len(metadatas)


def _update_metadata_counters(
    meta: Dict[str, Any],
    ext_counter: Counter,
    file_counter: Counter,
    unique_files: Set[str],
    authors: Set[str],
    titles: Set[str],
    ingestion_dates: List[str],
    chunks_per_file: Dict[str, int],
) -> None:
    ext = meta.get("ext", "unknown")
    ext_counter[ext] += 1

    source_path = meta.get("source_path", "unknown")
    file_counter[source_path] += 1
    unique_files.add(source_path)
    chunks_per_file[source_path] += 1

    if "author" in meta and meta["author"]:
        authors.add(meta["author"])

    if "title" in meta and meta["title"]:
        titles.add(meta["title"])

    if "ingested_at" in meta:
        ingestion_dates.append(meta["ingested_at"])


def _finalize_metadata_stats(
    ext_counter: Counter,
    file_counter: Counter,
    unique_files: Set[str],
    authors: Set[str],
    titles: Set[str],
    ingestion_dates: List[str],
    chunks_per_file: Dict[str, int],
) -> Dict[str, Any]:
    stats = {
        "unique_files": len(unique_files),
        "file_types": dict(ext_counter.most_common()),
        "top_files": dict(file_counter.most_common(10)),
        "avg_chunks_per_file": sum(chunks_per_file.values()) / len(chunks_per_file) if chunks_per_file else 0,
        "max_chunks_per_file": max(chunks_per_file.values()) if chunks_per_file else 0,
        "min_chunks_per_file": min(chunks_per_file.values()) if chunks_per_file else 0,
    }

    if authors:
        stats["unique_authors"] = len(authors)
        stats["authors"] = sorted(list(authors))[:20]

    if titles:
        stats["unique_titles"] = len(titles)

    if ingestion_dates:
        sorted_dates = sorted(ingestion_dates)
        stats["first_ingested"] = sorted_dates[0]
        stats["last_ingested"] = sorted_dates[-1]

    return stats


def get_collection_stats(
    collection: Collection,
    limit: Optional[int] = None,
    batch_size: int = 5000,
    error_report_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Get detailed statistics about a collection.

    Args:
        collection: ChromaDB collection
        limit: Maximum number of chunks to analyze (for performance)

    Returns:
        Dictionary with detailed statistics
    """
    collection = require_chroma_collection(collection, "get_collection_stats")
    count = collection.count()

    if count == 0:
        return {
            "total_chunks": 0,
            "message": "Collection is empty",
        }

    try:
        if limit is None or limit <= 0 or limit >= count:
            ext_counter = Counter()
            file_counter = Counter()
            unique_files: Set[str] = set()
            authors: Set[str] = set()
            titles: Set[str] = set()
            ingestion_dates: List[str] = []
            chunks_per_file: Dict[str, int] = defaultdict(int)
            analyzed = 0

            for meta in iter_collection_metadatas(collection, batch_size=batch_size):
                _update_metadata_counters(
                    meta,
                    ext_counter,
                    file_counter,
                    unique_files,
                    authors,
                    titles,
                    ingestion_dates,
                    chunks_per_file,
                )
                analyzed += 1
                if progress_callback and analyzed % batch_size == 0:
                    progress_callback(analyzed)

            if analyzed == 0:
                return {
                    "total_chunks": count,
                    "message": "No metadata available",
                }

            stats = _finalize_metadata_stats(
                ext_counter,
                file_counter,
                unique_files,
                authors,
                titles,
                ingestion_dates,
                chunks_per_file,
            )
            stats["total_chunks"] = count
            stats["analyzed_chunks"] = analyzed
            stats["analysis_mode"] = "full"
        else:
            sample_size = min(count, limit)
            results = collection.get(
                limit=sample_size,
                include=["metadatas"]
            )

            metadatas = results.get("metadatas", [])

            if not metadatas:
                return {
                    "total_chunks": count,
                    "message": "No metadata available",
                }

            stats = analyze_metadata(metadatas, count)
            stats["total_chunks"] = count
            stats["analyzed_chunks"] = len(metadatas)
            stats["analysis_mode"] = "sample"

        if error_report_path:
            report = load_error_report(error_report_path)
            if report:
                stats["parse_errors"] = summarize_error_report(report, error_report_path)

        return stats

    except Exception as e:
        return {
            "total_chunks": count,
            "error": str(e),
        }


def analyze_metadata(metadatas: List[Dict[str, Any]], total_count: int) -> Dict[str, Any]:
    """
    Analyze metadata from collection chunks.

    Args:
        metadatas: List of metadata dictionaries
        total_count: Total number of chunks in collection

    Returns:
        Dictionary with analyzed statistics
    """
    # Count by file extension
    ext_counter = Counter()

    # Count by source file
    file_counter = Counter()

    # Track unique files
    unique_files: Set[str] = set()

    # Track authors if available
    authors: Set[str] = set()

    # Track titles if available
    titles: Set[str] = set()

    # Track ingestion dates
    ingestion_dates: List[str] = []

    # Track chunks per file
    chunks_per_file: Dict[str, int] = defaultdict(int)

    for meta in metadatas:
        _update_metadata_counters(
            meta,
            ext_counter,
            file_counter,
            unique_files,
            authors,
            titles,
            ingestion_dates,
            chunks_per_file,
        )

    return _finalize_metadata_stats(
        ext_counter,
        file_counter,
        unique_files,
        authors,
        titles,
        ingestion_dates,
        chunks_per_file,
    )


def load_error_report(path: Path) -> Optional[Dict[str, Any]]:
    """Load ingest error report if present."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_error_report(report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Summarize error report for stats output."""
    failed = report.get("failed_documents", []) or []
    page_errors = report.get("page_errors", []) or []
    summary = {
        "failed_documents": len(failed),
        "page_errors": len(page_errors),
        "generated_at": report.get("generated_at"),
        "path": str(path),
    }
    if failed:
        summary["sample_failed_documents"] = [
            item.get("source_path", "unknown") for item in failed[:5]
        ]
    return summary


def format_stats_report(stats: Dict[str, Any]) -> str:
    """
    Format statistics into a readable report.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted string report
    """
    lines = []

    lines.append("=" * 60)
    lines.append("COLLECTION STATISTICS")
    lines.append("=" * 60)

    # Basic info
    total = stats.get("total_chunks", 0)
    lines.append(f"\nTotal chunks:        {total:,}")

    if "analyzed_chunks" in stats:
        analyzed = stats["analyzed_chunks"]
        pct = (analyzed / total * 100) if total > 0 else 0
        lines.append(f"Analyzed chunks:     {analyzed:,} ({pct:.1f}%)")
        if stats.get("analysis_mode") == "sample":
            lines.append("Analysis mode:       sample")

    # File statistics
    if "unique_files" in stats:
        lines.append(f"Unique files:        {stats['unique_files']:,}")

    if "avg_chunks_per_file" in stats:
        lines.append(f"Avg chunks/file:     {stats['avg_chunks_per_file']:.1f}")
        lines.append(f"Max chunks/file:     {stats['max_chunks_per_file']:,}")
        lines.append(f"Min chunks/file:     {stats['min_chunks_per_file']:,}")

    # File types
    if "file_types" in stats:
        lines.append("\n" + "-" * 60)
        lines.append("FILE TYPES")
        lines.append("-" * 60)
        for ext, count in stats["file_types"].items():
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {ext:10s}  {count:7,} chunks  ({pct:5.1f}%)")

    # Top files
    if "top_files" in stats:
        lines.append("\n" + "-" * 60)
        lines.append("TOP 10 FILES (by chunk count)")
        lines.append("-" * 60)
        for file_path, count in list(stats["top_files"].items())[:10]:
            # Shorten path for display
            display_path = Path(file_path).name
            if len(file_path) > 50:
                display_path = "..." + file_path[-47:]
            lines.append(f"  {count:5,} chunks  {display_path}")

    # Author statistics
    if "unique_authors" in stats:
        lines.append("\n" + "-" * 60)
        lines.append("AUTHORS")
        lines.append("-" * 60)
        lines.append(f"Unique authors: {stats['unique_authors']}")
        if "authors" in stats and stats["authors"]:
            for author in stats["authors"][:10]:
                lines.append(f"  - {author}")

    # Ingestion time range
    if "first_ingested" in stats and "last_ingested" in stats:
        lines.append("\n" + "-" * 60)
        lines.append("INGESTION TIMELINE")
        lines.append("-" * 60)
        lines.append(f"First ingested: {stats['first_ingested']}")
        lines.append(f"Last ingested:  {stats['last_ingested']}")

    # Parse errors (from last ingest run)
    if "parse_errors" in stats:
        parse_errors = stats["parse_errors"]
        lines.append("\n" + "-" * 60)
        lines.append("PARSE ERRORS")
        lines.append("-" * 60)
        lines.append(f"Failed documents: {parse_errors.get('failed_documents', 0):,}")
        lines.append(f"Page errors:      {parse_errors.get('page_errors', 0):,}")
        if parse_errors.get("generated_at"):
            lines.append(f"Last report:      {parse_errors['generated_at']}")
        if parse_errors.get("path"):
            lines.append(f"Report path:      {parse_errors['path']}")
        sample = parse_errors.get("sample_failed_documents")
        if sample:
            lines.append("Sample failures:")
            for entry in sample:
                display_path = Path(entry).name
                if len(entry) > 50:
                    display_path = "..." + entry[-47:]
                lines.append(f"  - {display_path}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def search_metadata(
    collection: Collection,
    query: Optional[str] = None,
    author: Optional[str] = None,
    ext: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search collection metadata by various criteria.

    Args:
        collection: ChromaDB collection
        query: Text query for file paths
        author: Filter by author
        ext: Filter by extension
        limit: Maximum results to return

    Returns:
        List of matching metadata dictionaries
    """
    collection = require_chroma_collection(collection, "search_metadata")
    # Build where clause
    where = {}

    if author:
        where["author"] = {"$eq": author}

    if ext:
        where["ext"] = {"$eq": ext}

    try:
        # Get results with filters
        results = collection.get(
            where=where if where else None,
            limit=limit,
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        # If query provided, filter by path substring
        if query and metadatas:
            filtered = []
            for meta in metadatas:
                source_path = meta.get("source_path", "")
                if query.lower() in source_path.lower():
                    filtered.append(meta)
            return filtered[:limit]

        return metadatas

    except Exception as e:
        print(f"Search error: {e}")
        return []


def get_file_chunks(collection: Collection, file_path: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a specific file.

    Args:
        collection: ChromaDB collection
        file_path: Path to file

    Returns:
        List of chunk metadata for the file
    """
    collection = require_chroma_collection(collection, "get_file_chunks")
    try:
        results = collection.get(
            where={"source_path": {"$eq": file_path}},
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        # Combine documents with metadata
        chunks = []
        for doc, meta in zip(documents, metadatas):
            chunk_info = meta.copy()
            chunk_info["text"] = doc
            chunks.append(chunk_info)

        # Sort by chunk_index
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        return chunks

    except Exception as e:
        print(f"Error getting file chunks: {e}")
        return []
