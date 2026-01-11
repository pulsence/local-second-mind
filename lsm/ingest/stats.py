"""
Collection statistics and introspection utilities for ingest pipeline.

Provides functions to inspect ChromaDB collections, analyze metadata,
and generate reports about the ingested knowledge base.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter, defaultdict
from datetime import datetime

import chromadb
from chromadb.api.models.Collection import Collection


def get_collection_info(collection: Collection) -> Dict[str, Any]:
    """
    Get basic information about a collection.

    Args:
        collection: ChromaDB collection

    Returns:
        Dictionary with collection information
    """
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


def get_collection_stats(collection: Collection, limit: int = 10000) -> Dict[str, Any]:
    """
    Get detailed statistics about a collection.

    Args:
        collection: ChromaDB collection
        limit: Maximum number of chunks to analyze (for performance)

    Returns:
        Dictionary with detailed statistics
    """
    count = collection.count()

    if count == 0:
        return {
            "total_chunks": 0,
            "message": "Collection is empty",
        }

    # Get a sample of chunks for analysis
    sample_size = min(count, limit)

    try:
        # Query for a sample (use get with limit)
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

        # Analyze metadata
        stats = analyze_metadata(metadatas, count)
        stats["total_chunks"] = count
        stats["analyzed_chunks"] = len(metadatas)

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
        # Extension
        ext = meta.get("ext", "unknown")
        ext_counter[ext] += 1

        # Source file
        source_path = meta.get("source_path", "unknown")
        file_counter[source_path] += 1
        unique_files.add(source_path)
        chunks_per_file[source_path] += 1

        # Author (if present)
        if "author" in meta and meta["author"]:
            authors.add(meta["author"])

        # Title (if present)
        if "title" in meta and meta["title"]:
            titles.add(meta["title"])

        # Ingestion date
        if "ingested_at" in meta:
            ingestion_dates.append(meta["ingested_at"])

    # Compute statistics
    stats = {
        "unique_files": len(unique_files),
        "file_types": dict(ext_counter.most_common()),
        "top_files": dict(file_counter.most_common(10)),
        "avg_chunks_per_file": sum(chunks_per_file.values()) / len(chunks_per_file) if chunks_per_file else 0,
        "max_chunks_per_file": max(chunks_per_file.values()) if chunks_per_file else 0,
        "min_chunks_per_file": min(chunks_per_file.values()) if chunks_per_file else 0,
    }

    # Add author/title stats if available
    if authors:
        stats["unique_authors"] = len(authors)
        stats["authors"] = sorted(list(authors))[:20]  # Top 20

    if titles:
        stats["unique_titles"] = len(titles)

    # Add ingestion time range if available
    if ingestion_dates:
        sorted_dates = sorted(ingestion_dates)
        stats["first_ingested"] = sorted_dates[0]
        stats["last_ingested"] = sorted_dates[-1]

    return stats


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
