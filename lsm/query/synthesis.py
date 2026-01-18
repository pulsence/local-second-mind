"""
Synthesis module for building context and formatting sources.

Provides utilities for building context blocks with citations and formatting source lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

from lsm.gui.shell.logging import get_logger
from .session import Candidate

logger = get_logger(__name__)


# -----------------------------
# Context Building
# -----------------------------
def build_context_block(candidates: List[Candidate]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build context block for LLM with source citations.

    Creates:
    - Context string for model input (sources prefixed [S1], [S2], ...)
    - Sources list for user display with metadata

    Args:
        candidates: List of candidates to include in context

    Returns:
        Tuple of (context_block, sources_list)
        - context_block: Formatted string with [S#] citations
        - sources_list: List of source metadata dicts

    Example:
        >>> context, sources = build_context_block(candidates)
        >>> print(context)
        [S1] /path/to/doc.md (name=doc.md, chunk_index=0)
        This is the first chunk text...

        [S2] /path/to/doc.md (name=doc.md, chunk_index=1)
        This is the second chunk text...
    """
    logger.debug(f"Building context block from {len(candidates)} candidates")

    sources_for_model: List[str] = []
    sources_for_user: List[Dict[str, Any]] = []

    for i, c in enumerate(candidates, start=1):
        label = f"S{i}"
        meta = c.meta or {}

        # Extract metadata
        source_path = meta.get("source_path", "unknown")
        source_name = meta.get("source_name")
        chunk_index = meta.get("chunk_index")
        ext = meta.get("ext")
        mtime_ns = meta.get("mtime_ns")
        file_hash = meta.get("file_hash")
        ingested_at = meta.get("ingested_at")
        title = meta.get("title")
        author = meta.get("author")

        # Build locator string with available metadata
        locator_bits = []
        if source_name:
            locator_bits.append(f"name={source_name}")
        if chunk_index is not None:
            locator_bits.append(f"chunk_index={chunk_index}")
        if ext:
            locator_bits.append(f"ext={ext}")
        if mtime_ns is not None:
            locator_bits.append(f"mtime_ns={mtime_ns}")
        if file_hash:
            locator_bits.append(f"file_hash={file_hash}")
        if ingested_at:
            locator_bits.append(f"ingested_at={ingested_at}")

        locator = " ".join(locator_bits)
        header = f"[{label}] {source_path}"
        if locator:
            header += f" ({locator})"

        # Add to model context
        sources_for_model.append(f"{header}\n{(c.text or '').strip()}\n")

        # Add to user sources list
        sources_for_user.append(
            {
                "label": label,
                "source_path": source_path,
                "source_name": source_name,
                "chunk_index": chunk_index,
                "ext": ext,
                "mtime_ns": mtime_ns,
                "file_hash": file_hash,
                "ingested_at": ingested_at,
                "title": title,
                "author": author,
            }
        )

    context_block = "\n\n".join(sources_for_model)

    logger.info(
        f"Built context block with {len(candidates)} sources "
        f"({len(context_block)} chars)"
    )

    return context_block, sources_for_user


# -----------------------------
# Fallback Answer Generation
# -----------------------------
def fallback_answer(
    question: str,
    candidates: List[Candidate],
    max_chars: int = 1200,
) -> str:
    """
    Generate minimal offline fallback answer.

    Returns top passages with citations when LLM is unavailable.
    This is not true synthesis, but useful when API is down.

    Args:
        question: User's question
        candidates: List of candidates to include
        max_chars: Maximum characters per excerpt

    Returns:
        Formatted fallback answer string

    Example:
        >>> answer = fallback_answer(
        ...     "What is Python?",
        ...     candidates,
        ...     max_chars=500
        ... )
    """
    logger.warning("Generating fallback answer (LLM unavailable)")

    lines = [
        "OpenAI is unavailable (quota/credentials). "
        "Showing the most relevant excerpts instead.",
        "",
        f"Question: {question}",
        "",
        "Top excerpts:",
    ]

    for i, c in enumerate(candidates, start=1):
        label = f"S{i}"
        excerpt = (c.text or "").strip()

        # Truncate long excerpts
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 50] + "\n...[truncated]..."

        meta = c.meta or {}
        source_path = meta.get("source_path", "unknown")
        chunk_index = meta.get("chunk_index", "NA")

        lines.append(
            f"\n[{label}] {source_path} (chunk_index={chunk_index})\n{excerpt}"
        )

    return "\n".join(lines)


# -----------------------------
# Source Formatting
# -----------------------------
def format_source_list(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources list for display.

    Groups sources by file and shows citation labels.

    Args:
        sources: List of source metadata dicts

    Returns:
        Formatted sources string

    Example:
        >>> sources = [
        ...     {"label": "S1", "source_path": "/docs/readme.md", "source_name": "readme.md"},
        ...     {"label": "S2", "source_path": "/docs/readme.md", "source_name": "readme.md"},
        ...     {"label": "S3", "source_path": "/docs/guide.md", "source_name": "guide.md"},
        ... ]
        >>> print(format_source_list(sources))
        Sources:
        - [S1] [S2] readme.md — /docs/readme.md
        - [S3] guide.md — /docs/guide.md
    """
    if not sources:
        return ""

    logger.debug(f"Formatting {len(sources)} sources for display")

    lines = ["", "Sources:"]
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    # Group sources by path
    for s in sources:
        path = (s.get("source_path") or "unknown").strip()
        label = (s.get("label") or "").strip()
        name = (s.get("source_name") or Path(path).name or "unknown").strip()

        if path not in grouped:
            grouped[path] = {
                "name": name,
                "labels": [],
            }
            order.append(path)

        if label and label not in grouped[path]["labels"]:
            grouped[path]["labels"].append(label)

    # Format grouped sources
    for path in order:
        entry = grouped[path]
        labels = " ".join(f"[{lbl}]" for lbl in entry["labels"])
        lines.append(f"- {labels} {entry['name']} — {path}")

    return "\n".join(lines)
