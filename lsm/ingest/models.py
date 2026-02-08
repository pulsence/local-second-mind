from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


# -----------------------------
# Page-Level Data
# -----------------------------
@dataclass
class PageSegment:
    """A segment of text from a single page of a document.

    Used by parsers that support page-level extraction (PDF, DOCX) to preserve
    page boundary information through the chunking pipeline.
    """

    text: str
    """Text content of this page segment."""

    page_number: int
    """1-based page number this segment belongs to."""


# -----------------------------
# Threading Classes
# -----------------------------
@dataclass
class ParseResult:
    """Result of parsing a single file."""

    source_path: str
    """Full path to source file."""

    fp: Path
    """Path object for source file."""

    mtime_ns: int
    """Modified time in nanoseconds."""

    size: int
    """File size in bytes."""

    file_hash: str
    """SHA-256 hash of file content."""

    chunks: List[str]
    """List of text chunks extracted from file."""

    ext: str
    """File extension (e.g., '.pdf', '.md')."""

    had_prev: bool
    """Whether manifest had an entry (controls delete(where))."""

    ok: bool
    """Whether parsing succeeded."""

    err: Optional[str] = None
    """Error message if parsing failed."""

    metadata: Optional[Dict[str, Any]] = None
    """Document metadata (author, title, etc.)."""

    chunk_positions: Optional[List[Dict[str, Any]]] = None
    """Position information for each chunk (offsets, page numbers, etc.)."""

    parse_errors: Optional[List[Dict[str, Any]]] = None
    """Per-page parse errors captured during extraction."""

    page_segments: Optional[List[PageSegment]] = None
    """Page-level text segments for page number tracking (PDF, DOCX)."""

    version: int = 1
    """Chunk version number for versioning support."""


@dataclass
class WriteJob:
    """
    One job corresponds to one file (so writer can delete/manifest-update per file).
    """

    source_path: str
    """Full path to source file."""

    fp: Path
    """Path object for source file."""

    mtime_ns: int
    """Modified time in nanoseconds."""

    size: int
    """File size in bytes."""

    file_hash: str
    """SHA-256 hash of file content."""

    ext: str
    """File extension (e.g., '.pdf', '.md')."""

    chunks: List[str]
    """List of text chunks."""

    embeddings: List[List[float]]
    """List of embedding vectors (one per chunk)."""

    had_prev: bool
    """Whether this file was previously ingested."""

    metadata: Optional[Dict[str, Any]] = None
    """Document metadata (author, title, etc.)."""

    chunk_positions: Optional[List[Dict[str, Any]]] = None
    """Position information for each chunk (offsets, page numbers, etc.)."""

    version: int = 1
    """Chunk version number for versioning support."""
