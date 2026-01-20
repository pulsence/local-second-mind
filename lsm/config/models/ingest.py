"""
Ingestion pipeline configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHROMA_FLUSH_INTERVAL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_DEVICE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXTENSIONS,
)


@dataclass
class IngestConfig:
    """
    Configuration for the ingestion pipeline.

    Controls how documents are discovered, parsed, chunked, and embedded.
    """

    roots: List[Path]
    """Root directories to scan for documents. Required."""

    embed_model: str = DEFAULT_EMBED_MODEL
    """Sentence-transformers model for embeddings."""

    device: str = DEFAULT_DEVICE
    """Device for embedding: 'cpu', 'cuda', 'cuda:0', etc."""

    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size for embedding operations."""

    persist_dir: Path = Path(".chroma")
    """Directory for ChromaDB persistent storage."""

    collection: str = DEFAULT_COLLECTION
    """ChromaDB collection name."""

    chroma_flush_interval: int = DEFAULT_CHROMA_FLUSH_INTERVAL
    """Number of chunks to buffer before flushing to ChromaDB."""

    manifest: Path = Path(".ingest/manifest.json")
    """Path to manifest file for incremental updates."""

    extensions: Optional[List[str]] = None
    """File extensions to process. If None, uses DEFAULT_EXTENSIONS."""

    override_extensions: bool = False
    """If True, replace default extensions. If False, merge with defaults."""

    exclude_dirs: Optional[List[str]] = None
    """Directory names to exclude. If None, uses DEFAULT_EXCLUDE_DIRS."""

    override_excludes: bool = False
    """If True, replace default excludes. If False, merge with defaults."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    """Character size for text chunks."""

    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    """Character overlap between consecutive chunks."""

    enable_ocr: bool = False
    """Enable OCR for image-based PDFs."""

    enable_ai_tagging: bool = False
    """Enable AI-powered chunk tagging."""

    tagging_model: str = "gpt-5.2"
    """Model to use for AI tagging if enabled."""

    tags_per_chunk: int = 3
    """Number of tags to generate per chunk if AI tagging enabled."""

    dry_run: bool = False
    """If True, simulate ingest without writing to database."""

    skip_errors: bool = True
    """If True, continue ingest when individual files/pages fail to parse."""

    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        if isinstance(self.roots, list):
            self.roots = [Path(r) if isinstance(r, str) else r for r in self.roots]

        if isinstance(self.persist_dir, str):
            self.persist_dir = Path(self.persist_dir)
        if isinstance(self.manifest, str):
            self.manifest = Path(self.manifest)

    @property
    def exts(self) -> Set[str]:
        """Get normalized set of file extensions."""
        result = set()

        if not self.override_extensions:
            result |= DEFAULT_EXTENSIONS

        if self.extensions:
            for ext in self.extensions:
                ext = str(ext).strip().lower()
                if not ext:
                    continue
                if not ext.startswith("."):
                    ext = "." + ext
                result.add(ext)

        return result

    @property
    def exclude_set(self) -> Set[str]:
        """Get normalized set of excluded directories."""
        result = set()

        if not self.override_excludes:
            result |= DEFAULT_EXCLUDE_DIRS

        if self.exclude_dirs:
            result |= {str(d).strip() for d in self.exclude_dirs if str(d).strip()}

        return result

    def validate(self) -> None:
        """Validate ingest configuration."""
        if not self.roots:
            raise ValueError("At least one root directory is required")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.chroma_flush_interval < 1:
            raise ValueError(
                f"chroma_flush_interval must be positive, got {self.chroma_flush_interval}"
            )

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
