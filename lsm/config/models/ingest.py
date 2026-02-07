"""
Ingestion pipeline configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .constants import (
    DEFAULT_CHROMA_FLUSH_INTERVAL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXTENSIONS,
)


@dataclass
class RootConfig:
    """Configuration for a single ingest root directory.

    Args:
        path: Root directory path to scan for documents.
        tags: Optional tags to propagate to all chunks from this root.
        content_type: Optional content type label for chunks from this root.
    """

    path: Path
    """Root directory path."""

    tags: Optional[List[str]] = None
    """Tags to propagate to all chunks ingested from this root."""

    content_type: Optional[str] = None
    """Content type label for documents under this root."""

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)


@dataclass
class IngestConfig:
    """
    Configuration for the ingestion pipeline.

    Controls how documents are discovered, parsed, chunked, and embedded.
    """

    roots: List[RootConfig]
    """Root directories to scan for documents. Required.

    Accepts strings, Path objects, dicts with ``path``/``tags``/``content_type``
    keys, or ``RootConfig`` instances. All are normalized to ``RootConfig`` in
    ``__post_init__``.
    """

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

    chunking_strategy: str = "structure"
    """Chunking strategy: 'structure' (heading/paragraph/sentence-aware) or
    'fixed' (simple character-based sliding window)."""

    enable_ocr: bool = False
    """Enable OCR for image-based PDFs."""

    enable_ai_tagging: bool = False
    """Enable AI-powered chunk tagging."""

    tags_per_chunk: int = 3
    """Number of tags to generate per chunk if AI tagging enabled."""

    dry_run: bool = False
    """If True, simulate ingest without writing to database."""

    skip_errors: bool = True
    """If True, continue ingest when individual files/pages fail to parse."""

    enable_language_detection: bool = False
    """Enable automatic language detection for ingested documents."""

    enable_translation: bool = False
    """Enable LLM-based translation of non-target-language chunks."""

    translation_target: str = "en"
    """Target language for translation (ISO 639-1 code)."""

    def __post_init__(self) -> None:
        """Normalize roots to RootConfig and convert string paths."""
        if isinstance(self.roots, list):
            normalized: List[RootConfig] = []
            for r in self.roots:
                if isinstance(r, RootConfig):
                    normalized.append(r)
                elif isinstance(r, dict):
                    normalized.append(
                        RootConfig(
                            path=Path(r["path"]),
                            tags=r.get("tags"),
                            content_type=r.get("content_type"),
                        )
                    )
                elif isinstance(r, (str, Path)):
                    normalized.append(RootConfig(path=Path(r)))
                else:
                    normalized.append(RootConfig(path=Path(r)))
            self.roots = normalized

        if isinstance(self.persist_dir, str):
            self.persist_dir = Path(self.persist_dir)
        if isinstance(self.manifest, str):
            self.manifest = Path(self.manifest)

    @property
    def root_paths(self) -> List[Path]:
        """Get list of root directory paths (convenience accessor)."""
        return [rc.path for rc in self.roots]

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

        valid_strategies = {"structure", "fixed"}
        if self.chunking_strategy not in valid_strategies:
            raise ValueError(
                f"chunking_strategy must be one of {valid_strategies}, "
                f"got '{self.chunking_strategy}'"
            )

        if self.enable_translation and not self.enable_language_detection:
            raise ValueError(
                "enable_language_detection must be True when "
                "enable_translation is True"
            )

        if self.enable_translation and not self.translation_target:
            raise ValueError(
                "translation_target is required when enable_translation is True"
            )
