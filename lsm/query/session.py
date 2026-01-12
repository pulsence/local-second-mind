"""
Session state management for query REPL.

Manages user session state, filters, and query artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Candidate:
    """
    A retrieved document chunk candidate.

    Represents a single chunk from the vector database with metadata.
    """

    cid: str
    """Chunk ID from ChromaDB."""

    text: str
    """Chunk text content."""

    meta: Dict[str, Any]
    """Metadata from ChromaDB (source_path, chunk_index, etc.)."""

    distance: Optional[float] = None
    """Distance score from vector search (lower is better)."""

    @property
    def source_path(self) -> str:
        """Get source file path from metadata."""
        return self.meta.get("source_path", "unknown")

    @property
    def source_name(self) -> str:
        """Get source file name from metadata."""
        return self.meta.get("source_name", "unknown")

    @property
    def chunk_index(self) -> int:
        """Get chunk index from metadata."""
        return self.meta.get("chunk_index", 0)

    @property
    def ext(self) -> str:
        """Get file extension from metadata."""
        return self.meta.get("ext", "")

    @property
    def relevance(self) -> float:
        """
        Convert distance to relevance score.

        Returns:
            Relevance score (1.0 - distance), clamped to [-1, 1]
        """
        if self.distance is None:
            return 0.0

        rel = 1.0 - self.distance

        # Clamp to reasonable range
        if rel > 1.0:
            return 1.0
        if rel < -1.0:
            return -1.0

        return rel


@dataclass
class SessionState:
    """
    REPL session state.

    Tracks user filters, model selection, and last query artifacts.
    """

    # Session-level filter overrides
    path_contains: Optional[Any] = None
    """Filter for files containing this string in path."""

    ext_allow: Optional[List[str]] = None
    """Only include files with these extensions."""

    ext_deny: Optional[List[str]] = None
    """Exclude files with these extensions."""

    # Model override (session)
    model: str = "gpt-5.2"
    """LLM model to use for this session."""

    available_models: Optional[List[str]] = None
    """List of available models (populated on demand)."""

    # Last-turn artifacts (for debugging and inspection)
    last_question: Optional[str] = None
    """Last question asked by user."""

    last_all_candidates: List[Candidate] = None
    """All candidates retrieved from vector DB."""

    last_filtered_candidates: List[Candidate] = None
    """Candidates after applying filters."""

    last_chosen: List[Candidate] = None
    """Final candidates used for answer generation."""

    last_label_to_candidate: Dict[str, Candidate] = None
    """Mapping from source labels ([S1], etc.) to candidates."""

    last_debug: Dict[str, Any] = None
    """Debug information from last query."""

    last_answer: Optional[str] = None
    """Last answer generated (for note saving)."""

    last_remote_sources: Optional[List[Dict[str, Any]]] = None
    """Last remote sources fetched (for note saving)."""

    last_local_sources_for_notes: Optional[List[Dict[str, Any]]] = None
    """Last local sources in dict format (for note saving)."""

    pinned_chunks: List[str] = None
    """Chunk IDs pinned for forced inclusion in context."""

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.available_models is None:
            self.available_models = []
        if self.last_all_candidates is None:
            self.last_all_candidates = []
        if self.last_filtered_candidates is None:
            self.last_filtered_candidates = []
        if self.last_chosen is None:
            self.last_chosen = []
        if self.last_label_to_candidate is None:
            self.last_label_to_candidate = {}
        if self.last_debug is None:
            self.last_debug = {}
        if self.pinned_chunks is None:
            self.pinned_chunks = []

    def clear_artifacts(self) -> None:
        """Clear last query artifacts."""
        self.last_question = None
        self.last_all_candidates = []
        self.last_filtered_candidates = []
        self.last_chosen = []
        self.last_label_to_candidate = {}
        self.last_debug = {}

    def has_filters(self) -> bool:
        """Check if any filters are active."""
        return bool(
            self.path_contains or
            self.ext_allow or
            self.ext_deny
        )

    def get_filter_summary(self) -> str:
        """
        Get a summary of active filters.

        Returns:
            Human-readable filter summary
        """
        if not self.has_filters():
            return "No filters active"

        parts = []

        if self.path_contains:
            if isinstance(self.path_contains, list):
                parts.append(f"Path contains: {', '.join(self.path_contains)}")
            else:
                parts.append(f"Path contains: {self.path_contains}")

        if self.ext_allow:
            parts.append(f"Extensions: {', '.join(self.ext_allow)}")

        if self.ext_deny:
            parts.append(f"Excluding: {', '.join(self.ext_deny)}")

        return "; ".join(parts)
