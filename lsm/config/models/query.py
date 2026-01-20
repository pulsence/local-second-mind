"""
Query pipeline configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .constants import DEFAULT_K, DEFAULT_K_RERANK, DEFAULT_MAX_PER_FILE, DEFAULT_MIN_RELEVANCE


@dataclass
class QueryConfig:
    """
    Configuration for the query pipeline.

    Controls retrieval, reranking, and answer synthesis.
    """

    k: int = DEFAULT_K
    """Number of chunks to retrieve from vector database."""

    retrieve_k: Optional[int] = None
    """Override k for initial retrieval (useful with filters). If None, uses k."""

    min_relevance: float = DEFAULT_MIN_RELEVANCE
    """Minimum relevance score (1 - distance) to proceed with LLM calls."""

    k_rerank: int = DEFAULT_K_RERANK
    """Number of chunks to keep after reranking."""

    rerank_strategy: str = "hybrid"
    """Reranking strategy: 'none', 'lexical', 'llm', 'hybrid'."""

    no_rerank: bool = False
    """If True, skip reranking entirely."""

    local_pool: Optional[int] = None
    """Pool size for local reranking. If None, computed from k and k_rerank."""

    max_per_file: int = DEFAULT_MAX_PER_FILE
    """Maximum chunks from any single file in final results."""

    mode: str = "grounded"
    """Query mode: 'grounded' (strict citations) or 'insight' (thematic analysis)."""

    path_contains: Optional[List[str]] = None
    """Filter to files whose path contains any of these strings."""

    ext_allow: Optional[List[str]] = None
    """Only include files with these extensions."""

    ext_deny: Optional[List[str]] = None
    """Exclude files with these extensions."""

    def __post_init__(self):
        """Compute derived values."""
        if self.local_pool is None:
            self.local_pool = max(self.k * 3, self.k_rerank * 4)

        if self.no_rerank:
            self.rerank_strategy = "none"

    def validate(self) -> None:
        """Validate query configuration."""
        if self.k < 1:
            raise ValueError(f"k must be positive, got {self.k}")

        if self.k_rerank < 1:
            raise ValueError(f"k_rerank must be positive, got {self.k_rerank}")

        if self.max_per_file < 1:
            raise ValueError(f"max_per_file must be positive, got {self.max_per_file}")

        if self.min_relevance < 0.0 or self.min_relevance > 1.0:
            raise ValueError(
                f"min_relevance must be between 0.0 and 1.0, got {self.min_relevance}"
            )

        valid_strategies = {"none", "lexical", "llm", "hybrid"}
        if self.rerank_strategy not in valid_strategies:
            raise ValueError(
                f"rerank_strategy must be one of {valid_strategies}, got '{self.rerank_strategy}'"
            )
