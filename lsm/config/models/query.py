"""
Query pipeline configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .constants import DEFAULT_K, DEFAULT_MIN_RELEVANCE

VALID_PROFILES = (
    "dense_only",
    "hybrid_rrf",
    "hyde_hybrid",
    "dense_cross_rerank",
    "llm_rerank",
    "multi_vector",
)


@dataclass
class QueryConfig:
    """
    Configuration for the query pipeline.

    Controls retrieval profiles, reranking, and answer synthesis.
    """

    k: int = DEFAULT_K
    """Number of chunks to use in final context."""

    retrieve_k: Optional[int] = None
    """Override k for initial retrieval (useful with filters). If None, uses k."""

    min_relevance: float = DEFAULT_MIN_RELEVANCE
    """Minimum relevance score (1 - distance) to proceed with LLM calls."""

    retrieval_profile: str = "hybrid_rrf"
    """Retrieval profile: one of VALID_PROFILES."""

    k_dense: int = 100
    """Number of candidates from dense (vector) recall."""

    k_sparse: int = 100
    """Number of candidates from sparse (BM25) recall."""

    rrf_dense_weight: float = 0.7
    """Dense retrieval weight for RRF fusion."""

    rrf_sparse_weight: float = 0.3
    """Sparse retrieval weight for RRF fusion."""

    mode: str = "grounded"
    """Query mode: 'grounded' (strict citations) or 'insight' (thematic analysis)."""

    path_contains: Optional[List[str]] = None
    """Filter to files whose path contains any of these strings."""

    ext_allow: Optional[List[str]] = None
    """Only include files with these extensions."""

    ext_deny: Optional[List[str]] = None
    """Exclude files with these extensions."""

    chat_mode: str = "single"
    """Response mode: 'single' (stateless) or 'chat' (maintains conversation)."""

    enable_llm_server_cache: bool = True
    """Enable provider-side server caching for chat follow-up turns when supported."""

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """Cross-encoder model for dense_cross_rerank profile."""

    hyde_num_samples: int = 2
    """Number of hypothetical documents to generate for HyDE."""

    hyde_temperature: float = 0.2
    """Temperature for HyDE document generation."""

    hyde_pooling: str = "mean"
    """Embedding pooling strategy for HyDE: 'mean' or 'max'."""

    dedup_threshold: float = 0.8
    """MinHash Jaccard threshold for near-duplicate suppression."""

    mmr_lambda: float = 0.7
    """MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity."""

    max_per_section: Optional[int] = None
    """Max candidates per heading section group. None = no cap."""

    temporal_boost_enabled: bool = False
    """Apply recency boost to recently modified documents."""

    temporal_boost_days: int = 30
    """Window in days for temporal recency boost."""

    temporal_boost_factor: float = 1.5
    """Boost multiplier for recent documents."""

    cluster_enabled: bool = False
    """Enable cluster-filtered retrieval (requires `lsm cluster build` first)."""

    cluster_algorithm: str = "kmeans"
    """Clustering algorithm: 'kmeans' or 'hdbscan'."""

    cluster_k: int = 50
    """Number of clusters for k-means."""

    cluster_top_n: int = 5
    """Number of top clusters to search during retrieval."""

    graph_expansion_enabled: bool = False
    """Enable graph-augmented retrieval (expands candidates via knowledge graph)."""

    graph_expansion_hops: int = 2
    """Maximum hops for graph expansion traversal."""

    def __post_init__(self):
        """Compute derived values."""
        self.chat_mode = (self.chat_mode or "single").strip().lower()

    def validate(self) -> None:
        """Validate query configuration."""
        if self.k < 1:
            raise ValueError(f"k must be positive, got {self.k}")

        if self.min_relevance < 0.0 or self.min_relevance > 1.0:
            raise ValueError(
                f"min_relevance must be between 0.0 and 1.0, got {self.min_relevance}"
            )

        if self.retrieval_profile not in VALID_PROFILES:
            raise ValueError(
                f"retrieval_profile must be one of {VALID_PROFILES}, got '{self.retrieval_profile}'"
            )

        valid_chat_modes = {"single", "chat"}
        if self.chat_mode not in valid_chat_modes:
            raise ValueError(
                f"chat_mode must be one of {valid_chat_modes}, got '{self.chat_mode}'"
            )

        valid_cluster_algorithms = {"kmeans", "hdbscan"}
        if self.cluster_algorithm not in valid_cluster_algorithms:
            raise ValueError(
                f"cluster_algorithm must be one of {valid_cluster_algorithms}, "
                f"got '{self.cluster_algorithm}'"
            )

        if self.cluster_k < 2:
            raise ValueError(f"cluster_k must be >= 2, got {self.cluster_k}")

        if self.cluster_top_n < 1:
            raise ValueError(f"cluster_top_n must be >= 1, got {self.cluster_top_n}")

        if self.graph_expansion_hops < 1:
            raise ValueError(
                f"graph_expansion_hops must be >= 1, got {self.graph_expansion_hops}"
            )
