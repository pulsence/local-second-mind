"""
Configuration data models for Local Second Mind.

Provides typed dataclasses for all configuration options, with validation
and default values. Designed to be forward-compatible with multi-provider
LLM support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


# -----------------------------------------------------------------------------
# Default Values
# -----------------------------------------------------------------------------

DEFAULT_EXTENSIONS = {
    ".txt", ".md", ".rst",
    ".pdf",
    ".docx",
    ".html", ".htm",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__",
    ".venv", "venv",
    "node_modules",
}

DEFAULT_COLLECTION = "local_kb"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHROMA_FLUSH_INTERVAL = 2000

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200

# Query defaults
DEFAULT_K = 12
DEFAULT_K_RERANK = 6
DEFAULT_MAX_PER_FILE = 2
DEFAULT_MIN_RELEVANCE = 0.25


# -----------------------------------------------------------------------------
# LLM Configuration (Future-Ready for Multi-Provider)
# -----------------------------------------------------------------------------

@dataclass
class FeatureLLMConfig:
    """
    Optional LLM configuration override for a specific feature.

    Allows per-feature provider/model selection while inheriting from main config.
    """

    provider: Optional[str] = None
    """Override provider for this feature. If None, inherits from main LLM config."""

    model: Optional[str] = None
    """Override model for this feature. If None, inherits from main LLM config."""

    api_key: Optional[str] = None
    """Override API key for this feature. If None, inherits from main LLM config."""

    temperature: Optional[float] = None
    """Override temperature for this feature. If None, inherits from main LLM config."""

    max_tokens: Optional[int] = None
    """Override max_tokens for this feature. If None, inherits from main LLM config."""

    def merge_with_base(self, base: "LLMConfig") -> "LLMConfig":
        """
        Create a merged LLM config using this override and base config.

        Args:
            base: Base LLM configuration to inherit from

        Returns:
            New LLMConfig with overrides applied
        """
        return LLMConfig(
            provider=self.provider if self.provider is not None else base.provider,
            model=self.model if self.model is not None else base.model,
            api_key=self.api_key if self.api_key is not None else base.api_key,
            temperature=self.temperature if self.temperature is not None else base.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else base.max_tokens,
        )


@dataclass
class LLMConfig:
    """
    LLM provider configuration.

    Designed to support multiple providers in the future (OpenAI, Anthropic, local models).
    Currently focused on OpenAI.

    Supports per-feature overrides for Query, AI Tagging, and AI Ranking.
    """

    provider: str = "openai"
    """LLM provider name. Currently only 'openai' is supported."""

    model: str = "gpt-5.2"
    """Model name to use. For OpenAI: gpt-5.2, gpt-4, etc."""

    api_key: Optional[str] = None
    """
    API key for the provider.
    If None, will attempt to load from environment variable based on provider.
    """

    temperature: float = 0.7
    """Temperature for LLM generation (0.0 = deterministic, 1.0 = creative)."""

    max_tokens: int = 2000
    """Maximum tokens to generate in responses."""

    # Per-feature overrides
    query: Optional[FeatureLLMConfig] = None
    """Optional LLM config override for query/answer synthesis."""

    tagging: Optional[FeatureLLMConfig] = None
    """Optional LLM config override for AI tagging."""

    ranking: Optional[FeatureLLMConfig] = None
    """Optional LLM config override for AI-powered reranking."""

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if not self.api_key:
            # Try provider-specific environment variable
            env_var = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var)

            # Fallback to OPENAI_API_KEY for backward compatibility
            if not self.api_key and self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")

    def validate(self) -> None:
        """Validate LLM configuration."""
        if not self.api_key:
            raise ValueError(
                f"API key required for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_API_KEY environment variable or provide in config."
            )

        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

    def get_query_config(self) -> "LLMConfig":
        """Get effective LLM config for query operations."""
        if self.query:
            return self.query.merge_with_base(self)
        return self

    def get_tagging_config(self) -> "LLMConfig":
        """Get effective LLM config for AI tagging operations."""
        if self.tagging:
            return self.tagging.merge_with_base(self)
        return self

    def get_ranking_config(self) -> "LLMConfig":
        """Get effective LLM config for AI ranking operations."""
        if self.ranking:
            return self.ranking.merge_with_base(self)
        return self


# -----------------------------------------------------------------------------
# Ingest Configuration
# -----------------------------------------------------------------------------

@dataclass
class IngestConfig:
    """
    Configuration for the ingestion pipeline.

    Controls how documents are discovered, parsed, chunked, and embedded.
    """

    roots: List[Path]
    """Root directories to scan for documents. Required."""

    # Embedding configuration
    embed_model: str = DEFAULT_EMBED_MODEL
    """Sentence-transformers model for embeddings."""

    device: str = DEFAULT_DEVICE
    """Device for embedding: 'cpu', 'cuda', 'cuda:0', etc."""

    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size for embedding operations."""

    # Storage configuration
    persist_dir: Path = Path(".chroma")
    """Directory for ChromaDB persistent storage."""

    collection: str = DEFAULT_COLLECTION
    """ChromaDB collection name."""

    chroma_flush_interval: int = DEFAULT_CHROMA_FLUSH_INTERVAL
    """Number of chunks to buffer before flushing to ChromaDB."""

    manifest: Path = Path(".ingest/manifest.json")
    """Path to manifest file for incremental updates."""

    # File filtering
    extensions: Optional[List[str]] = None
    """File extensions to process. If None, uses DEFAULT_EXTENSIONS."""

    override_extensions: bool = False
    """If True, replace default extensions. If False, merge with defaults."""

    exclude_dirs: Optional[List[str]] = None
    """Directory names to exclude. If None, uses DEFAULT_EXCLUDE_DIRS."""

    override_excludes: bool = False
    """If True, replace default excludes. If False, merge with defaults."""

    # Chunking configuration
    chunk_size: int = DEFAULT_CHUNK_SIZE
    """Character size for text chunks."""

    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    """Character overlap between consecutive chunks."""

    # Advanced features (future)
    enable_ocr: bool = False
    """Enable OCR for image-based PDFs."""

    enable_ai_tagging: bool = False
    """Enable AI-powered chunk tagging."""

    tagging_model: str = "gpt-5.2"
    """Model to use for AI tagging if enabled."""

    tags_per_chunk: int = 3
    """Number of tags to generate per chunk if AI tagging enabled."""

    # Operational
    dry_run: bool = False
    """If True, simulate ingest without writing to database."""

    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        # Convert roots to Path objects
        if isinstance(self.roots, list):
            self.roots = [Path(r) if isinstance(r, str) else r for r in self.roots]

        # Convert other paths
        if isinstance(self.persist_dir, str):
            self.persist_dir = Path(self.persist_dir)
        if isinstance(self.manifest, str):
            self.manifest = Path(self.manifest)

    @property
    def exts(self) -> Set[str]:
        """Get normalized set of file extensions."""
        result = set()

        # Add defaults unless overridden
        if not self.override_extensions:
            result |= DEFAULT_EXTENSIONS

        # Add custom extensions
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

        # Add defaults unless overridden
        if not self.override_excludes:
            result |= DEFAULT_EXCLUDE_DIRS

        # Add custom excludes
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
            raise ValueError(f"chroma_flush_interval must be positive, got {self.chroma_flush_interval}")

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )


# -----------------------------------------------------------------------------
# Query Configuration
# -----------------------------------------------------------------------------

@dataclass
class QueryConfig:
    """
    Configuration for the query pipeline.

    Controls retrieval, reranking, and answer synthesis.
    """

    # Retrieval
    k: int = DEFAULT_K
    """Number of chunks to retrieve from vector database."""

    retrieve_k: Optional[int] = None
    """Override k for initial retrieval (useful with filters). If None, uses k."""

    min_relevance: float = DEFAULT_MIN_RELEVANCE
    """Minimum relevance score (1 - distance) to proceed with LLM calls."""

    # Reranking
    k_rerank: int = DEFAULT_K_RERANK
    """Number of chunks to keep after reranking."""

    rerank_strategy: str = "hybrid"
    """Reranking strategy: 'none', 'lexical', 'llm', 'hybrid'."""

    no_rerank: bool = False
    """If True, skip reranking entirely."""

    local_pool: Optional[int] = None
    """Pool size for local reranking. If None, computed from k and k_rerank."""

    # Diversity
    max_per_file: int = DEFAULT_MAX_PER_FILE
    """Maximum chunks from any single file in final results."""

    # Query mode
    mode: str = "grounded"
    """Query mode: 'grounded' (strict citations) or 'insight' (thematic analysis)."""

    # Filters (optional)
    path_contains: Optional[List[str]] = None
    """Filter to files whose path contains any of these strings."""

    ext_allow: Optional[List[str]] = None
    """Only include files with these extensions."""

    ext_deny: Optional[List[str]] = None
    """Exclude files with these extensions."""

    def __post_init__(self):
        """Compute derived values."""
        # Compute local_pool if not specified
        if self.local_pool is None:
            self.local_pool = max(self.k * 3, self.k_rerank * 4)

        # Handle no_rerank flag
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
            raise ValueError(f"min_relevance must be between 0.0 and 1.0, got {self.min_relevance}")

        valid_strategies = {"none", "lexical", "llm", "hybrid"}
        if self.rerank_strategy not in valid_strategies:
            raise ValueError(
                f"rerank_strategy must be one of {valid_strategies}, got '{self.rerank_strategy}'"
            )

        valid_modes = {"grounded", "insight"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")


# -----------------------------------------------------------------------------
# Top-Level Configuration
# -----------------------------------------------------------------------------

@dataclass
class LSMConfig:
    """
    Top-level configuration for Local Second Mind.

    Combines ingest, query, and LLM configurations.
    """

    ingest: IngestConfig
    """Ingest pipeline configuration."""

    query: QueryConfig
    """Query pipeline configuration."""

    llm: LLMConfig
    """LLM provider configuration."""

    config_path: Optional[Path] = None
    """Path to the config file (for resolving relative paths)."""

    def __post_init__(self):
        """Resolve relative paths and validate."""
        if self.config_path:
            # Resolve paths relative to config file location
            base_dir = self.config_path.parent

            # Resolve ingest paths
            if not self.ingest.persist_dir.is_absolute():
                self.ingest.persist_dir = (base_dir / self.ingest.persist_dir).resolve()

            if not self.ingest.manifest.is_absolute():
                self.ingest.manifest = (base_dir / self.ingest.manifest).resolve()

    def validate(self) -> None:
        """Validate entire configuration."""
        self.ingest.validate()
        self.query.validate()
        self.llm.validate()

    @property
    def persist_dir(self) -> Path:
        """Shortcut to ingest persist_dir."""
        return self.ingest.persist_dir

    @property
    def collection(self) -> str:
        """Shortcut to ingest collection name."""
        return self.ingest.collection

    @property
    def embed_model(self) -> str:
        """Shortcut to ingest embed_model."""
        return self.ingest.embed_model

    @property
    def device(self) -> str:
        """Shortcut to ingest device."""
        return self.ingest.device

    @property
    def batch_size(self) -> int:
        """Shortcut to ingest batch_size."""
        return self.ingest.batch_size
