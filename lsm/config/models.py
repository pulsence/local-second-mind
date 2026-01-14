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
DEFAULT_VDB_PROVIDER = "chromadb"
DEFAULT_CHROMA_HNSW_SPACE = "cosine"

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

    base_url: Optional[str] = None
    """Override base URL (for local/hosted providers)."""

    endpoint: Optional[str] = None
    """Override endpoint URL (e.g., Azure OpenAI)."""

    api_version: Optional[str] = None
    """Override API version (e.g., Azure OpenAI)."""

    deployment_name: Optional[str] = None
    """Override deployment name (e.g., Azure OpenAI)."""

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
            base_url=self.base_url if self.base_url is not None else base.base_url,
            endpoint=self.endpoint if self.endpoint is not None else base.endpoint,
            api_version=self.api_version if self.api_version is not None else base.api_version,
            deployment_name=(
                self.deployment_name if self.deployment_name is not None else base.deployment_name
            ),
        )


@dataclass
class LLMConfig:
    """
    LLM provider configuration.

    Supports multiple providers (OpenAI, Anthropic, Gemini, local models, Azure OpenAI).

    Supports per-feature overrides for Query, AI Tagging, and AI Ranking.
    """

    provider: str = "openai"
    """LLM provider name."""

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

    base_url: Optional[str] = None
    """Base URL for local or hosted providers (e.g., Ollama)."""

    endpoint: Optional[str] = None
    """Provider endpoint URL (e.g., Azure OpenAI resource endpoint)."""

    api_version: Optional[str] = None
    """Provider API version (e.g., Azure OpenAI)."""

    deployment_name: Optional[str] = None
    """Provider deployment name (e.g., Azure OpenAI deployment)."""

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
            if self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            else:
                env_var = f"{self.provider.upper()}_API_KEY"
                self.api_key = os.getenv(env_var)

            # Fallback to OPENAI_API_KEY for backward compatibility
            if not self.api_key and self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.base_url:
            # Local provider defaults
            if self.provider in {"local", "ollama"}:
                self.base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            else:
                self.base_url = os.getenv("LLM_BASE_URL")

        if not self.endpoint and self.provider == "azure_openai":
            self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not self.api_version and self.provider == "azure_openai":
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not self.deployment_name and self.provider == "azure_openai":
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def validate(self) -> None:
        """Validate LLM configuration."""
        if self.provider not in {"local", "ollama"} and not self.api_key:
            raise ValueError(
                f"API key required for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_API_KEY environment variable or provide in config."
            )

        if self.provider == "azure_openai":
            if not self.endpoint:
                raise ValueError(
                    "Azure OpenAI requires 'endpoint'. "
                    "Set llm.endpoint or AZURE_OPENAI_ENDPOINT."
                )
            if not self.api_version:
                raise ValueError(
                    "Azure OpenAI requires 'api_version'. "
                    "Set llm.api_version or AZURE_OPENAI_API_VERSION."
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

    skip_errors: bool = True
    """If True, continue ingest when individual files/pages fail to parse."""

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
# Vector DB Configuration
# -----------------------------------------------------------------------------

@dataclass
class VectorDBConfig:
    """
    Configuration for vector database providers.
    """

    provider: str = DEFAULT_VDB_PROVIDER
    """Vector DB provider name (e.g., 'chromadb', 'postgresql')."""

    collection: str = DEFAULT_COLLECTION
    """Collection or namespace name."""

    persist_dir: Path = Path(".chroma")
    """ChromaDB persistence directory (Chroma-only)."""

    chroma_hnsw_space: str = DEFAULT_CHROMA_HNSW_SPACE
    """ChromaDB HNSW space (e.g., 'cosine')."""

    # PostgreSQL/pgvector configuration
    connection_string: Optional[str] = None
    """PostgreSQL connection string."""

    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    index_type: str = "hnsw"
    """Index type for pgvector (e.g., 'hnsw', 'ivfflat')."""

    pool_size: int = 5
    """Connection pool size for providers that support pooling."""

    def __post_init__(self):
        if isinstance(self.persist_dir, str):
            self.persist_dir = Path(self.persist_dir)

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("vectordb.provider must be set")

        if self.provider == "chromadb":
            if not self.persist_dir:
                raise ValueError("vectordb.persist_dir is required for ChromaDB")
            if not self.collection:
                raise ValueError("vectordb.collection is required for ChromaDB")

        if self.provider == "postgresql":
            if not (self.connection_string or (self.host and self.database and self.user)):
                raise ValueError(
                    "PostgreSQL vectordb requires connection_string or host/database/user"
                )


# -----------------------------------------------------------------------------
# Mode System Configuration
# -----------------------------------------------------------------------------

@dataclass
class LocalSourcePolicy:
    """
    Configuration for local knowledge base retrieval.

    Controls how chunks are retrieved from the local ChromaDB collection.
    """

    min_relevance: float = DEFAULT_MIN_RELEVANCE
    """Minimum relevance score (1 - distance) to include chunks."""

    k: int = DEFAULT_K
    """Number of chunks to retrieve."""

    k_rerank: int = DEFAULT_K_RERANK
    """Number of chunks to keep after reranking."""


@dataclass
class RemoteSourcePolicy:
    """
    Configuration for remote source retrieval (web search, APIs, etc).

    Controls whether and how remote sources are fetched and ranked.
    """

    enabled: bool = False
    """Whether remote sources are enabled for this mode."""

    rank_strategy: str = "weighted"
    """How to rank remote sources: 'weighted', 'sequential', 'interleaved'."""

    max_results: int = 5
    """Maximum number of remote results to fetch."""


@dataclass
class ModelKnowledgePolicy:
    """
    Configuration for using LLM's built-in knowledge.

    Controls whether the LLM can use its training knowledge in synthesis.
    """

    enabled: bool = False
    """Whether model knowledge is allowed."""

    require_label: bool = True
    """If True, model knowledge must be explicitly labeled in the answer."""


@dataclass
class SourcePolicyConfig:
    """
    Complete source policy configuration for a query mode.

    Defines what sources (local, remote, model) are available and how to use them.
    """

    local: LocalSourcePolicy = field(default_factory=LocalSourcePolicy)
    """Local knowledge base retrieval policy."""

    remote: RemoteSourcePolicy = field(default_factory=RemoteSourcePolicy)
    """Remote source retrieval policy."""

    model_knowledge: ModelKnowledgePolicy = field(default_factory=ModelKnowledgePolicy)
    """Model knowledge usage policy."""


@dataclass
class NotesConfig:
    """
    Configuration for automatic notes writing.

    Controls whether and how query sessions are saved as Markdown notes.
    """

    enabled: bool = True
    """Whether to write notes for queries."""

    dir: str = "notes"
    """Directory for notes (relative to config file or absolute)."""

    template: str = "default"
    """Template to use for notes formatting."""

    filename_format: str = "timestamp"
    """Filename format: 'timestamp', 'query_slug', 'incremental'."""

    integration: str = "none"
    """Note integration target: 'none', 'obsidian', or 'logseq'."""

    wikilinks: bool = False
    """Use [[wikilink]] formatting for source references."""

    backlinks: bool = False
    """Include a backlinks section with source wikilinks."""

    include_tags: bool = False
    """Include tags derived from source metadata in the note header."""


@dataclass
class ModeConfig:
    """
    Complete configuration for a query mode.

    Defines synthesis style, source policies, and notes behavior for a named mode.
    """

    synthesis_style: str = "grounded"
    """Synthesis style: 'grounded' (strict citations) or 'insight' (thematic analysis)."""

    source_policy: SourcePolicyConfig = field(default_factory=SourcePolicyConfig)
    """Source policy configuration."""

    notes: NotesConfig = field(default_factory=NotesConfig)
    """Notes configuration."""

    def validate(self) -> None:
        """Validate mode configuration."""
        valid_styles = {"grounded", "insight"}
        if self.synthesis_style not in valid_styles:
            raise ValueError(
                f"synthesis_style must be one of {valid_styles}, got '{self.synthesis_style}'"
            )


@dataclass
class RemoteProviderConfig:
    """
    Configuration for a remote source provider (e.g., web search, API).

    Defines how to connect to and weight a remote source provider.
    """

    type: str
    """Provider type: 'web_search', 'api', etc."""

    enabled: bool = True
    """Whether this provider is active."""

    weight: float = 1.0
    """Weight for ranking/ordering this provider's results."""

    # Provider-specific settings (extensible)
    api_key: Optional[str] = None
    """API key if required by the provider."""

    endpoint: Optional[str] = None
    """Custom endpoint URL if applicable."""

    max_results: Optional[int] = None
    """Override max results for this provider."""

    def validate(self) -> None:
        """Validate provider configuration."""
        if self.weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


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

        # Note: Mode validation happens at LSMConfig level since modes are dynamic


# -----------------------------------------------------------------------------
# Top-Level Configuration
# -----------------------------------------------------------------------------

@dataclass
class LSMConfig:
    """
    Top-level configuration for Local Second Mind.

    Combines ingest, query, LLM, and mode configurations.
    """

    ingest: IngestConfig
    """Ingest pipeline configuration."""

    query: QueryConfig
    """Query pipeline configuration."""

    llm: LLMConfig
    """LLM provider configuration."""

    vectordb: VectorDBConfig
    """Vector DB provider configuration."""

    modes: Optional[dict[str, ModeConfig]] = None
    """Registry of available query modes. If None, uses built-in defaults."""

    remote_providers: Optional[dict[str, RemoteProviderConfig]] = None
    """Registry of remote source providers (web search, APIs, etc)."""

    config_path: Optional[Path] = None
    """Path to the config file (for resolving relative paths)."""

    def __post_init__(self):
        """Resolve relative paths and initialize defaults."""
        if self.config_path:
            # Resolve paths relative to config file location
            base_dir = self.config_path.parent

            # Resolve ingest paths
            if not self.ingest.persist_dir.is_absolute():
                self.ingest.persist_dir = (base_dir / self.ingest.persist_dir).resolve()

            if not self.ingest.manifest.is_absolute():
                self.ingest.manifest = (base_dir / self.ingest.manifest).resolve()

            if self.vectordb and not self.vectordb.persist_dir.is_absolute():
                self.vectordb.persist_dir = (base_dir / self.vectordb.persist_dir).resolve()

        # Initialize built-in modes if not provided
        if self.modes is None:
            self.modes = self._get_builtin_modes()

    def validate(self) -> None:
        """Validate entire configuration."""
        self.ingest.validate()
        self.query.validate()
        self.llm.validate()
        self.vectordb.validate()

        # Validate mode registry
        if self.modes:
            for mode_name, mode_config in self.modes.items():
                mode_config.validate()

        # Validate that query.mode references a valid mode
        if self.query.mode not in self.modes:
            raise ValueError(
                f"query.mode '{self.query.mode}' not found in modes registry. "
                f"Available modes: {list(self.modes.keys())}"
            )

        # Validate remote providers
        if self.remote_providers:
            for provider_name, provider_config in self.remote_providers.items():
                provider_config.validate()

    @staticmethod
    def _get_builtin_modes() -> dict[str, ModeConfig]:
        """
        Get built-in default query modes.

        Returns:
            Dictionary of mode name to ModeConfig
        """
        return {
            "grounded": ModeConfig(
                synthesis_style="grounded",
                source_policy=SourcePolicyConfig(
                    local=LocalSourcePolicy(
                        min_relevance=0.25,
                        k=12,
                        k_rerank=6,
                    ),
                    remote=RemoteSourcePolicy(enabled=False),
                    model_knowledge=ModelKnowledgePolicy(enabled=False),
                ),
                notes=NotesConfig(
                    enabled=True,
                    dir="notes",
                    template="default",
                ),
            ),
            "insight": ModeConfig(
                synthesis_style="insight",
                source_policy=SourcePolicyConfig(
                    local=LocalSourcePolicy(
                        min_relevance=0.20,
                        k=14,
                        k_rerank=8,
                    ),
                    remote=RemoteSourcePolicy(enabled=False),
                    model_knowledge=ModelKnowledgePolicy(
                        enabled=True,
                        require_label=True,
                    ),
                ),
                notes=NotesConfig(
                    enabled=True,
                    dir="notes",
                    template="default",
                ),
            ),
            "hybrid": ModeConfig(
                synthesis_style="grounded",
                source_policy=SourcePolicyConfig(
                    local=LocalSourcePolicy(
                        min_relevance=0.25,
                        k=12,
                        k_rerank=6,
                    ),
                    remote=RemoteSourcePolicy(
                        enabled=True,
                        rank_strategy="weighted",
                        max_results=5,
                    ),
                    model_knowledge=ModelKnowledgePolicy(
                        enabled=True,
                        require_label=True,
                    ),
                ),
                notes=NotesConfig(
                    enabled=True,
                    dir="notes",
                    template="default",
                ),
            ),
        }

    def get_mode_config(self, mode_name: Optional[str] = None) -> ModeConfig:
        """
        Get the effective mode configuration.

        Args:
            mode_name: Mode name to look up. If None, uses query.mode.

        Returns:
            ModeConfig for the specified mode

        Raises:
            ValueError: If mode_name is not found in modes registry
        """
        name = mode_name or self.query.mode

        if name not in self.modes:
            raise ValueError(
                f"Mode '{name}' not found in modes registry. "
                f"Available modes: {list(self.modes.keys())}"
            )

        return self.modes[name]

    def get_active_remote_providers(self) -> dict[str, RemoteProviderConfig]:
        """
        Get all enabled remote providers.

        Returns:
            Dictionary of enabled remote providers
        """
        if not self.remote_providers:
            return {}

        return {
            name: config
            for name, config in self.remote_providers.items()
            if config.enabled
        }

    @property
    def persist_dir(self) -> Path:
        """Shortcut to vectordb persist_dir."""
        return self.vectordb.persist_dir

    @property
    def collection(self) -> str:
        """Shortcut to vectordb collection name."""
        return self.vectordb.collection

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
