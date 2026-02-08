"""
Query mode and source policy configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .constants import DEFAULT_K, DEFAULT_K_RERANK, DEFAULT_MIN_RELEVANCE


@dataclass
class LocalSourcePolicy:
    """
    Configuration for local knowledge base retrieval.

    Controls how chunks are retrieved from the local ChromaDB collection.
    """

    enabled: bool = True
    """Whether local sources are enabled for this mode."""

    min_relevance: float = DEFAULT_MIN_RELEVANCE
    """Minimum relevance score (1 - distance) to include chunks."""

    k: int = DEFAULT_K
    """Number of chunks to retrieve."""

    k_rerank: int = DEFAULT_K_RERANK
    """Number of chunks to keep after reranking."""


@dataclass
class RemoteProviderRef:
    """
    Reference to a remote provider with optional weight override.

    Used in mode configurations to specify provider-specific weights
    that override the global provider weight.
    """

    source: str
    """Name of the remote provider (must match a configured provider)."""

    weight: Optional[float] = None
    """Optional weight override for this mode (0.0-1.0). If None, uses global weight."""

    def validate(self) -> None:
        """Validate provider reference."""
        if self.weight is not None and self.weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


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

    remote_providers: Optional[List[Union[str, RemoteProviderRef]]] = None
    """Optional list of remote provider names or refs to use in this mode.

    Supports two formats:
    - String list: ["brave", "wikipedia", "arxiv"]
    - Inline weights: [{"source": "brave", "weight": 0.6}, {"source": "arxiv", "weight": 0.9}]
    - Mixed: ["brave", {"source": "arxiv", "weight": 0.9}]
    """

    def get_provider_names(self) -> List[str]:
        """Get list of provider names from refs."""
        if not self.remote_providers:
            return []
        names = []
        for ref in self.remote_providers:
            if isinstance(ref, str):
                names.append(ref)
            elif isinstance(ref, RemoteProviderRef):
                names.append(ref.source)
            elif isinstance(ref, dict):
                names.append(ref.get("source", ""))
        return names

    def get_provider_weight(self, provider_name: str) -> Optional[float]:
        """Get mode-specific weight override for a provider."""
        if not self.remote_providers:
            return None
        for ref in self.remote_providers:
            if isinstance(ref, str):
                if ref.lower() == provider_name.lower():
                    return None  # Use global weight
            elif isinstance(ref, RemoteProviderRef):
                if ref.source.lower() == provider_name.lower():
                    return ref.weight
            elif isinstance(ref, dict):
                if ref.get("source", "").lower() == provider_name.lower():
                    return ref.get("weight")
        return None


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
    Global configuration for automatic notes writing.
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
class ModeChatsConfig:
    """
    Per-mode overrides for chat transcript behavior.
    """

    auto_save: Optional[bool] = None
    """Override global chats.auto_save when set."""

    dir: Optional[str] = None
    """Override global chats.dir for this mode when set."""

    def __post_init__(self) -> None:
        if self.dir is not None:
            normalized = str(self.dir).strip()
            self.dir = normalized or None

    def validate(self) -> None:
        """Validate mode chat overrides."""
        if self.dir is not None and not self.dir.strip():
            raise ValueError("mode chats.dir cannot be empty when provided")


@dataclass
class ModeConfig:
    """
    Complete configuration for a query mode.

    Defines synthesis style and source policies for a named mode.
    """

    synthesis_style: str = "grounded"
    """Synthesis style: 'grounded' (strict citations) or 'insight' (thematic analysis)."""

    source_policy: SourcePolicyConfig = field(default_factory=SourcePolicyConfig)
    """Source policy configuration."""

    chats: Optional[ModeChatsConfig] = None
    """Optional per-mode chat save overrides."""

    def validate(self) -> None:
        """Validate mode configuration."""
        valid_styles = {"grounded", "insight"}
        if self.synthesis_style not in valid_styles:
            raise ValueError(
                f"synthesis_style must be one of {valid_styles}, got '{self.synthesis_style}'"
            )
        if self.chats is not None:
            self.chats.validate()


@dataclass
class RemoteProviderConfig:
    """
    Configuration for a remote source provider (e.g., web search, API).

    Defines how to connect to and weight a remote source provider.
    """

    name: str
    """Provider name used for display and selection."""

    type: str
    """Provider type: 'web_search', 'api', etc."""

    weight: float = 1.0
    """Weight for ranking/ordering this provider's results."""

    api_key: Optional[str] = None
    """API key if required by the provider."""

    endpoint: Optional[str] = None
    """Custom endpoint URL if applicable."""

    max_results: Optional[int] = None
    """Override max results for this provider."""

    language: Optional[str] = None
    """Language code for providers that support localization."""

    user_agent: Optional[str] = None
    """User-Agent header for HTTP requests."""

    timeout: Optional[int] = None
    """Request timeout in seconds."""

    min_interval_seconds: Optional[float] = None
    """Minimum seconds between requests."""

    section_limit: Optional[int] = None
    """Max sections to include in extracted snippets."""

    snippet_max_chars: Optional[int] = None
    """Max characters to include in extracted snippets."""

    include_disambiguation: Optional[bool] = None
    """Include disambiguation pages when supported."""

    def validate(self) -> None:
        """Validate provider configuration."""
        if self.weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
