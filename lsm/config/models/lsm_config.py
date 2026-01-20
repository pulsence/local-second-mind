"""
Top-level configuration model for Local Second Mind.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

from .ingest import IngestConfig
from .llm import LLMRegistryConfig
from .modes import (
    LocalSourcePolicy,
    ModeConfig,
    ModelKnowledgePolicy,
    NotesConfig,
    RemoteProviderConfig,
    RemoteSourcePolicy,
    SourcePolicyConfig,
)
from .query import QueryConfig
from .vectordb import VectorDBConfig


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

    llm: LLMRegistryConfig
    """Ordered LLM provider configuration registry."""

    vectordb: VectorDBConfig
    """Vector DB provider configuration."""

    modes: Optional[dict[str, ModeConfig]] = None
    """Registry of available query modes. If None, uses built-in defaults."""

    remote_providers: Optional[list[RemoteProviderConfig]] = None
    """List of remote source providers (web search, APIs, etc)."""

    config_path: Optional[Path] = None
    """Path to the config file (for resolving relative paths)."""

    def __post_init__(self):
        """Resolve relative paths and initialize defaults."""
        if self.config_path:
            base_dir = self.config_path.parent

            if not self.ingest.persist_dir.is_absolute():
                self.ingest.persist_dir = (base_dir / self.ingest.persist_dir).resolve()

            if not self.ingest.manifest.is_absolute():
                self.ingest.manifest = (base_dir / self.ingest.manifest).resolve()

            if self.vectordb and not self.vectordb.persist_dir.is_absolute():
                self.vectordb.persist_dir = (base_dir / self.vectordb.persist_dir).resolve()

        if self.modes is None:
            self.modes = self._get_builtin_modes()

    def validate(self) -> None:
        """Validate entire configuration."""
        self.ingest.validate()
        self.query.validate()
        self.llm.validate()
        self.vectordb.validate()

        if self.modes:
            for mode_name, mode_config in self.modes.items():
                try:
                    mode_config.validate()
                except Exception as exc:
                    warnings.warn(f"Mode '{mode_name}' failed validation: {exc}")

        if self.modes and self.query.mode not in self.modes:
            fallback = "grounded"
            if fallback not in self.modes and self.modes:
                fallback = next(iter(self.modes.keys()))
            warnings.warn(
                f"query.mode '{self.query.mode}' not found in modes registry. "
                f"Falling back to '{fallback}'."
            )
            self.query.mode = fallback

        if self.remote_providers:
            seen_names = set()
            valid_providers = []
            for provider_config in self.remote_providers:
                if not provider_config.name:
                    warnings.warn("Skipping remote provider entry missing a name.")
                    continue
                if provider_config.name in seen_names:
                    warnings.warn(f"Skipping duplicate remote provider name: {provider_config.name}")
                    continue
                try:
                    provider_config.validate()
                except Exception as exc:
                    warnings.warn(f"Skipping remote provider '{provider_config.name}': {exc}")
                    continue
                seen_names.add(provider_config.name)
                valid_providers.append(provider_config)
            self.remote_providers = valid_providers or None

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

    def get_active_remote_providers(
        self,
        allowed_names: Optional[Set[str]] = None,
    ) -> list[RemoteProviderConfig]:
        """
        Get all enabled remote providers.

        Returns:
            Dictionary of enabled remote providers
        """
        if not self.remote_providers:
            return []

        enabled = [config for config in self.remote_providers if config.enabled]
        if not allowed_names:
            return enabled

        normalized = {name.lower() for name in allowed_names if name}
        return [config for config in enabled if config.name.lower() in normalized]

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
