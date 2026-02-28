"""
Top-level configuration model for Local Second Mind.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from lsm.paths import ensure_global_folders

from .global_config import GlobalConfig
from .chats import ChatsConfig
from .agents import AgentConfig
from .ingest import IngestConfig
from .llm import LLMRegistryConfig
from .modes import (
    ModeConfig,
    BUILT_IN_MODES,
    NotesConfig,
    RemoteProviderConfig,
    RemoteProviderChainConfig,
    RemoteConfig,
    get_builtin_modes,
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

    global_settings: GlobalConfig = field(default_factory=GlobalConfig)
    """Global settings shared across multiple modules."""

    modes: Optional[dict[str, ModeConfig]] = None
    """Registry of available query modes. If None, uses built-in defaults."""

    remote: RemoteConfig = field(default_factory=RemoteConfig)
    """Global remote configuration."""

    remote_providers: Optional[list[RemoteProviderConfig]] = None
    """List of remote source providers (web search, APIs, etc)."""

    remote_provider_chains: Optional[list[RemoteProviderChainConfig]] = None
    """Optional named remote provider chains."""

    notes: NotesConfig = field(default_factory=NotesConfig)
    """Global notes configuration applied across modes."""

    chats: ChatsConfig = field(default_factory=ChatsConfig)
    """Global chat transcript configuration."""

    agents: Optional[AgentConfig] = None
    """Optional agent framework configuration."""

    config_path: Optional[Path] = None
    """Path to the config file (for resolving relative paths)."""

    def __post_init__(self):
        """Resolve relative paths and initialize defaults."""
        ensure_global_folders(self.global_settings.global_folder)

        base_dir: Optional[Path] = self.global_settings.global_folder
        if base_dir is None and self.config_path is not None:
            base_dir = self.config_path.parent

        if base_dir is not None:
            if self.vectordb and not self.vectordb.path.is_absolute():
                self.vectordb.path = (base_dir / self.vectordb.path).resolve()

        if self.agents is not None and not self.agents.agents_folder.is_absolute():
            if self.global_settings.global_folder is not None:
                self.agents.agents_folder = (
                    self.global_settings.global_folder / self.agents.agents_folder
                ).resolve()
            elif self.config_path:
                self.agents.agents_folder = (
                    self.config_path.parent / self.agents.agents_folder
                ).resolve()
            else:
                self.agents.agents_folder = self.agents.agents_folder.resolve()

        if self.modes is None:
            self.modes = get_builtin_modes()

    def validate(self) -> None:
        """Validate entire configuration."""
        self.global_settings.validate()
        self.ingest.validate()
        self.query.validate()
        self.llm.validate()
        self.vectordb.validate()
        self.chats.validate()
        if self.remote is not None:
            try:
                self.remote.validate()
            except Exception as exc:
                warnings.warn(f"Remote config failed validation: {exc}")
        if self.agents is not None:
            self.agents.validate()

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

        self._merge_preconfigured_chains()

        if self.remote_provider_chains:
            provider_names = {p.name for p in (self.remote_providers or [])}
            seen_chain_names = set()
            valid_chains = []
            for chain in self.remote_provider_chains:
                if not chain.name:
                    warnings.warn("Skipping remote provider chain missing a name.")
                    continue
                if chain.name in seen_chain_names:
                    warnings.warn(f"Skipping duplicate remote provider chain name: {chain.name}")
                    continue
                try:
                    chain.validate()
                except Exception as exc:
                    warnings.warn(f"Skipping remote provider chain '{chain.name}': {exc}")
                    continue
                missing = [
                    link.source for link in chain.links
                    if link.source not in provider_names
                ]
                if missing:
                    warnings.warn(
                        f"Skipping remote provider chain '{chain.name}' because "
                        f"providers are not configured: {sorted(set(missing))}"
                    )
                    continue
                seen_chain_names.add(chain.name)
                valid_chains.append(chain)
            self.remote_provider_chains = valid_chains or None

    def _merge_preconfigured_chains(self) -> None:
        if not self.remote or not self.remote.chains:
            return
        try:
            from lsm.remote.chains import get_preconfigured_chain_configs
        except Exception as exc:
            warnings.warn(f"Failed to load preconfigured chains: {exc}")
            return
        preconfigured = get_preconfigured_chain_configs(self.remote.chains)
        if not preconfigured:
            return
        existing = {chain.name.lower(): chain for chain in (self.remote_provider_chains or [])}
        ordered: list[RemoteProviderChainConfig] = list(
            self.remote_provider_chains or []
        )
        for chain in preconfigured:
            key = chain.name.lower()
            if key in existing:
                continue
            existing[key] = chain
            ordered.append(chain)
        self.remote_provider_chains = ordered or None

    @staticmethod
    def _get_builtin_modes() -> dict[str, ModeConfig]:
        """
        Get built-in default query modes.

        Returns:
            Dictionary of mode name to ModeConfig
        """
        _ = BUILT_IN_MODES
        return get_builtin_modes()

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
        Get configured remote providers, optionally filtered by name.

        Returns:
            List of matching remote provider configurations.
        """
        if not self.remote_providers:
            return []

        if not allowed_names:
            return list(self.remote_providers)

        normalized = {name.lower() for name in allowed_names if name}
        return [
            config for config in self.remote_providers if config.name.lower() in normalized
        ]

    def set_remote_provider_weight(
        self,
        provider_name: str,
        weight: float,
    ) -> bool:
        """
        Set the weight for a remote provider.
        """
        for provider_config in self.remote_providers or []:
            if provider_config.name.lower() == provider_name.lower():
                provider_config.weight = weight
                return True
        return False

    def get_remote_provider_chain(self, chain_name: str) -> Optional[RemoteProviderChainConfig]:
        """Get a configured remote provider chain by name."""
        for chain in self.remote_provider_chains or []:
            if chain.name.lower() == chain_name.lower():
                return chain
        return None

    @property
    def persist_dir(self) -> Path:
        """Compatibility alias for vectordb path."""
        return self.vectordb.path

    @property
    def collection(self) -> str:
        """Shortcut to vectordb collection name."""
        return self.vectordb.collection

    @property
    def global_folder(self) -> Optional[Path]:
        """Shortcut to global_settings.global_folder."""
        return self.global_settings.global_folder

    @property
    def embed_model(self) -> str:
        """Shortcut to global_settings.embed_model."""
        return self.global_settings.embed_model

    @property
    def device(self) -> str:
        """Shortcut to global_settings.device."""
        return self.global_settings.device

    @property
    def batch_size(self) -> int:
        """Shortcut to global_settings.batch_size."""
        return self.global_settings.batch_size

    @property
    def embedding_dimension(self) -> Optional[int]:
        """Shortcut to global_settings.embedding_dimension."""
        return self.global_settings.embedding_dimension
