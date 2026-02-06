"""
LLM provider configuration models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .constants import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE


@dataclass
class FeatureLLMConfig:
    """
    Optional LLM configuration override for a specific feature.

    Allows per-feature model selection while inheriting from provider config.
    """

    model: Optional[str] = None
    """Override model for this feature. If None, inherits from provider config."""

    api_key: Optional[str] = None
    """Override API key for this feature. If None, inherits from provider config."""

    temperature: Optional[float] = None
    """Override temperature for this feature. If None, inherits from provider config."""

    max_tokens: Optional[int] = None
    """Override max_tokens for this feature. If None, inherits from provider config."""

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
            provider=base.provider,
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
    Effective LLM provider configuration used at runtime.
    """

    provider: str
    """LLM provider name."""

    model: str
    """Model name to use. For OpenAI: gpt-5.2, gpt-4, etc."""

    api_key: Optional[str] = None
    """
    API key for the provider.
    If None, will attempt to load from environment variable based on provider.
    """

    temperature: float = DEFAULT_LLM_TEMPERATURE
    """Temperature for LLM generation (0.0 = deterministic, 1.0 = creative)."""

    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    """Maximum tokens to generate in responses."""

    base_url: Optional[str] = None
    """Base URL for local or hosted providers (e.g., Ollama)."""

    endpoint: Optional[str] = None
    """Provider endpoint URL (e.g., Azure OpenAI resource endpoint)."""

    api_version: Optional[str] = None
    """Provider API version (e.g., Azure OpenAI)."""

    deployment_name: Optional[str] = None
    """Provider deployment name (e.g., Azure OpenAI deployment)."""

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if not self.api_key:
            # Try provider-specific environment variable
            if self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider in {"anthropic", "claude"}:
                self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
            else:
                env_var = f"{self.provider.upper()}_API_KEY"
                self.api_key = os.getenv(env_var)

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
            if self.provider in {"anthropic", "claude"}:
                raise ValueError(
                    f"API key required for provider '{self.provider}'. "
                    "Set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable or provide in config."
                )
            raise ValueError(
                f"API key required for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_API_KEY environment variable or provide in config."
            )

        if self.provider == "azure_openai":
            if not self.endpoint:
                raise ValueError(
                    "Azure OpenAI requires 'endpoint'. "
                    "Set llms[].endpoint or AZURE_OPENAI_ENDPOINT."
                )
            if not self.api_version:
                raise ValueError(
                    "Azure OpenAI requires 'api_version'. "
                    "Set llms[].api_version or AZURE_OPENAI_API_VERSION."
                )

        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class LLMProviderConfig:
    """
    Provider configuration entry within ordered LLM registry.
    """

    provider_name: str
    """Provider name (e.g., 'openai', 'gemini', 'anthropic')."""

    api_key: Optional[str] = None
    """Default API key for this provider."""

    base_url: Optional[str] = None
    """Base URL for local or hosted providers (e.g., Ollama)."""

    endpoint: Optional[str] = None
    """Provider endpoint URL (e.g., Azure OpenAI resource endpoint)."""

    api_version: Optional[str] = None
    """Provider API version (e.g., Azure OpenAI)."""

    deployment_name: Optional[str] = None
    """Provider deployment name (e.g., Azure OpenAI deployment)."""

    query: Optional[FeatureLLMConfig] = None
    """Optional LLM config override for query/answer synthesis."""

    tagging: Optional[FeatureLLMConfig] = None
    """Optional LLM config override for AI tagging."""

    ranking: Optional[FeatureLLMConfig] = None
    """Optional LLM config override for AI-powered reranking."""

    def _resolve_feature(self, feature: FeatureLLMConfig) -> LLMConfig:
        model = feature.model
        if not model:
            raise ValueError(
                f"Model required for provider '{self.provider_name}'. "
                "Set a feature-level model under query/tagging/ranking."
            )

        temperature = (
            feature.temperature if feature.temperature is not None else DEFAULT_LLM_TEMPERATURE
        )
        max_tokens = feature.max_tokens if feature.max_tokens is not None else DEFAULT_LLM_MAX_TOKENS

        return LLMConfig(
            provider=self.provider_name,
            model=model,
            api_key=feature.api_key if feature.api_key is not None else self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=feature.base_url if feature.base_url is not None else self.base_url,
            endpoint=feature.endpoint if feature.endpoint is not None else self.endpoint,
            api_version=feature.api_version if feature.api_version is not None else self.api_version,
            deployment_name=(
                feature.deployment_name
                if feature.deployment_name is not None
                else self.deployment_name
            ),
        )

    def resolve_query(self) -> Optional[LLMConfig]:
        if not self.query:
            return None
        return self._resolve_feature(self.query)

    def resolve_tagging(self) -> Optional[LLMConfig]:
        if not self.tagging:
            return None
        return self._resolve_feature(self.tagging)

    def resolve_ranking(self) -> Optional[LLMConfig]:
        if not self.ranking:
            return None
        return self._resolve_feature(self.ranking)

    def resolve_first_available(self) -> Optional[LLMConfig]:
        for feature in (self.query, self.tagging, self.ranking):
            if feature:
                return self._resolve_feature(feature)
        return None

    def validate(self) -> None:
        if not self.provider_name:
            raise ValueError("llms[].provider_name is required")

        if not any((self.query, self.tagging, self.ranking)):
            raise ValueError(
                f"Provider '{self.provider_name}' must define at least one of "
                "query/tagging/ranking."
            )

        for label, feature in (
            ("query", self.query),
            ("tagging", self.tagging),
            ("ranking", self.ranking),
        ):
            if feature and not feature.model:
                raise ValueError(
                    f"Model required for {label} on provider '{self.provider_name}'. "
                    "Set a feature-level model."
                )


@dataclass
class LLMRegistryConfig:
    """
    Ordered list of LLM providers with per-feature selection.
    """

    llms: List[LLMProviderConfig]
    """Ordered list of LLM providers. Later providers override feature selection."""

    def _select_feature(self, feature_name: str) -> LLMConfig:
        selected = None
        for provider in self.llms:
            feature = getattr(provider, feature_name)
            if feature is not None:
                selected = provider._resolve_feature(feature)
        if not selected:
            raise ValueError(f"No LLM provider configured for feature '{feature_name}'")
        return selected

    def get_query_config(self) -> LLMConfig:
        return self._select_feature("query")

    def get_tagging_config(self) -> LLMConfig:
        return self._select_feature("tagging")

    def get_ranking_config(self) -> LLMConfig:
        return self._select_feature("ranking")

    def get_feature_provider_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for provider in self.llms:
            if provider.query:
                mapping["query"] = provider.provider_name
            if provider.tagging:
                mapping["tagging"] = provider.provider_name
            if provider.ranking:
                mapping["ranking"] = provider.provider_name
        return mapping

    def get_provider_names(self) -> List[str]:
        return [provider.provider_name for provider in self.llms]

    def get_provider_by_name(self, name: str) -> Optional[LLMProviderConfig]:
        for provider in self.llms:
            if provider.provider_name == name:
                return provider
        return None

    def override_feature_model(self, feature_name: str, model: str) -> None:
        """
        Override the model for a specific feature in-place.

        Args:
            feature_name: One of 'query', 'tagging', 'ranking'
            model: Model name to apply
        """
        if feature_name not in {"query", "tagging", "ranking"}:
            raise ValueError(f"Unknown feature '{feature_name}'")

        target = None
        for provider in self.llms:
            feature = getattr(provider, feature_name)
            if feature is not None:
                target = feature
        if not target:
            raise ValueError(f"No LLM provider configured for feature '{feature_name}'")

        target.model = model

    def set_feature_selection(self, feature_name: str, provider_name: str, model: str) -> None:
        """
        Set the provider + model used for a feature, clearing other providers' selections.

        Args:
            feature_name: One of 'query', 'tagging', 'ranking'
            provider_name: Provider name to select
            model: Model name to set for the feature
        """
        if feature_name not in {"query", "tagging", "ranking"}:
            raise ValueError(f"Unknown feature '{feature_name}'")

        target = self.get_provider_by_name(provider_name)
        if not target:
            raise ValueError(f"Provider '{provider_name}' not found in llms list")

        for provider in self.llms:
            feature = getattr(provider, feature_name)
            if provider is target:
                if feature is None:
                    setattr(provider, feature_name, FeatureLLMConfig(model=model))
                else:
                    feature.model = model
            else:
                if feature is not None:
                    setattr(provider, feature_name, None)

    def validate(self) -> None:
        if not self.llms:
            raise ValueError("llms must be a non-empty list")

        for provider in self.llms:
            provider.validate()

        required = {"query"}
        available = self.get_feature_provider_map().keys()
        missing = required - set(available)
        if missing:
            raise ValueError(
                f"Missing LLM feature configs: {sorted(missing)}. "
                "Define these under llms[].query/tagging/ranking."
            )

        self.get_query_config().validate()
        if any(provider.tagging for provider in self.llms):
            self.get_tagging_config().validate()
        if any(provider.ranking for provider in self.llms):
            self.get_ranking_config().validate()
