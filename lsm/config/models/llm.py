"""
LLM provider configuration models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE


@dataclass
class LLMServiceConfig:
    """
    Configuration for a named LLM service.

    Maps a logical service name (e.g., 'query', 'tagging', 'ranking')
    to a specific provider and model with optional parameter overrides.
    """

    provider: str
    """Provider name this service uses (must match a providers[] entry)."""

    model: str
    """Model name to use for this service."""

    temperature: Optional[float] = None
    """Temperature override. If None, uses DEFAULT_LLM_TEMPERATURE."""

    max_tokens: Optional[int] = None
    """Max tokens override. If None, uses DEFAULT_LLM_MAX_TOKENS."""

    def validate(self) -> None:
        """Validate service configuration."""
        if not self.provider:
            raise ValueError("Service 'provider' is required")
        if not self.model:
            raise ValueError("Service 'model' is required")
        if self.temperature is not None and (self.temperature < 0.0 or self.temperature > 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


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
            if self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider in {"anthropic", "claude"}:
                self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
            else:
                env_var = f"{self.provider.upper()}_API_KEY"
                self.api_key = os.getenv(env_var)

        if not self.base_url:
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
    Provider connection configuration within the LLM registry.
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

    def validate(self) -> None:
        """Validate provider configuration."""
        if not self.provider_name:
            raise ValueError("providers[].provider_name is required")


@dataclass
class LLMRegistryConfig:
    """
    LLM provider registry with named services.

    Providers define connection details. Services map logical names
    to a specific provider + model combination.
    """

    providers: List[LLMProviderConfig]
    """List of available LLM provider connections."""

    services: Dict[str, LLMServiceConfig] = field(default_factory=dict)
    """Named service configurations mapping to providers."""

    def resolve_service(self, name: str) -> LLMConfig:
        """
        Resolve a named service to a fully-hydrated LLMConfig.

        Looks up the service by name, falls back to 'default' if not found,
        then merges service model parameters with provider connection details.

        Args:
            name: Service name (e.g., 'query', 'tagging', 'ranking', or user-defined)

        Returns:
            Fully resolved LLMConfig ready for provider factory.

        Raises:
            ValueError: If service name not found and no 'default' service exists.
            ValueError: If service references a provider not in providers list.
        """
        service = self.services.get(name)
        if service is None:
            service = self.services.get("default")
        if service is None:
            raise ValueError(
                f"No LLM service configured for '{name}' and no 'default' service defined. "
                f"Available services: {sorted(self.services.keys())}"
            )

        provider = self._get_provider(service.provider)
        if provider is None:
            raise ValueError(
                f"Service '{name}' references provider '{service.provider}' "
                f"which is not in the providers list. "
                f"Available providers: {[p.provider_name for p in self.providers]}"
            )

        return LLMConfig(
            provider=provider.provider_name,
            model=service.model,
            api_key=provider.api_key,
            temperature=(
                service.temperature
                if service.temperature is not None
                else DEFAULT_LLM_TEMPERATURE
            ),
            max_tokens=(
                service.max_tokens
                if service.max_tokens is not None
                else DEFAULT_LLM_MAX_TOKENS
            ),
            base_url=provider.base_url,
            endpoint=provider.endpoint,
            api_version=provider.api_version,
            deployment_name=provider.deployment_name,
        )

    def resolve_any_for_provider(self, provider_name: str) -> Optional[LLMConfig]:
        """
        Resolve the first service that uses the given provider.

        Used for testing provider availability (e.g., health checks).

        Args:
            provider_name: Provider name to look up.

        Returns:
            Resolved LLMConfig if any service uses this provider, else None.
        """
        for name, service in self.services.items():
            if service.provider == provider_name:
                try:
                    return self.resolve_service(name)
                except ValueError:
                    continue
        return None

    def _get_provider(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Find a provider by name."""
        for p in self.providers:
            if p.provider_name == provider_name:
                return p
        return None

    def get_query_config(self) -> LLMConfig:
        """Resolve the 'query' service."""
        return self.resolve_service("query")

    def get_tagging_config(self) -> LLMConfig:
        """Resolve the 'tagging' service."""
        return self.resolve_service("tagging")

    def get_ranking_config(self) -> LLMConfig:
        """Resolve the 'ranking' service."""
        return self.resolve_service("ranking")

    def get_feature_provider_map(self) -> dict[str, str]:
        """Build a map of well-known service names to their provider names."""
        mapping: dict[str, str] = {}
        for name in ("query", "tagging", "ranking"):
            service = self.services.get(name)
            if service is not None:
                mapping[name] = service.provider
        return mapping

    def get_provider_names(self) -> List[str]:
        """Get all registered provider names."""
        return [p.provider_name for p in self.providers]

    def get_provider_by_name(self, name: str) -> Optional[LLMProviderConfig]:
        """Find a provider by name."""
        return self._get_provider(name)

    def set_feature_selection(
        self, feature_name: str, provider_name: str, model: str
    ) -> None:
        """
        Set the provider + model for a service, creating or updating the entry.

        Args:
            feature_name: Service name (e.g., 'query', 'tagging', 'ranking')
            provider_name: Provider name to use
            model: Model name to set
        """
        if self._get_provider(provider_name) is None:
            raise ValueError(f"Provider '{provider_name}' not found in providers list")

        existing = self.services.get(feature_name)
        if existing is not None:
            existing.provider = provider_name
            existing.model = model
        else:
            self.services[feature_name] = LLMServiceConfig(
                provider=provider_name,
                model=model,
            )

    def override_feature_model(self, feature_name: str, model: str) -> None:
        """
        Override just the model for an existing service.

        Args:
            feature_name: Service name
            model: New model name
        """
        service = self.services.get(feature_name)
        if service is None:
            service = self.services.get("default")
        if service is None:
            raise ValueError(f"No service '{feature_name}' or 'default' to override")
        service.model = model

    def validate(self) -> None:
        """Validate registry configuration."""
        if not self.providers:
            raise ValueError("llms.providers must be a non-empty list")

        if not self.services:
            raise ValueError("llms.services must be a non-empty dict")

        for provider in self.providers:
            provider.validate()

        for service_name, service in self.services.items():
            service.validate()
            if self._get_provider(service.provider) is None:
                raise ValueError(
                    f"Service '{service_name}' references provider '{service.provider}' "
                    f"which is not in the providers list"
                )

        if "query" not in self.services and "default" not in self.services:
            raise ValueError(
                "LLM services must define at least a 'query' or 'default' service"
            )

        self.get_query_config().validate()
        if "tagging" in self.services:
            self.get_tagging_config().validate()
        if "ranking" in self.services:
            self.get_ranking_config().validate()
