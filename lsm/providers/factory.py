"""
Provider factory for creating LLM providers.

Handles provider instantiation based on configuration.
"""

from __future__ import annotations

from typing import Dict, Type

from lsm.config.models import LLMConfig
from lsm.cli.logging import get_logger
from .base import BaseLLMProvider
from .openai import OpenAIProvider

logger = get_logger(__name__)

# Registry of available providers
PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    # Future providers:
    # "anthropic": AnthropicProvider,
    # "local": LocalProvider,
    # "cohere": CohereProvider,
}


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Create an LLM provider based on configuration.

    Args:
        config: LLM configuration

    Returns:
        Initialized provider instance

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> from lsm.config import load_config_from_file
        >>> config = load_config_from_file("config.json")
        >>> provider = create_provider(config.llm)
        >>> print(provider)
        openai/gpt-5.2
    """
    provider_name = config.provider.lower()

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported LLM provider: '{provider_name}'. "
            f"Available providers: {available}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]

    logger.debug(f"Creating provider: {provider_name}")
    provider = provider_class(config)

    if not provider.is_available():
        logger.warning(
            f"Provider '{provider_name}' is not available. "
            f"Check API key configuration."
        )

    return provider


def list_available_providers() -> list[str]:
    """
    Get list of available provider names.

    Returns:
        List of provider names

    Example:
        >>> list_available_providers()
        ['openai']
    """
    return list(PROVIDER_REGISTRY.keys())


def register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None:
    """
    Register a custom provider.

    This allows users to add their own provider implementations.

    Args:
        name: Provider name
        provider_class: Provider class (must inherit from BaseLLMProvider)

    Raises:
        TypeError: If provider_class doesn't inherit from BaseLLMProvider

    Example:
        >>> class MyProvider(BaseLLMProvider):
        ...     # Implementation
        ...     pass
        >>> register_provider("myprovider", MyProvider)
    """
    if not issubclass(provider_class, BaseLLMProvider):
        raise TypeError(
            f"Provider class must inherit from BaseLLMProvider, "
            f"got {provider_class}"
        )

    logger.info(f"Registering custom provider: {name}")
    PROVIDER_REGISTRY[name] = provider_class
