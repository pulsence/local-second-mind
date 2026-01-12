"""
Factory for creating remote source providers.

Provides a registry-based system for instantiating remote providers
based on configuration.
"""

from __future__ import annotations

from typing import Dict, Type, Any

from .base import BaseRemoteProvider
from lsm.cli.logging import get_logger

logger = get_logger(__name__)

# Global registry of remote provider types
_PROVIDER_REGISTRY: Dict[str, Type[BaseRemoteProvider]] = {}


def register_remote_provider(provider_type: str, provider_class: Type[BaseRemoteProvider]) -> None:
    """
    Register a remote provider type.

    Args:
        provider_type: Type identifier (e.g., "web_search", "brave_search")
        provider_class: Provider class implementing BaseRemoteProvider

    Example:
        >>> register_remote_provider("brave_search", BraveSearchProvider)
    """
    _PROVIDER_REGISTRY[provider_type] = provider_class
    logger.debug(f"Registered remote provider: {provider_type} -> {provider_class.__name__}")


def create_remote_provider(
    provider_type: str,
    config: Dict[str, Any],
) -> BaseRemoteProvider:
    """
    Create a remote provider instance.

    Args:
        provider_type: Type identifier registered with register_remote_provider()
        config: Provider-specific configuration

    Returns:
        Remote provider instance

    Raises:
        ValueError: If provider_type is not registered

    Example:
        >>> config = {"api_key": "...", "max_results": 5}
        >>> provider = create_remote_provider("brave_search", config)
    """
    if provider_type not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown remote provider type: '{provider_type}'. "
            f"Available types: {list(_PROVIDER_REGISTRY.keys())}"
        )

    provider_class = _PROVIDER_REGISTRY[provider_type]
    provider = provider_class(config)

    # Validate configuration
    provider.validate_config()

    return provider


def get_registered_providers() -> Dict[str, Type[BaseRemoteProvider]]:
    """
    Get all registered remote provider types.

    Returns:
        Dictionary mapping provider type to provider class
    """
    return dict(_PROVIDER_REGISTRY)
