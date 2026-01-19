"""
Factory for vector database providers.
"""

from __future__ import annotations

from typing import Dict, Type

from lsm.logging import get_logger
from lsm.config.models import VectorDBConfig
from .base import BaseVectorDBProvider
from .chromadb import ChromaDBProvider

logger = get_logger(__name__)

PostgreSQLProvider = None

try:
    from .postgresql import PostgreSQLProvider
except Exception as e:
    logger.debug(f"PostgreSQL provider not available: {e}")


PROVIDER_REGISTRY: Dict[str, Type[BaseVectorDBProvider]] = {
    "chromadb": ChromaDBProvider,
}

if PostgreSQLProvider is not None:
    PROVIDER_REGISTRY["postgresql"] = PostgreSQLProvider


def create_vectordb_provider(config: VectorDBConfig) -> BaseVectorDBProvider:
    """
    Create a vector DB provider from config.

    Args:
        config: Vector DB configuration

    Returns:
        Initialized provider instance
    """
    provider_name = (config.provider or "chromadb").lower()

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported vector DB provider: '{provider_name}'. "
            f"Available providers: {available}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]
    logger.debug(f"Creating vector DB provider: {provider_name}")
    provider = provider_class(config)

    if not provider.is_available():
        logger.warning(
            f"Vector DB provider '{provider_name}' is not available. "
            f"Check configuration and dependencies."
        )

    return provider


def list_available_providers() -> list[str]:
    """Return list of available vector DB providers."""
    return list(PROVIDER_REGISTRY.keys())


def register_provider(name: str, provider_class: Type[BaseVectorDBProvider]) -> None:
    """Register a custom vector DB provider."""
    if not issubclass(provider_class, BaseVectorDBProvider):
        raise TypeError(
            f"Provider class must inherit from BaseVectorDBProvider, "
            f"got {provider_class}"
        )

    logger.info(f"Registering vector DB provider: {name}")
    PROVIDER_REGISTRY[name] = provider_class
