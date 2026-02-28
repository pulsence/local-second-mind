"""
Factory for vector database providers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Type

from lsm.logging import get_logger
from lsm.config.models import VectorDBConfig
from .base import BaseVectorDBProvider

logger = get_logger(__name__)

PROVIDER_REGISTRY: Dict[str, str | Type[BaseVectorDBProvider]] = {
    "sqlite": "lsm.vectordb.sqlite_vec:SQLiteVecProvider",
    "postgresql": "lsm.vectordb.postgresql:PostgreSQLProvider",
}


def _load_provider_class(ref: str | Type[BaseVectorDBProvider]) -> Type[BaseVectorDBProvider]:
    if isinstance(ref, type):
        return ref
    module_name, class_name = ref.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def create_vectordb_provider(config: VectorDBConfig) -> BaseVectorDBProvider:
    """
    Create a vector DB provider from config.

    Args:
        config: Vector DB configuration

    Returns:
        Initialized provider instance
    """
    provider_name = (config.provider or "sqlite").lower()

    if provider_name == "chromadb":
        raise ValueError(
            "ChromaDB is no longer a production provider in v0.8.0. "
            "Run migration tooling and switch config.vectordb.provider to 'sqlite' or 'postgresql'."
        )

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported vector DB provider: '{provider_name}'. "
            f"Available providers: {available}"
        )

    provider_ref = PROVIDER_REGISTRY[provider_name]
    try:
        provider_class = _load_provider_class(provider_ref)
    except Exception as exc:
        raise ValueError(
            f"Vector DB provider '{provider_name}' is configured but unavailable: {exc}"
        ) from exc
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
