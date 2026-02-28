"""
Vector database providers.
"""

from .base import BaseVectorDBProvider, VectorDBGetResult, VectorDBQueryResult
from .factory import create_vectordb_provider, list_available_providers, register_provider
from .sqlite_vec import SQLiteVecProvider


def create_vectordb(config):
    """Compatibility alias for create_vectordb_provider."""
    return create_vectordb_provider(config)

__all__ = [
    "BaseVectorDBProvider",
    "VectorDBGetResult",
    "VectorDBQueryResult",
    "SQLiteVecProvider",
    "create_vectordb_provider",
    "create_vectordb",
    "list_available_providers",
    "register_provider",
]
