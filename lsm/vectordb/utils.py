"""
Utilities for bridging vector DB providers with legacy Chroma code paths.
"""

from __future__ import annotations

from typing import Any

from lsm.vectordb.base import BaseVectorDBProvider


def require_chroma_collection(obj: Any, action: str):
    """
    Return a ChromaDB collection from a provider or collection-like object.

    Raises a helpful error when the operation requires ChromaDB APIs.
    """
    if isinstance(obj, BaseVectorDBProvider):
        if getattr(obj, "name", "") == "chromadb" and hasattr(obj, "get_collection"):
            return obj.get_collection()
        raise ValueError(
            f"{action} requires ChromaDB provider. "
            f"Current provider: {getattr(obj, 'name', 'unknown')}"
        )
    return obj
