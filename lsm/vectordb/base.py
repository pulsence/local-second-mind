"""
Base interface for vector database providers.

Defines the contract all vector DB providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lsm.config.models import VectorDBConfig


@dataclass
class VectorDBQueryResult:
    """Normalized query result across vector DB providers."""
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[Optional[float]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "distances": self.distances,
        }


class BaseVectorDBProvider(ABC):
    """
    Abstract base class for vector database providers.

    Providers must implement add/query/delete/count/health operations.
    """

    def __init__(self, config: VectorDBConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'chromadb', 'postgresql')."""
        pass

    @property
    def collection(self) -> Optional[str]:
        """Collection or namespace name, if applicable."""
        return self.config.collection

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if provider is usable with current config."""
        pass

    @abstractmethod
    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add or upsert chunk vectors into the database."""
        pass

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> VectorDBQueryResult:
        """Run a similarity search and return normalized results."""
        pass

    @abstractmethod
    def delete_by_id(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        """Delete vectors that match a metadata filter."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total vector count."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return provider-specific stats."""
        pass

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """Run provider-specific optimization (VACUUM, compaction, etc.)."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return connection and health status."""
        pass

    def update_metadatas(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Update metadata for existing vectors by ID.

        Args:
            ids: List of vector IDs to update.
            metadatas: List of metadata dicts (one per ID).

        Raises:
            NotImplementedError: If the provider does not support metadata updates.
        """
        raise NotImplementedError(f"{self.name} does not support update_metadatas")

    def get_by_filter(self, filters: Dict[str, Any], include: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve vectors matching a metadata filter.

        Args:
            filters: Metadata filter dict.
            include: Fields to include (e.g. ``["metadatas"]``).

        Returns:
            Dict with ``ids``, ``metadatas``, and optionally other fields.

        Raises:
            NotImplementedError: If the provider does not support filtered retrieval.
        """
        raise NotImplementedError(f"{self.name} does not support get_by_filter")
