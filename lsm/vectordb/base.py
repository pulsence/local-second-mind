"""
Base interface for vector database providers.

Defines the contract all vector DB providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass
class VectorDBGetResult:
    """Normalized result from a get (non-similarity) retrieval.

    Fields other than ``ids`` are populated only when requested via
    the ``include`` parameter of :meth:`BaseVectorDBProvider.get`.
    """
    ids: List[str] = field(default_factory=list)
    documents: Optional[List[str]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[List[List[float]]] = None

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ids": self.ids}
        if self.documents is not None:
            result["documents"] = self.documents
        if self.metadatas is not None:
            result["metadatas"] = self.metadatas
        if self.embeddings is not None:
            result["embeddings"] = self.embeddings
        return result


@dataclass
class PruneCriteria:
    """Criteria for pruning non-current chunk versions."""

    max_versions: Optional[int] = None
    older_than_days: Optional[int] = None


class BaseVectorDBProvider(ABC):
    """
    Abstract base class for vector database providers.

    Providers must implement add/get/query/update/delete/count/health operations.
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
    def get(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include: Optional[List[str]] = None,
    ) -> VectorDBGetResult:
        """Retrieve vectors by ID and/or metadata filter.

        At least one of ``ids`` or ``filters`` should be provided, or
        ``limit`` to fetch a paginated batch.

        Args:
            ids: Specific vector IDs to retrieve.
            filters: Metadata filter dict. Simple ``{"key": "value"}`` means
                equality. ChromaDB-style ``{"key": {"$eq": "value"}}`` is also
                accepted and normalized by providers that need it.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            include: Fields to include in the result. Valid values are
                ``"documents"``, ``"metadatas"``, and ``"embeddings"``.
                Defaults to ``["metadatas"]`` if not specified.

        Returns:
            VectorDBGetResult with requested fields populated.
        """
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
    def delete_all(self) -> int:
        """Delete all vectors in the collection.

        Returns:
            Number of vectors deleted.
        """
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

    @abstractmethod
    def update_metadatas(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Update metadata for existing vectors by ID.

        Args:
            ids: List of vector IDs to update.
            metadatas: List of metadata dicts (one per ID). Replaces the
                existing metadata entirely for each vector.
        """
        pass

    @abstractmethod
    def prune_old_versions(self, criteria: PruneCriteria) -> int:
        """Delete non-current chunk versions that match prune criteria."""
        pass
