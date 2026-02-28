"""
ChromaDB provider implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lsm.logging import get_logger
from lsm.config.models import VectorDBConfig
from .base import BaseVectorDBProvider, PruneCriteria, VectorDBGetResult, VectorDBQueryResult

logger = get_logger(__name__)


class ChromaDBProvider(BaseVectorDBProvider):
    """Vector DB provider backed by ChromaDB."""

    def __init__(self, config: VectorDBConfig) -> None:
        super().__init__(config)
        self._client = None
        self._collection = None

    @property
    def name(self) -> str:
        return "chromadb"

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as exc:
            raise RuntimeError(
                "ChromaDB dependency is not available. Install 'chromadb' to use this provider."
            ) from exc

        self._client = chromadb.PersistentClient(
            path=str(self.config.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection,
            metadata={"hnsw:space": getattr(self.config, "chroma_hnsw_space", "cosine")},
        )
        return self._collection

    def _get_max_batch_size(self, collection) -> int:
        max_bs = None
        try:
            max_bs = collection._client.get_max_batch_size()
        except Exception:
            pass

        if max_bs is None:
            max_bs = getattr(collection, "max_batch_size", None)

        if not isinstance(max_bs, int) or max_bs <= 0:
            max_bs = 4000

        return max_bs

    def is_available(self) -> bool:
        try:
            self._ensure_collection()
            return True
        except Exception as e:
            logger.debug(f"ChromaDB provider unavailable: {e}")
            return False

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        if not ids:
            return

        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas, and embeddings must have the same length")

        collection = self._ensure_collection()
        max_bs = self._get_max_batch_size(collection)

        n = len(ids)
        for i in range(0, n, max_bs):
            j = min(i + max_bs, n)
            sub_ids = ids[i:j]
            sub_docs = documents[i:j]
            sub_metas = metadatas[i:j]
            sub_embs = embeddings[i:j]

            try:
                collection.upsert(
                    ids=sub_ids,
                    documents=sub_docs,
                    metadatas=sub_metas,
                    embeddings=sub_embs,
                )
                continue
            except AttributeError:
                pass

            try:
                collection.add(
                    ids=sub_ids,
                    documents=sub_docs,
                    metadatas=sub_metas,
                    embeddings=sub_embs,
                )
            except Exception as e:
                try:
                    collection.delete(ids=sub_ids)
                    collection.add(
                        ids=sub_ids,
                        documents=sub_docs,
                        metadatas=sub_metas,
                        embeddings=sub_embs,
                    )
                except Exception:
                    raise e

    def get(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include: Optional[List[str]] = None,
    ) -> VectorDBGetResult:
        """Retrieve vectors by ID and/or metadata filter."""
        collection = self._ensure_collection()
        inc = include or ["metadatas"]

        kwargs: Dict[str, Any] = {"include": inc}
        if ids is not None:
            kwargs["ids"] = ids
        if filters:
            kwargs["where"] = filters
        if limit is not None:
            kwargs["limit"] = limit
        if offset:
            kwargs["offset"] = offset

        try:
            results = collection.get(**kwargs)
        except TypeError:
            # Some ChromaDB versions don't support offset
            kwargs.pop("offset", None)
            results = collection.get(**kwargs)

        return VectorDBGetResult(
            ids=results.get("ids", []),
            documents=results.get("documents") if "documents" in inc else None,
            metadatas=results.get("metadatas") if "metadatas" in inc else None,
            embeddings=results.get("embeddings") if "embeddings" in inc else None,
        )

    def query(
        self,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> VectorDBQueryResult:
        collection = self._ensure_collection()
        res = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        return VectorDBQueryResult(
            ids=[str(cid) for cid in ids],
            documents=[doc or "" for doc in docs],
            metadatas=[meta or {} for meta in metas],
            distances=[dist for dist in dists],
        )

    def delete_by_id(self, ids: List[str]) -> None:
        if not ids:
            return
        collection = self._ensure_collection()
        collection.delete(ids=ids)

    def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        if not filters:
            raise ValueError("filters must be a non-empty dict")
        collection = self._ensure_collection()
        collection.delete(where=filters)

    def delete_all(self) -> int:
        """Delete all vectors in the collection."""
        collection = self._ensure_collection()
        results = collection.get(include=[])
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        collection = self._ensure_collection()
        return int(collection.count())

    def get_stats(self) -> Dict[str, Any]:
        collection = self._ensure_collection()
        stats = {
            "provider": self.name,
            "collection": collection.name,
            "count": collection.count(),
        }
        try:
            if collection.metadata:
                stats["metadata"] = collection.metadata
        except Exception:
            pass
        return stats

    def optimize(self) -> Dict[str, Any]:
        return {"provider": self.name, "status": "not_supported"}

    def health_check(self) -> Dict[str, Any]:
        try:
            count = self.count()
            return {"provider": self.name, "status": "ok", "count": count}
        except Exception as e:
            return {"provider": self.name, "status": "error", "error": str(e)}

    def update_metadatas(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Update metadata for existing vectors by ID."""
        if not ids:
            return
        collection = self._ensure_collection()
        collection.update(ids=ids, metadatas=metadatas)

    def prune_old_versions(self, criteria: PruneCriteria) -> int:
        _ = criteria
        # Chroma is retained for migration/testing only in v0.8.0.
        return 0

    def _get_raw_collection(self):
        """Return underlying ChromaDB collection for migration tools only."""
        return self._ensure_collection()
