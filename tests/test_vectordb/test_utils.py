import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import BaseVectorDBProvider, VectorDBQueryResult
from lsm.vectordb.utils import require_chroma_collection


class DummyProvider(BaseVectorDBProvider):
    def __init__(self, config: VectorDBConfig, provider_name: str = "dummy") -> None:
        super().__init__(config)
        self._provider_name = provider_name
        self._collection = object()

    @property
    def name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return True

    def add_chunks(self, ids, documents, metadatas, embeddings) -> None:
        return None

    def query(self, embedding, top_k, filters=None) -> VectorDBQueryResult:
        return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])

    def delete_by_id(self, ids) -> None:
        return None

    def delete_by_filter(self, filters) -> None:
        return None

    def count(self) -> int:
        return 0

    def get_stats(self) -> dict:
        return {}

    def optimize(self) -> dict:
        return {}

    def health_check(self) -> dict:
        return {}

    def get_collection(self):
        return self._collection


def test_require_chroma_collection_passthrough_object() -> None:
    obj = object()
    assert require_chroma_collection(obj, "/test") is obj


def test_require_chroma_collection_with_chroma_provider(tmp_path) -> None:
    provider = DummyProvider(
        VectorDBConfig(provider="chromadb", persist_dir=tmp_path / ".db", collection="kb"),
        provider_name="chromadb",
    )
    assert require_chroma_collection(provider, "/test") is provider.get_collection()


def test_require_chroma_collection_rejects_non_chroma_provider(tmp_path) -> None:
    provider = DummyProvider(
        VectorDBConfig(provider="postgresql", persist_dir=tmp_path / ".db", collection="kb"),
        provider_name="postgresql",
    )
    with pytest.raises(ValueError, match="requires ChromaDB provider"):
        require_chroma_collection(provider, "/test")
