"""Tests verifying the BaseVectorDBProvider ABC contract."""

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import BaseVectorDBProvider, VectorDBGetResult, VectorDBQueryResult


class DummyProvider(BaseVectorDBProvider):
    """Minimal concrete implementation for testing the ABC contract."""

    def __init__(self, config: VectorDBConfig, provider_name: str = "dummy") -> None:
        super().__init__(config)
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return True

    def add_chunks(self, ids, documents, metadatas, embeddings) -> None:
        return None

    def get(self, ids=None, filters=None, limit=None, offset=0, include=None) -> VectorDBGetResult:
        return VectorDBGetResult()

    def query(self, embedding, top_k, filters=None) -> VectorDBQueryResult:
        return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])

    def delete_by_id(self, ids) -> None:
        return None

    def delete_by_filter(self, filters) -> None:
        return None

    def delete_all(self) -> int:
        return 0

    def count(self) -> int:
        return 0

    def get_stats(self) -> dict:
        return {}

    def optimize(self) -> dict:
        return {}

    def health_check(self) -> dict:
        return {}

    def update_metadatas(self, ids, metadatas) -> None:
        return None


def test_dummy_provider_instantiates(tmp_path) -> None:
    cfg = VectorDBConfig(provider="dummy", persist_dir=tmp_path / ".db", collection="kb")
    provider = DummyProvider(cfg)
    assert provider.name == "dummy"
    assert provider.is_available() is True
    assert provider.count() == 0
    assert provider.delete_all() == 0


def test_dummy_provider_get_returns_empty(tmp_path) -> None:
    cfg = VectorDBConfig(provider="dummy", persist_dir=tmp_path / ".db", collection="kb")
    provider = DummyProvider(cfg)
    result = provider.get()
    assert result.ids == []
    assert result.documents is None
    assert result.metadatas is None


def test_dummy_provider_query_returns_empty(tmp_path) -> None:
    cfg = VectorDBConfig(provider="dummy", persist_dir=tmp_path / ".db", collection="kb")
    provider = DummyProvider(cfg)
    result = provider.query([0.1, 0.2], top_k=5)
    assert result.ids == []
    assert result.documents == []
