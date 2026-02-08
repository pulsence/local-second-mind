import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import BaseVectorDBProvider, VectorDBGetResult, VectorDBQueryResult
from lsm.vectordb.factory import (
    PROVIDER_REGISTRY,
    create_vectordb_provider,
    list_available_providers,
    register_provider,
)


class DummyProvider(BaseVectorDBProvider):
    @property
    def name(self) -> str:
        return "dummy"

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
        return {"provider": "dummy"}

    def optimize(self) -> dict:
        return {"ok": True}

    def health_check(self) -> dict:
        return {"ok": True}

    def update_metadatas(self, ids, metadatas) -> None:
        return None


def test_list_available_providers_contains_chromadb() -> None:
    assert "chromadb" in list_available_providers()


def test_create_provider_from_registered_class(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(PROVIDER_REGISTRY, "dummy", DummyProvider)
    provider = create_vectordb_provider(
        VectorDBConfig(provider="dummy", persist_dir=tmp_path / ".db", collection="c")
    )
    assert isinstance(provider, DummyProvider)


def test_create_provider_rejects_unknown_provider(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported vector DB provider"):
        create_vectordb_provider(
            VectorDBConfig(provider="missing", persist_dir=tmp_path / ".db", collection="c")
        )


def test_register_provider_rejects_non_subclass() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseVectorDBProvider"):
        register_provider("bad", object)


def test_register_provider_adds_to_registry(monkeypatch) -> None:
    monkeypatch.setitem(PROVIDER_REGISTRY, "dummy", "placeholder")
    register_provider("dummy", DummyProvider)
    assert PROVIDER_REGISTRY["dummy"] is DummyProvider
