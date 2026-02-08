import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import BaseVectorDBProvider, VectorDBGetResult, VectorDBQueryResult


def test_query_result_as_dict() -> None:
    result = VectorDBQueryResult(
        ids=["a"],
        documents=["doc"],
        metadatas=[{"k": "v"}],
        distances=[0.1],
    )

    assert result.as_dict()["ids"] == ["a"]


def test_query_result_empty_values() -> None:
    result = VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])
    assert result.as_dict() == {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "distances": [],
    }


def test_base_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseVectorDBProvider(VectorDBConfig())


class TestVectorDBGetResult:
    """Tests for VectorDBGetResult dataclass."""

    def test_default_construction(self) -> None:
        result = VectorDBGetResult()
        assert result.ids == []
        assert result.documents is None
        assert result.metadatas is None
        assert result.embeddings is None

    def test_full_construction(self) -> None:
        result = VectorDBGetResult(
            ids=["a", "b"],
            documents=["doc1", "doc2"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert result.ids == ["a", "b"]
        assert result.documents == ["doc1", "doc2"]
        assert result.metadatas == [{"k": "v1"}, {"k": "v2"}]
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]

    def test_as_dict_ids_only(self) -> None:
        result = VectorDBGetResult(ids=["a"])
        d = result.as_dict()
        assert d == {"ids": ["a"]}
        assert "documents" not in d
        assert "metadatas" not in d
        assert "embeddings" not in d

    def test_as_dict_with_documents(self) -> None:
        result = VectorDBGetResult(ids=["a"], documents=["doc1"])
        d = result.as_dict()
        assert d == {"ids": ["a"], "documents": ["doc1"]}

    def test_as_dict_with_metadatas(self) -> None:
        result = VectorDBGetResult(ids=["a"], metadatas=[{"k": "v"}])
        d = result.as_dict()
        assert d == {"ids": ["a"], "metadatas": [{"k": "v"}]}

    def test_as_dict_full(self) -> None:
        result = VectorDBGetResult(
            ids=["a"],
            documents=["doc1"],
            metadatas=[{"k": "v"}],
            embeddings=[[0.1]],
        )
        d = result.as_dict()
        assert d == {
            "ids": ["a"],
            "documents": ["doc1"],
            "metadatas": [{"k": "v"}],
            "embeddings": [[0.1]],
        }

    def test_empty_result(self) -> None:
        result = VectorDBGetResult(ids=[])
        assert result.as_dict() == {"ids": []}
