import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import BaseVectorDBProvider, VectorDBQueryResult


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
