from lsm.vectordb.base import VectorDBQueryResult


def test_query_result_as_dict() -> None:
    result = VectorDBQueryResult(
        ids=["a"],
        documents=["doc"],
        metadatas=[{"k": "v"}],
        distances=[0.1],
    )

    assert result.as_dict()["ids"] == ["a"]

