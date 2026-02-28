import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.chromadb import ChromaDBProvider
from lsm.vectordb.base import VectorDBGetResult, VectorDBQueryResult


def _provider(tmp_path) -> ChromaDBProvider:
    return ChromaDBProvider(
        VectorDBConfig(provider="chromadb", path=tmp_path / ".chroma", collection="kb")
    )


def test_get_max_batch_size_uses_collection_client_value(tmp_path) -> None:
    provider = _provider(tmp_path)
    collection = type(
        "Collection",
        (),
        {"_client": type("Client", (), {"get_max_batch_size": lambda self: 321})()},
    )()
    assert provider._get_max_batch_size(collection) == 321


def test_get_max_batch_size_falls_back_to_default(tmp_path) -> None:
    provider = _provider(tmp_path)
    collection = type("Collection", (), {"_client": object(), "max_batch_size": None})()
    assert provider._get_max_batch_size(collection) == 4000


def test_add_chunks_returns_early_for_empty_ids(tmp_path, mocker) -> None:
    provider = _provider(tmp_path)
    ensure_mock = mocker.patch.object(provider, "_ensure_collection")
    provider.add_chunks([], [], [], [])
    ensure_mock.assert_not_called()


def test_add_chunks_validates_lengths(tmp_path) -> None:
    provider = _provider(tmp_path)
    with pytest.raises(ValueError, match="must have the same length"):
        provider.add_chunks(ids=["1"], documents=[], metadatas=[], embeddings=[])


def test_add_chunks_uses_upsert_when_available(tmp_path, mocker) -> None:
    provider = _provider(tmp_path)
    collection = mocker.MagicMock()
    collection.upsert.return_value = None
    mocker.patch.object(provider, "_ensure_collection", return_value=collection)
    mocker.patch.object(provider, "_get_max_batch_size", return_value=2)

    provider.add_chunks(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{}, {}, {}],
        embeddings=[[0.1], [0.2], [0.3]],
    )

    assert collection.upsert.call_count == 2


def test_add_chunks_falls_back_to_add_when_upsert_missing(tmp_path, mocker) -> None:
    provider = _provider(tmp_path)
    collection = mocker.MagicMock()
    collection.upsert.side_effect = AttributeError("no upsert")
    mocker.patch.object(provider, "_ensure_collection", return_value=collection)
    mocker.patch.object(provider, "_get_max_batch_size", return_value=10)

    provider.add_chunks(
        ids=["1"],
        documents=["a"],
        metadatas=[{}],
        embeddings=[[0.1]],
    )

    collection.add.assert_called_once()


def test_query_normalizes_response_values(tmp_path, mocker) -> None:
    provider = _provider(tmp_path)
    collection = mocker.MagicMock()
    collection.query.return_value = {
        "ids": [[123]],
        "documents": [[None]],
        "metadatas": [[None]],
        "distances": [[0.42]],
    }
    mocker.patch.object(provider, "_ensure_collection", return_value=collection)

    result = provider.query([0.1, 0.2], 5, filters={"ext": ".txt"})

    assert isinstance(result, VectorDBQueryResult)
    assert result.ids == ["123"]
    assert result.documents == [""]
    assert result.metadatas == [{}]
    assert result.distances == [0.42]
    collection.query.assert_called_once()


def test_delete_by_filter_requires_non_empty_filters(tmp_path) -> None:
    provider = _provider(tmp_path)
    with pytest.raises(ValueError, match="non-empty dict"):
        provider.delete_by_filter({})


def test_health_check_reports_error(tmp_path, mocker) -> None:
    provider = _provider(tmp_path)
    mocker.patch.object(provider, "count", side_effect=RuntimeError("boom"))
    health = provider.health_check()
    assert health["status"] == "error"
    assert "boom" in health["error"]


class TestChromaDBGet:
    """Tests for ChromaDBProvider.get() method."""

    def test_get_by_ids(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {
            "ids": ["a", "b"],
            "metadatas": [{"k": "v1"}, {"k": "v2"}],
        }
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        result = provider.get(ids=["a", "b"])

        assert isinstance(result, VectorDBGetResult)
        assert result.ids == ["a", "b"]
        assert result.metadatas == [{"k": "v1"}, {"k": "v2"}]
        assert result.documents is None
        collection.get.assert_called_once_with(include=["metadatas"], ids=["a", "b"])

    def test_get_by_filter(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {
            "ids": ["x"],
            "metadatas": [{"source_path": "/test"}],
        }
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        result = provider.get(filters={"source_path": "/test"})

        assert result.ids == ["x"]
        collection.get.assert_called_once_with(
            include=["metadatas"], where={"source_path": "/test"}
        )

    def test_get_with_limit_and_offset(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {"ids": ["a"], "metadatas": [{}]}
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        provider.get(limit=10, offset=5, include=["metadatas"])

        collection.get.assert_called_once_with(
            include=["metadatas"], limit=10, offset=5
        )

    def test_get_with_documents_include(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {
            "ids": ["a"],
            "documents": ["doc text"],
            "metadatas": [{"k": "v"}],
        }
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        result = provider.get(ids=["a"], include=["documents", "metadatas"])

        assert result.documents == ["doc text"]
        assert result.metadatas == [{"k": "v"}]

    def test_get_offset_fallback_on_type_error(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        # First call with offset raises TypeError, second without succeeds
        collection.get.side_effect = [
            TypeError("unexpected keyword argument 'offset'"),
            {"ids": ["a"], "metadatas": [{}]},
        ]
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        result = provider.get(limit=10, offset=5)

        assert result.ids == ["a"]
        assert collection.get.call_count == 2

    def test_get_empty_result(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {"ids": [], "metadatas": []}
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        result = provider.get(ids=["nonexistent"])

        assert result.ids == []


class TestChromaDBDeleteAll:
    """Tests for ChromaDBProvider.delete_all() method."""

    def test_delete_all_returns_count(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {"ids": ["a", "b", "c"]}
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        count = provider.delete_all()

        assert count == 3
        collection.delete.assert_called_once_with(ids=["a", "b", "c"])

    def test_delete_all_empty_collection(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        collection.get.return_value = {"ids": []}
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        count = provider.delete_all()

        assert count == 0
        collection.delete.assert_not_called()


class TestChromaDBUpdateMetadatas:
    """Tests for ChromaDBProvider.update_metadatas() method."""

    def test_update_metadatas_skips_empty(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        ensure_mock = mocker.patch.object(provider, "_ensure_collection")
        provider.update_metadatas([], [])
        ensure_mock.assert_not_called()

    def test_update_metadatas_delegates_to_collection(self, tmp_path, mocker) -> None:
        provider = _provider(tmp_path)
        collection = mocker.MagicMock()
        mocker.patch.object(provider, "_ensure_collection", return_value=collection)

        provider.update_metadatas(["a"], [{"new_key": "val"}])

        collection.update.assert_called_once_with(
            ids=["a"], metadatas=[{"new_key": "val"}]
        )
