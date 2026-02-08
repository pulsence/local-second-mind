"""Tests for PostgreSQL + pgvector provider (mocked — no live DB required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import VectorDBGetResult, VectorDBQueryResult


def _pg_config(**overrides) -> VectorDBConfig:
    defaults = dict(
        provider="postgresql",
        connection_string="postgresql://u:p@localhost/testdb",
        collection="test_kb",
    )
    defaults.update(overrides)
    return VectorDBConfig(**defaults)


@pytest.fixture()
def mock_pg_deps(monkeypatch):
    """Patch psycopg2 and pgvector imports so PostgreSQLProvider can be instantiated."""
    import lsm.vectordb.postgresql as pg_mod

    mock_pool_mod = MagicMock()
    mock_sql_mod = MagicMock()
    mock_execute_values = MagicMock()
    mock_register_vector = MagicMock()

    monkeypatch.setattr(pg_mod, "pool", mock_pool_mod)
    monkeypatch.setattr(pg_mod, "sql", mock_sql_mod)
    monkeypatch.setattr(pg_mod, "execute_values", mock_execute_values)
    monkeypatch.setattr(pg_mod, "register_vector", mock_register_vector)
    monkeypatch.setattr(pg_mod, "PSYCOPG2_AVAILABLE", True)
    monkeypatch.setattr(pg_mod, "PGVECTOR_AVAILABLE", True)

    # Make sql.Identifier and sql.SQL return strings for easy assertion
    mock_sql_mod.Identifier.side_effect = lambda x: x
    mock_sql_mod.SQL.side_effect = lambda x: x
    mock_sql_mod.Literal.side_effect = lambda x: str(x)

    return {
        "pool": mock_pool_mod,
        "sql": mock_sql_mod,
        "execute_values": mock_execute_values,
        "register_vector": mock_register_vector,
    }


@pytest.fixture()
def provider(mock_pg_deps):
    from lsm.vectordb.postgresql import PostgreSQLProvider
    return PostgreSQLProvider(_pg_config())


class TestFilterNormalization:
    """Tests for _normalize_filters static method."""

    def test_simple_equality_passes_through(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        result = PostgreSQLProvider._normalize_filters({"key": "value"})
        assert result == {"key": "value"}

    def test_dollar_eq_normalized(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        result = PostgreSQLProvider._normalize_filters({"key": {"$eq": "value"}})
        assert result == {"key": "value"}

    def test_unsupported_operator_raises(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        with pytest.raises(ValueError, match="only supports \\$eq"):
            PostgreSQLProvider._normalize_filters({"key": {"$gt": 5}})

    def test_mixed_filters(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        result = PostgreSQLProvider._normalize_filters({
            "a": "plain",
            "b": {"$eq": "wrapped"},
        })
        assert result == {"a": "plain", "b": "wrapped"}

    def test_boolean_value_passes_through(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        result = PostgreSQLProvider._normalize_filters({"is_current": True})
        assert result == {"is_current": True}


class TestSanitizeTableName:
    """Tests for _sanitize_table_name static method."""

    def test_normal_name(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        assert PostgreSQLProvider._sanitize_table_name("local_kb") == "chunks_local_kb"

    def test_special_characters(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        assert PostgreSQLProvider._sanitize_table_name("my-collection!") == "chunks_my_collection"

    def test_empty_string(self):
        from lsm.vectordb.postgresql import PostgreSQLProvider
        assert PostgreSQLProvider._sanitize_table_name("") == "chunks_local_kb"


class TestPostgreSQLGet:
    """Tests for get() method."""

    def test_get_returns_empty_when_no_table(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mocker.patch.object(provider, "_get_conn")
        mocker.patch.object(provider, "_table_exists", return_value=False)

        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.get(ids=["a"])

        assert isinstance(result, VectorDBGetResult)
        assert result.ids == []

    def test_get_by_ids_builds_correct_query(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("a", {"key": "val1"}),
            ("b", {"key": "val2"}),
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.get(ids=["a", "b"], include=["metadatas"])

        assert result.ids == ["a", "b"]
        assert result.metadatas == [{"key": "val1"}, {"key": "val2"}]
        assert result.documents is None

    def test_get_with_documents(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("a", "hello world", {"k": "v"}),
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.get(
                ids=["a"],
                include=["documents", "metadatas"],
            )

        assert result.ids == ["a"]
        assert result.documents == ["hello world"]
        assert result.metadatas == [{"k": "v"}]


class TestPostgreSQLUpdateMetadatas:
    """Tests for update_metadatas() method."""

    def test_update_skips_empty_ids(self, provider, mocker):
        mock_conn = MagicMock()
        mocker.patch.object(provider, "_get_conn")
        provider.update_metadatas([], [])
        # Should not try to get a connection

    def test_update_validates_length_mismatch(self, provider):
        with pytest.raises(ValueError, match="same length"):
            provider.update_metadatas(["a"], [{"k": "v1"}, {"k": "v2"}])

    def test_update_executes_sql_for_each_id(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            provider.update_metadatas(
                ["a", "b"],
                [{"new": "val1"}, {"new": "val2"}],
            )

        assert mock_cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()


class TestPostgreSQLDeleteAll:
    """Tests for delete_all() method."""

    def test_delete_all_returns_zero_when_no_table(self, provider, mocker):
        mock_conn = MagicMock()
        mocker.patch.object(provider, "_table_exists", return_value=False)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.delete_all()

        assert result == 0

    def test_delete_all_returns_row_count(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 42
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.delete_all()

        assert result == 42
        mock_conn.commit.assert_called_once()


class TestPostgreSQLAddChunks:
    """Tests for add_chunks() method."""

    def test_add_chunks_returns_early_for_empty(self, provider, mocker):
        mocker.patch.object(provider, "_get_conn")
        provider.add_chunks([], [], [], [])

    def test_add_chunks_validates_lengths(self, provider):
        with pytest.raises(ValueError, match="same length"):
            provider.add_chunks(["a"], [], [], [])

    def test_add_chunks_validates_embedding_dimension(self, provider):
        with pytest.raises(ValueError, match="non-empty"):
            provider.add_chunks(["a"], ["doc"], [{}], [[]])


class TestPostgreSQLQuery:
    """Tests for query() method."""

    def test_query_returns_empty_for_no_embedding(self, provider):
        result = provider.query([], 5)
        assert result.ids == []

    def test_query_returns_empty_when_no_table(self, provider, mocker):
        mock_conn = MagicMock()
        mocker.patch.object(provider, "_table_exists", return_value=False)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.query([0.1, 0.2], 5)

        assert result.ids == []


class TestPostgreSQLDeleteByFilter:
    """Tests for delete_by_filter() method."""

    def test_delete_by_filter_requires_non_empty(self, provider):
        with pytest.raises(ValueError, match="non-empty dict"):
            provider.delete_by_filter({})

    def test_delete_by_filter_normalizes_operators(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            provider.delete_by_filter({"key": {"$eq": "val"}})

        # Should have been called with normalized filter
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        assert json.dumps({"key": "val"}) in str(call_args)


class TestPostgreSQLHealth:
    """Tests for health_check() method."""

    def test_health_check_missing_deps(self, provider, mocker, monkeypatch):
        import lsm.vectordb.postgresql as pg_mod
        mocker.patch.object(provider, "is_available", return_value=False)
        monkeypatch.setattr(pg_mod, "PSYCOPG2_AVAILABLE", False)
        monkeypatch.setattr(pg_mod, "PGVECTOR_AVAILABLE", False)

        health = provider.health_check()
        assert health["status"] == "missing_dependencies"
