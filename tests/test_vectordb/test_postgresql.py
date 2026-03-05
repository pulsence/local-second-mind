"""Tests for PostgreSQL + pgvector provider (mocked — no live DB required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from lsm.config.models import DBConfig
from lsm.vectordb.base import VectorDBGetResult, VectorDBQueryResult


def _pg_config(**overrides) -> DBConfig:
    defaults = dict(
        provider="postgresql",
        connection_string="postgresql://u:p@localhost/testdb",
        collection="test_kb",
    )
    defaults.update(overrides)
    return DBConfig(**defaults)


@pytest.fixture()
def mock_pg_deps(monkeypatch):
    """Patch psycopg2 and pgvector imports so PostgreSQLProvider can be instantiated."""
    import lsm.vectordb.postgresql as pg_mod

    mock_pool_mod = MagicMock()
    mock_sql_mod = MagicMock()
    mock_execute_values = MagicMock()
    mock_register_vector = MagicMock()

    mock_psycopg2_mod = MagicMock()
    mock_psycopg2_mod.OperationalError = type("OperationalError", (Exception,), {})

    monkeypatch.setattr(pg_mod, "psycopg2", mock_psycopg2_mod)
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
        "psycopg2": mock_psycopg2_mod,
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


class TestPostgreSQLPrune:
    """Tests for prune_old_versions() — PostgreSQL parity."""

    def test_prune_returns_zero_when_no_table(self, provider, mocker):
        from lsm.vectordb.base import PruneCriteria

        mock_conn = MagicMock()
        mocker.patch.object(provider, "_table_exists", return_value=False)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.prune_old_versions(PruneCriteria())

        assert result == 0

    def test_prune_validates_max_versions(self, provider):
        from lsm.vectordb.base import PruneCriteria

        with pytest.raises(ValueError, match="max_versions must be >= 1"):
            provider.prune_old_versions(PruneCriteria(max_versions=0))

    def test_prune_validates_older_than_days(self, provider):
        from lsm.vectordb.base import PruneCriteria

        with pytest.raises(ValueError, match="older_than_days must be >= 0"):
            provider.prune_old_versions(PruneCriteria(older_than_days=-1))


class TestPostgreSQLGraphInsert:
    """Tests for graph_insert_nodes/edges — PostgreSQL parity."""

    def test_graph_insert_nodes_empty(self, provider, mocker):
        mocker.patch.object(provider, "_get_conn")
        provider.graph_insert_nodes([])
        # Should not try to get a connection

    def test_graph_insert_nodes_creates_tables_and_inserts(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            provider.graph_insert_nodes([
                {"node_id": "n1", "node_type": "file", "label": "test.md", "source_path": "test.md"},
            ])

        # Should have executed CREATE TABLE (graph tables) + INSERT
        assert mock_cursor.execute.call_count >= 1
        mock_conn.commit.assert_called()

    def test_graph_insert_edges_empty(self, provider, mocker):
        mocker.patch.object(provider, "_get_conn")
        provider.graph_insert_edges([])

    def test_graph_insert_edges_inserts_data(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            provider.graph_insert_edges([
                {"src_id": "n1", "dst_id": "n2", "edge_type": "contains", "weight": 1.0},
            ])

        assert mock_cursor.execute.call_count >= 1
        mock_conn.commit.assert_called()


class TestPostgreSQLGraphTraverse:
    """Tests for graph_traverse() — PostgreSQL parity."""

    def test_traverse_empty_start(self, provider):
        result = provider.graph_traverse([])
        assert result == []

    def test_traverse_returns_reachable_nodes(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("n1",), ("n2",), ("n3",)]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.graph_traverse(["n1"], max_hops=2)

        assert set(result) == {"n1", "n2", "n3"}
        mock_cursor.execute.assert_called()

    def test_traverse_with_edge_types(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("n1",), ("n2",)]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.graph_traverse(
                ["n1"], max_hops=1, edge_types=["contains"]
            )

        assert set(result) == {"n1", "n2"}


class TestPostgreSQLEnsureDatabase:
    """Tests for _ensure_database() — auto-create database if missing."""

    def test_creates_database_when_missing_connection_string(self, mock_pg_deps):
        """Database is created when it doesn't exist (connection string config)."""
        from lsm.vectordb.postgresql import PostgreSQLProvider

        provider = PostgreSQLProvider(
            _pg_config(connection_string="postgresql://u:p@localhost/mydb")
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # DB does not exist
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_psycopg2 = mock_pg_deps["psycopg2"]
        mock_psycopg2.connect.return_value = mock_conn
        provider._ensure_database()

        # Should connect to "postgres" maintenance DB
        mock_psycopg2.connect.assert_called_once()
        dsn_arg = mock_psycopg2.connect.call_args
        assert "postgres" in str(dsn_arg)

        # Should check for DB existence and create it
        assert mock_cursor.execute.call_count == 2
        # First call: SELECT from pg_database
        first_call_args = mock_cursor.execute.call_args_list[0]
        assert "pg_database" in str(first_call_args)
        # Second call: CREATE DATABASE
        second_call_args = mock_cursor.execute.call_args_list[1]
        assert "mydb" in str(second_call_args)
        mock_conn.close.assert_called_once()

    def test_skips_creation_when_database_exists(self, mock_pg_deps):
        """No CREATE DATABASE when the database already exists."""
        from lsm.vectordb.postgresql import PostgreSQLProvider

        provider = PostgreSQLProvider(
            _pg_config(connection_string="postgresql://u:p@localhost/existingdb")
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)  # DB exists
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_psycopg2 = mock_pg_deps["psycopg2"]
        mock_psycopg2.connect.return_value = mock_conn
        provider._ensure_database()

        # Only the SELECT, no CREATE
        assert mock_cursor.execute.call_count == 1

    def test_creates_database_component_config(self, mock_pg_deps):
        """Database creation works with component-based config (host/port/database)."""
        from lsm.vectordb.postgresql import PostgreSQLProvider

        provider = PostgreSQLProvider(_pg_config(
            connection_string=None,
            host="dbhost",
            port=5432,
            database="newdb",
            user="admin",
            password="secret",
        ))

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # DB does not exist
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_psycopg2 = mock_pg_deps["psycopg2"]
        mock_psycopg2.connect.return_value = mock_conn
        provider._ensure_database()

        # Should connect to "postgres" database on same host
        connect_kwargs = mock_psycopg2.connect.call_args[1]
        assert connect_kwargs["database"] == "postgres"
        assert connect_kwargs["host"] == "dbhost"
        assert connect_kwargs["user"] == "admin"
        # Should CREATE DATABASE
        assert mock_cursor.execute.call_count == 2

    def test_graceful_failure_on_operational_error(self, mock_pg_deps):
        """OperationalError connecting to maintenance DB is silently ignored."""
        from lsm.vectordb.postgresql import PostgreSQLProvider

        provider = PostgreSQLProvider(
            _pg_config(connection_string="postgresql://u:p@unreachable/mydb")
        )

        mock_psycopg2 = mock_pg_deps["psycopg2"]
        mock_psycopg2.connect.side_effect = mock_psycopg2.OperationalError("conn refused")
        # Should not raise
        provider._ensure_database()

    def test_no_database_name_is_noop(self, mock_pg_deps):
        """Empty database name in connection string is a no-op."""
        from lsm.vectordb.postgresql import PostgreSQLProvider

        provider = PostgreSQLProvider(_pg_config(connection_string="postgresql://u:p@host/"))

        mock_psycopg2 = mock_pg_deps["psycopg2"]
        provider._ensure_database()

        mock_psycopg2.connect.assert_not_called()


class TestPostgreSQLEnsurePoolExtension:
    """Tests for _ensure_pool() calling _ensure_extension() eagerly."""

    def test_ensure_pool_calls_ensure_extension(self, mock_pg_deps):
        """_ensure_pool() should call _ensure_extension() after pool creation."""
        from lsm.vectordb.postgresql import PostgreSQLProvider

        provider = PostgreSQLProvider(_pg_config())

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(provider, "_ensure_database"):
            # Mock the pool to return a mock connection
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_conn
            mock_pg_deps["pool"].ThreadedConnectionPool.return_value = mock_pool_instance

            provider._ensure_pool()

        # _ensure_extension runs CREATE EXTENSION IF NOT EXISTS vector
        mock_cursor.execute.assert_called()
        assert any(
            "vector" in str(c) for c in mock_cursor.execute.call_args_list
        )


class TestPostgreSQLFTSQuery:
    """Tests for fts_query() method (PostgreSQL full-text search parity)."""

    def test_empty_text_returns_empty(self, provider):
        result = provider.fts_query("", 5)
        assert result.ids == []
        assert result.documents == []

    def test_whitespace_text_returns_empty(self, provider):
        result = provider.fts_query("   ", 5)
        assert result.ids == []

    def test_zero_top_k_returns_empty(self, provider):
        result = provider.fts_query("search query", 0)
        assert result.ids == []

    def test_negative_top_k_returns_empty(self, provider):
        result = provider.fts_query("search query", -1)
        assert result.ids == []

    def test_returns_empty_when_no_table(self, provider, mocker):
        mock_conn = MagicMock()
        mocker.patch.object(provider, "_table_exists", return_value=False)
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.fts_query("search term", 5)
        assert result.ids == []

    def test_fts_query_executes_tsquery(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("id1", "doc one text", {"source": "a.md"}, -0.5),
            ("id2", "doc two text", {"source": "b.md"}, -0.3),
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        mocker.patch.object(provider, "_ensure_fts")
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.fts_query("search term", 5)

        assert result.ids == ["id1", "id2"]
        assert result.documents == ["doc one text", "doc two text"]
        assert len(result.distances) == 2
        mock_cursor.execute.assert_called_once()

    def test_fts_query_returns_empty_on_no_match(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        mocker.patch.object(provider, "_ensure_fts")
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.fts_query("nonexistent query", 5)

        assert result.ids == []
        assert result.documents == []

    def test_fts_query_result_structure(self, provider, mocker):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("chunk_abc", "The quick brown fox", {"key": "val"}, -0.8),
        ]
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mocker.patch.object(provider, "_table_exists", return_value=True)
        mocker.patch.object(provider, "_ensure_fts")
        with patch.object(provider, "_get_conn") as ctx:
            ctx.return_value.__enter__ = lambda s: mock_conn
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = provider.fts_query("fox", 10)

        assert isinstance(result, VectorDBQueryResult)
        assert result.ids == ["chunk_abc"]
        assert result.documents == ["The quick brown fox"]
        assert result.metadatas == [{"key": "val"}]
        assert result.distances == [-0.8]
