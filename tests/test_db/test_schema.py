"""Tests for lsm.db.schema — application schema DDL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lsm.db.schema import APPLICATION_TABLES, ensure_application_schema
from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames


def _in_memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ------------------------------------------------------------------
# SQLite tests
# ------------------------------------------------------------------


def test_ensure_application_schema_creates_all_14_tables() -> None:
    """Verify all 14 application tables are created."""
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    names = {str(row["name"]) for row in rows}

    tn = DEFAULT_TABLE_NAMES
    expected = {
        tn.chunks,
        tn.schema_versions,
        tn.manifest,
        tn.reranker_cache,
        tn.agent_memories,
        tn.agent_memory_candidates,
        tn.agent_schedules,
        tn.cluster_centroids,
        tn.graph_nodes,
        tn.graph_edges,
        tn.embedding_models,
        tn.job_status,
        tn.stats_cache,
        tn.remote_cache,
    }
    assert len(expected) == 14, "Expected exactly 14 application tables"
    for table in expected:
        assert table in names, f"Missing table: {table}"
    conn.close()


def test_ensure_application_schema_is_idempotent() -> None:
    conn = _in_memory_conn()
    ensure_application_schema(conn)
    # Insert a row to verify second call doesn't drop tables
    conn.execute(
        "INSERT INTO lsm_job_status(job_name, status) VALUES ('test', 'ok')"
    )
    conn.commit()

    # Call again — should not raise or destroy data
    ensure_application_schema(conn)

    row = conn.execute(
        "SELECT status FROM lsm_job_status WHERE job_name = 'test'"
    ).fetchone()
    assert row is not None
    assert row["status"] == "ok"
    conn.close()


def test_application_tables_constant_matches_created_tables() -> None:
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    names = {str(row["name"]) for row in rows}

    # APPLICATION_TABLES should be a subset of created tables
    table_set = set(APPLICATION_TABLES)
    assert table_set.issubset(names)
    # And every non-internal table should be in APPLICATION_TABLES
    internal = {"sqlite_sequence"}
    app_tables = names - internal
    assert app_tables == table_set
    conn.close()


def test_indexes_are_created() -> None:
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
    ).fetchall()
    index_names = {str(row["name"]) for row in rows}

    expected_indexes = {
        "idx_lsm_chunks_source_path",
        "idx_lsm_chunks_is_current",
        "idx_lsm_chunks_ext",
        "idx_lsm_manifest_updated_at",
        "idx_lsm_agent_memory_candidates_status",
        "idx_lsm_agent_memories_scope_type",
        "idx_lsm_agent_memories_expires_at",
        "idx_lsm_agent_schedules_next_run",
        "idx_lsm_graph_edges_src",
        "idx_lsm_graph_edges_dst",
        "idx_lsm_stats_cache_cached_at",
        "idx_lsm_remote_cache_provider",
        "idx_lsm_remote_cache_expires_at",
    }
    for idx in expected_indexes:
        assert idx in index_names, f"Missing index: {idx}"
    conn.close()


def test_custom_prefix() -> None:
    """Schema creation works with a custom table prefix."""
    conn = _in_memory_conn()
    tn = TableNames(prefix="test_")
    ensure_application_schema(conn, table_names=tn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    names = {str(row["name"]) for row in rows}

    assert "test_chunks" in names
    assert "test_job_status" in names
    assert "lsm_chunks" not in names
    conn.close()


def test_graph_nodes_source_path_index() -> None:
    """Verify graph_nodes source_path index exists."""
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_lsm_graph_nodes_source_path'"
    ).fetchall()
    assert len(rows) == 1
    conn.close()


def test_chunks_table_has_expected_columns() -> None:
    """Verify chunks table has all 25+ expected columns."""
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    cursor = conn.execute("PRAGMA table_info(lsm_chunks)")
    columns = {row["name"] for row in cursor.fetchall()}

    expected_columns = {
        "chunk_id", "source_path", "source_name", "chunk_text",
        "heading", "heading_path", "page_number", "paragraph_index",
        "mtime_ns", "file_hash", "version", "is_current",
        "node_type", "root_tags", "folder_tags", "content_type",
        "cluster_id", "cluster_size", "simhash", "ext",
        "chunk_index", "ingested_at", "start_char", "end_char",
        "chunk_length", "metadata_json",
    }
    for col in expected_columns:
        assert col in columns, f"Missing column: {col}"
    conn.close()


def test_schema_versions_has_autoincrement() -> None:
    """Verify schema_versions uses autoincrement on SQLite."""
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    # Insert without specifying id
    conn.execute(
        "INSERT INTO lsm_schema_versions (lsm_version) VALUES ('0.8.0')"
    )
    conn.execute(
        "INSERT INTO lsm_schema_versions (lsm_version) VALUES ('0.8.1')"
    )
    conn.commit()

    rows = conn.execute(
        "SELECT id FROM lsm_schema_versions ORDER BY id"
    ).fetchall()
    ids = [row["id"] for row in rows]
    assert ids == [1, 2]
    conn.close()
