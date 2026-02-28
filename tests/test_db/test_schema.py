from __future__ import annotations

import sqlite3
from pathlib import Path

from lsm.db.schema import APPLICATION_TABLES, ensure_application_schema


def _in_memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def test_ensure_application_schema_creates_all_tables() -> None:
    conn = _in_memory_conn()
    ensure_application_schema(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    names = {str(row["name"]) for row in rows}

    for table in APPLICATION_TABLES:
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
    # (sqlite_master also includes internal SQLite tables)
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
