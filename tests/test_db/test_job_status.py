"""Tests for startup advisory checks (Phase 16.2)."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from lsm.db.job_status import (
    Advisory,
    _check_cluster_status,
    _check_graph_status,
    _check_finetune_status,
    _get_corpus_size,
    _get_job_status,
    check_job_advisories,
    record_job_status,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory SQLite database with required tables."""
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS lsm_chunks (
            chunk_id TEXT PRIMARY KEY,
            is_current INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS lsm_job_status (
            job_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            corpus_size INTEGER,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS lsm_embedding_models (
            model_id TEXT PRIMARY KEY,
            base_model TEXT,
            path TEXT,
            dimension INTEGER,
            created_at TEXT,
            is_active INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS lsm_graph_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_id TEXT,
            dst_id TEXT,
            edge_type TEXT,
            weight REAL DEFAULT 1.0
        );
    """)
    return db


def _insert_chunks(conn, count):
    for i in range(count):
        conn.execute(
            "INSERT INTO lsm_chunks (chunk_id, is_current) VALUES (?, 1)",
            (f"chunk_{i}",),
        )
    conn.commit()


def _make_config(
    cluster_enabled: bool = False,
    graph_enabled: bool = False,
    finetune_enabled: bool = True,
):
    return SimpleNamespace(
        query=SimpleNamespace(
            cluster_enabled=cluster_enabled,
            graph_expansion_enabled=graph_enabled,
        ),
        global_settings=SimpleNamespace(
            finetune_enabled=finetune_enabled,
        ),
    )


# ------------------------------------------------------------------
# Tests: get_corpus_size
# ------------------------------------------------------------------


class TestGetCorpusSize:
    def test_empty_corpus(self, conn):
        assert _get_corpus_size(conn) == 0

    def test_counts_current_only(self, conn):
        _insert_chunks(conn, 5)
        conn.execute("INSERT INTO lsm_chunks (chunk_id, is_current) VALUES ('old', 0)")
        conn.commit()
        assert _get_corpus_size(conn) == 5


# ------------------------------------------------------------------
# Tests: get_job_status
# ------------------------------------------------------------------


class TestGetJobStatus:
    def test_missing_job(self, conn):
        assert _get_job_status(conn, "nonexistent") is None

    def test_existing_job(self, conn):
        record_job_status(conn, "test_job", corpus_size=100)
        status = _get_job_status(conn, "test_job")
        assert status is not None
        assert status["status"] == "completed"
        assert status["corpus_size"] == 100


# ------------------------------------------------------------------
# Tests: record_job_status
# ------------------------------------------------------------------


class TestRecordJobStatus:
    def test_records_status(self, conn):
        record_job_status(conn, "cluster_build", corpus_size=50)
        row = conn.execute(
            "SELECT status, corpus_size FROM lsm_job_status WHERE job_name = ?",
            ("cluster_build",),
        ).fetchone()
        assert row[0] == "completed"
        assert row[1] == 50

    def test_overwrites_on_rerun(self, conn):
        record_job_status(conn, "cluster_build", corpus_size=50)
        record_job_status(conn, "cluster_build", corpus_size=100)
        row = conn.execute(
            "SELECT corpus_size FROM lsm_job_status WHERE job_name = ?",
            ("cluster_build",),
        ).fetchone()
        assert row[0] == 100


# ------------------------------------------------------------------
# Tests: check_cluster_status
# ------------------------------------------------------------------


class TestCheckClusterStatus:
    def test_no_advisory_when_disabled(self, conn):
        config = _make_config(cluster_enabled=False)
        result = _check_cluster_status(conn, config)
        assert result == []

    def test_advisory_when_never_run(self, conn):
        _insert_chunks(conn, 10)
        config = _make_config(cluster_enabled=True)
        result = _check_cluster_status(conn, config)
        assert len(result) == 1
        assert "never been built" in result[0].message
        assert result[0].action == "lsm cluster build"

    def test_no_advisory_when_current(self, conn):
        _insert_chunks(conn, 100)
        record_job_status(conn, "cluster_build", corpus_size=100)
        config = _make_config(cluster_enabled=True)
        result = _check_cluster_status(conn, config)
        assert result == []

    def test_advisory_when_stale(self, conn):
        _insert_chunks(conn, 130)
        record_job_status(conn, "cluster_build", corpus_size=100)
        config = _make_config(cluster_enabled=True)
        result = _check_cluster_status(conn, config)
        assert len(result) == 1
        assert "grown" in result[0].message

    def test_no_advisory_when_growth_under_20pct(self, conn):
        _insert_chunks(conn, 115)
        record_job_status(conn, "cluster_build", corpus_size=100)
        config = _make_config(cluster_enabled=True)
        result = _check_cluster_status(conn, config)
        assert result == []


# ------------------------------------------------------------------
# Tests: check_finetune_status
# ------------------------------------------------------------------


class TestCheckFinetuneStatus:
    def test_no_advisory_small_corpus(self, conn):
        _insert_chunks(conn, 50)
        result = _check_finetune_status(conn, None)
        assert result == []

    def test_advisory_large_corpus_no_model(self, conn):
        _insert_chunks(conn, 200)
        result = _check_finetune_status(conn, None)
        assert len(result) == 1
        assert "fine-tuned" in result[0].message
        assert result[0].action == "lsm finetune train"

    def test_no_advisory_with_active_model(self, conn):
        _insert_chunks(conn, 200)
        conn.execute(
            "INSERT INTO lsm_embedding_models (model_id, is_active) VALUES ('m1', 1)",
        )
        conn.commit()
        result = _check_finetune_status(conn, None)
        assert result == []

    def test_no_advisory_when_feature_disabled(self, conn):
        _insert_chunks(conn, 200)
        config = _make_config(finetune_enabled=False)
        result = _check_finetune_status(conn, config)
        assert result == []


# ------------------------------------------------------------------
# Tests: check_graph_status
# ------------------------------------------------------------------


class TestCheckGraphStatus:
    def test_no_advisory_when_disabled(self, conn):
        config = _make_config(graph_enabled=False)
        result = _check_graph_status(conn, config)
        assert result == []

    def test_advisory_when_enabled_and_no_thematic_edges(self, conn):
        _insert_chunks(conn, 25)
        config = _make_config(graph_enabled=True)
        result = _check_graph_status(conn, config)
        assert len(result) == 1
        assert result[0].action == "lsm graph build-links"

    def test_no_advisory_when_thematic_edges_exist(self, conn):
        _insert_chunks(conn, 25)
        conn.execute(
            "INSERT INTO lsm_graph_edges (src_id, dst_id, edge_type, weight) "
            "VALUES ('a', 'b', 'thematic', 0.9)"
        )
        conn.commit()
        config = _make_config(graph_enabled=True)
        result = _check_graph_status(conn, config)
        assert result == []


# ------------------------------------------------------------------
# Tests: check_job_advisories (integration)
# ------------------------------------------------------------------


class TestCheckJobAdvisories:
    def test_no_advisories_empty(self, conn):
        result = check_job_advisories(conn)
        assert result == []

    def test_combines_advisories(self, conn):
        _insert_chunks(conn, 200)
        config = _make_config(cluster_enabled=True)
        result = check_job_advisories(conn, config)
        # Should get both cluster (never run) and finetune (no active model)
        assert len(result) == 2
        actions = {a.action for a in result}
        assert "lsm cluster build" in actions
        assert "lsm finetune train" in actions

    def test_no_advisories_all_current(self, conn):
        _insert_chunks(conn, 200)
        record_job_status(conn, "cluster_build", corpus_size=200)
        conn.execute(
            "INSERT INTO lsm_graph_edges (src_id, dst_id, edge_type, weight) "
            "VALUES ('a', 'b', 'thematic', 0.9)"
        )
        conn.execute(
            "INSERT INTO lsm_embedding_models (model_id, is_active) VALUES ('m1', 1)",
        )
        conn.commit()
        config = _make_config(cluster_enabled=True, graph_enabled=True)
        result = check_job_advisories(conn, config)
        assert result == []

    def test_combines_graph_advisory_when_graph_enabled(self, conn):
        _insert_chunks(conn, 50)
        config = _make_config(graph_enabled=True)
        result = check_job_advisories(conn, config)
        actions = {a.action for a in result}
        assert "lsm graph build-links" in actions
