from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.sqlite_vec import SQLiteVecProvider


def _vector(x: float, y: float = 0.0) -> list[float]:
    values = [0.0] * 384
    values[0] = float(x)
    values[1] = float(y)
    return values


def _provider(tmp_path: Path) -> SQLiteVecProvider:
    cfg = VectorDBConfig(provider="sqlite", path=tmp_path / "db", collection="test")
    return SQLiteVecProvider(cfg)


def test_schema_creation_on_fresh_database(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    expected_tables = {
        "lsm_chunks",
        "vec_chunks",
        "chunks_fts",
        "lsm_manifest",
        "lsm_schema_versions",
        "lsm_reranker_cache",
        "lsm_agent_memories",
        "lsm_agent_memory_candidates",
        "lsm_agent_schedules",
        "lsm_cluster_centroids",
        "lsm_graph_nodes",
        "lsm_graph_edges",
        "lsm_embedding_models",
        "lsm_job_status",
    }

    rows = provider.connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
    ).fetchall()
    names = {str(row["name"]) for row in rows}
    assert expected_tables.issubset(names)


def test_add_chunks_is_atomic_across_all_tables(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    ids = ["doc-1", "doc-2"]
    docs = ["hello world", "second row"]
    metas = [
        {"source_path": "/docs/a.md", "source_name": "a.md", "ext": ".md", "is_current": True},
        {"source_name": "missing-source-path"},  # triggers rollback
    ]
    vectors = [_vector(1.0), _vector(0.0, 1.0)]

    with pytest.raises(ValueError, match="metadata.source_path is required"):
        provider.add_chunks(ids, docs, metas, vectors)

    chunks_count = provider.connection.execute("SELECT COUNT(*) AS n FROM lsm_chunks").fetchone()["n"]
    vec_count = provider.connection.execute("SELECT COUNT(*) AS n FROM vec_chunks").fetchone()["n"]
    fts_count = provider.connection.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    assert chunks_count == 0
    assert vec_count == 0
    assert fts_count == 0


def test_query_returns_knn_distance_ordering(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    ids = ["a", "b", "c"]
    docs = ["alpha", "beta", "gamma"]
    metas = [
        {"source_path": "/docs/a.md", "ext": ".md", "is_current": True},
        {"source_path": "/docs/b.md", "ext": ".md", "is_current": True},
        {"source_path": "/docs/c.md", "ext": ".md", "is_current": True},
    ]
    vectors = [_vector(1.0, 0.0), _vector(0.0, 1.0), _vector(0.5, 0.5)]
    provider.add_chunks(ids, docs, metas, vectors)

    result = provider.query(_vector(1.0, 0.0), top_k=3)
    assert result.ids[0] == "a"
    assert result.ids[1] == "c"
    assert result.distances[0] <= result.distances[1] <= result.distances[2]


def test_query_metadata_filters_path_ext_and_current(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    ids = ["one", "two", "three"]
    docs = ["one", "two", "three"]
    metas = [
        {"source_path": "/notes/alpha.md", "ext": ".md", "is_current": True},
        {"source_path": "/notes/beta.txt", "ext": ".txt", "is_current": True},
        {"source_path": "/archive/alpha.md", "ext": ".md", "is_current": False},
    ]
    vectors = [_vector(1.0), _vector(0.9), _vector(0.8)]
    provider.add_chunks(ids, docs, metas, vectors)

    result = provider.query(
        _vector(1.0),
        top_k=10,
        filters={
            "path_contains": "notes",
            "ext_allow": [".md", ".txt"],
            "ext_deny": [".txt"],
            "is_current": True,
        },
    )
    assert result.ids == ["one"]


def test_delete_by_id_removes_from_chunk_vec_and_fts_tables(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.add_chunks(
        ids=["a", "b"],
        documents=["alpha", "beta"],
        metadatas=[
            {"source_path": "/docs/a.md", "ext": ".md", "is_current": True},
            {"source_path": "/docs/b.md", "ext": ".md", "is_current": True},
        ],
        embeddings=[_vector(1.0), _vector(0.5)],
    )

    provider.delete_by_id(["a"])
    chunk_ids = {str(row["chunk_id"]) for row in provider.connection.execute("SELECT chunk_id FROM lsm_chunks")}
    vec_ids = {str(row["chunk_id"]) for row in provider.connection.execute("SELECT chunk_id FROM vec_chunks")}
    fts_ids = {str(row["chunk_id"]) for row in provider.connection.execute("SELECT chunk_id FROM chunks_fts")}
    assert "a" not in chunk_ids
    assert "a" not in vec_ids
    assert "a" not in fts_ids
    assert "b" in chunk_ids


def test_fts_content_sync_triggers_stay_consistent(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.add_chunks(
        ids=["x"],
        documents=["initial content"],
        metadatas=[{"source_path": "/docs/x.md", "ext": ".md", "is_current": True}],
        embeddings=[_vector(1.0)],
    )

    before = provider.connection.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    assert before == 1

    provider.connection.execute(
        "UPDATE lsm_chunks SET chunk_text = ? WHERE chunk_id = ?",
        ("updated phrase", "x"),
    )
    provider.connection.commit()
    row = provider.connection.execute(
        "SELECT chunk_text FROM chunks_fts WHERE chunk_id = ?",
        ("x",),
    ).fetchone()
    assert row is not None
    assert row["chunk_text"] == "updated phrase"

    provider.delete_by_id(["x"])
    after = provider.connection.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    assert after == 0


def test_wal_mode_is_set(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    mode = provider.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_concurrent_reads_do_not_block(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.add_chunks(
        ids=["c1"],
        documents=["concurrency"],
        metadatas=[{"source_path": "/docs/c1.md", "ext": ".md", "is_current": True}],
        embeddings=[_vector(1.0)],
    )

    db_path = Path(provider.connection.execute("PRAGMA database_list").fetchone()[2])
    started = threading.Event()
    done = threading.Event()

    def _writer() -> None:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE lsm_chunks SET source_name = ? WHERE chunk_id = ?", ("name", "c1"))
        started.set()
        time.sleep(0.4)
        conn.rollback()
        conn.close()
        done.set()

    thread = threading.Thread(target=_writer, daemon=True)
    thread.start()
    assert started.wait(timeout=2.0)

    reader = sqlite3.connect(str(db_path))
    t0 = time.perf_counter()
    _ = reader.execute("SELECT COUNT(*) FROM lsm_chunks").fetchone()[0]
    elapsed = time.perf_counter() - t0
    reader.close()

    assert elapsed < 0.2
    assert done.wait(timeout=2.0)


def test_health_check_passes_and_fails_for_missing_extension(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    health_ok = provider.health_check()
    assert health_ok["status"] == "ok"

    provider._extension_loaded = False
    health_fail = provider.health_check()
    assert health_fail["status"] == "error"
    assert "extension" in health_fail["error"]

