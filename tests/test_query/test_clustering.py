"""
Tests for cluster-aware retrieval (Phase 14.2).

Validates cluster building, centroid queries, and cluster-filtered
retrieval integration.
"""

from __future__ import annotations

import sqlite3
from struct import pack, unpack
from typing import Any, Dict, List

import numpy as np
import pytest

from lsm.db.clustering import build_clusters, get_top_clusters


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with required schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE lsm_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_path TEXT,
            source_name TEXT,
            chunk_text TEXT,
            heading TEXT,
            heading_path TEXT,
            page_number TEXT,
            paragraph_index INTEGER,
            mtime_ns INTEGER,
            file_hash TEXT,
            version INTEGER,
            is_current INTEGER DEFAULT 1,
            node_type TEXT DEFAULT 'chunk',
            root_tags TEXT,
            folder_tags TEXT,
            content_type TEXT,
            cluster_id INTEGER,
            cluster_size INTEGER,
            simhash INTEGER,
            ext TEXT,
            chunk_index INTEGER,
            ingested_at TEXT,
            start_char INTEGER,
            end_char INTEGER,
            chunk_length INTEGER,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE lsm_cluster_centroids (
            cluster_id INTEGER PRIMARY KEY,
            centroid BLOB,
            size INTEGER
        );
    """)

    return conn


def _create_test_db_with_vec() -> sqlite3.Connection:
    """Create test DB with both lsm_chunks and a mock vec_chunks table.

    Since sqlite-vec extension is not available in tests, we use a regular
    table with the same column layout.
    """
    conn = _create_test_db()

    conn.execute("""
        CREATE TABLE vec_chunks (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB,
            is_current INTEGER DEFAULT 1,
            node_type TEXT,
            source_path TEXT,
            cluster_id INTEGER
        )
    """)

    return conn


def _insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    embedding: List[float],
    source_path: str = "/test.md",
) -> None:
    """Insert a chunk into both tables."""
    blob = pack(f"{len(embedding)}f", *embedding)
    conn.execute(
        "INSERT INTO lsm_chunks (chunk_id, source_path, chunk_text, metadata_json) "
        "VALUES (?, ?, ?, '{}')",
        (chunk_id, source_path, f"text for {chunk_id}"),
    )
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding, is_current, source_path) "
        "VALUES (?, ?, 1, ?)",
        (chunk_id, blob, source_path),
    )
    conn.commit()


# ------------------------------------------------------------------
# Tests: build_clusters
# ------------------------------------------------------------------

def test_build_clusters_assigns_ids():
    conn = _create_test_db_with_vec()

    # Insert 10 chunks with 4-dim embeddings
    np.random.seed(42)
    for i in range(10):
        vec = np.random.randn(4).astype(np.float32).tolist()
        _insert_chunk(conn, f"chunk_{i}", vec)

    result = build_clusters(conn, algorithm="kmeans", k=3)

    assert result["n_clusters"] == 3
    assert result["n_chunks"] == 10
    assert result["algorithm"] == "kmeans"

    # Verify all chunks have cluster_id assigned
    rows = conn.execute("SELECT cluster_id FROM lsm_chunks").fetchall()
    for row in rows:
        assert row[0] is not None
        assert 0 <= row[0] < 3


def test_build_clusters_stores_centroids():
    conn = _create_test_db_with_vec()

    np.random.seed(42)
    for i in range(10):
        vec = np.random.randn(4).astype(np.float32).tolist()
        _insert_chunk(conn, f"chunk_{i}", vec)

    build_clusters(conn, algorithm="kmeans", k=3)

    centroids = conn.execute("SELECT * FROM lsm_cluster_centroids").fetchall()
    assert len(centroids) == 3
    for row in centroids:
        assert row["size"] > 0
        blob = bytes(row["centroid"])
        dim = len(blob) // 4
        assert dim == 4


def test_build_clusters_updates_vec_chunks():
    conn = _create_test_db_with_vec()

    np.random.seed(42)
    for i in range(10):
        vec = np.random.randn(4).astype(np.float32).tolist()
        _insert_chunk(conn, f"chunk_{i}", vec)

    build_clusters(conn, algorithm="kmeans", k=3)

    rows = conn.execute("SELECT cluster_id FROM vec_chunks").fetchall()
    for row in rows:
        assert row[0] is not None


def test_build_clusters_empty_db():
    conn = _create_test_db_with_vec()

    result = build_clusters(conn, algorithm="kmeans", k=3)

    assert result["n_clusters"] == 0
    assert result["n_chunks"] == 0


def test_build_clusters_k_exceeds_chunks():
    conn = _create_test_db_with_vec()

    # Only 3 chunks but k=50
    for i in range(3):
        vec = [float(i)] * 4
        _insert_chunk(conn, f"chunk_{i}", vec)

    result = build_clusters(conn, algorithm="kmeans", k=50)

    # k is capped to n_chunks
    assert result["n_clusters"] == 3
    assert result["n_chunks"] == 3


# ------------------------------------------------------------------
# Tests: get_top_clusters
# ------------------------------------------------------------------

def test_get_top_clusters_returns_closest():
    conn = _create_test_db()

    # Create 3 centroids: one aligned with each axis
    c0 = [1.0, 0.0, 0.0, 0.0]
    c1 = [0.0, 1.0, 0.0, 0.0]
    c2 = [0.0, 0.0, 1.0, 0.0]

    for cid, centroid in enumerate([c0, c1, c2]):
        blob = pack(f"{len(centroid)}f", *centroid)
        conn.execute(
            "INSERT INTO lsm_cluster_centroids (cluster_id, centroid, size) VALUES (?, ?, ?)",
            (cid, blob, 10),
        )
    conn.commit()

    # Query aligned with c0
    query = [1.0, 0.0, 0.0, 0.0]
    top = get_top_clusters(query, conn, top_n=1)
    assert top == [0]

    # Query aligned with c1
    query = [0.0, 1.0, 0.0, 0.0]
    top = get_top_clusters(query, conn, top_n=1)
    assert top == [1]


def test_get_top_clusters_top_n():
    conn = _create_test_db()

    for cid in range(5):
        centroid = [0.0] * 4
        centroid[cid % 4] = 1.0
        blob = pack("4f", *centroid)
        conn.execute(
            "INSERT INTO lsm_cluster_centroids (cluster_id, centroid, size) VALUES (?, ?, ?)",
            (cid, blob, 5),
        )
    conn.commit()

    query = [1.0, 0.5, 0.0, 0.0]
    top = get_top_clusters(query, conn, top_n=2)
    assert len(top) == 2


def test_get_top_clusters_empty_centroids():
    conn = _create_test_db()

    result = get_top_clusters([1.0, 0.0, 0.0], conn, top_n=3)
    assert result == []


# ------------------------------------------------------------------
# Tests: config fields
# ------------------------------------------------------------------

def test_query_config_cluster_fields():
    from lsm.config.models.query import QueryConfig

    config = QueryConfig(
        cluster_enabled=True,
        cluster_algorithm="kmeans",
        cluster_k=100,
        cluster_top_n=10,
    )
    assert config.cluster_enabled is True
    assert config.cluster_algorithm == "kmeans"
    assert config.cluster_k == 100
    assert config.cluster_top_n == 10


def test_query_config_cluster_defaults():
    from lsm.config.models.query import QueryConfig

    config = QueryConfig()
    assert config.cluster_enabled is False
    assert config.cluster_algorithm == "kmeans"
    assert config.cluster_k == 50
    assert config.cluster_top_n == 5


def test_query_config_cluster_validation():
    from lsm.config.models.query import QueryConfig

    config = QueryConfig(cluster_algorithm="invalid")
    with pytest.raises(ValueError, match="cluster_algorithm"):
        config.validate()

    config = QueryConfig(cluster_k=1)
    with pytest.raises(ValueError, match="cluster_k"):
        config.validate()

    config = QueryConfig(cluster_top_n=0)
    with pytest.raises(ValueError, match="cluster_top_n"):
        config.validate()


# ------------------------------------------------------------------
# Tests: end-to-end build + query
# ------------------------------------------------------------------

def test_build_then_query_clusters():
    """Build clusters, then query top clusters — end-to-end."""
    conn = _create_test_db_with_vec()

    # Create two distinct groups of vectors
    np.random.seed(42)
    for i in range(5):
        vec = np.array([1.0, 0.0, 0.0, 0.0]) + np.random.randn(4) * 0.1
        _insert_chunk(conn, f"group_a_{i}", vec.tolist())
    for i in range(5):
        vec = np.array([0.0, 1.0, 0.0, 0.0]) + np.random.randn(4) * 0.1
        _insert_chunk(conn, f"group_b_{i}", vec.tolist())

    build_clusters(conn, algorithm="kmeans", k=2)

    # Query near group A
    top = get_top_clusters([1.0, 0.0, 0.0, 0.0], conn, top_n=1)
    assert len(top) == 1

    # All group_a chunks should be in the returned cluster
    cluster_id = top[0]
    group_a_rows = conn.execute(
        "SELECT cluster_id FROM lsm_chunks WHERE chunk_id LIKE 'group_a_%'"
    ).fetchall()
    for row in group_a_rows:
        assert row[0] == cluster_id
