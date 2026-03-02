"""Integration tests for SQLiteVecProvider transaction and savepoint behavior.

These tests exercise the real sqlite-vec extension to verify that:
- vec0 virtual table operations work correctly with transaction() savepoints
- Graph operations (delete/insert) work with transaction() context manager
- The pipeline flush pattern (add_chunks + manifest in separate transactions)
  doesn't cause commit/savepoint conflicts
- Interleaved graph and chunk writes on the same connection are safe
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lsm.config.models import DBConfig
from lsm.db.transaction import transaction
from lsm.db.tables import DEFAULT_TABLE_NAMES
from lsm.ingest.manifest import _ensure_manifest_table, upsert_manifest_entries
from lsm.vectordb.sqlite_vec import SQLiteVecProvider


def _vector(x: float, y: float = 0.0) -> list[float]:
    values = [0.0] * 384
    values[0] = float(x)
    values[1] = float(y)
    return values


def _meta(source_path: str, **overrides) -> dict:
    base = {"source_path": source_path, "source_name": source_path.split("/")[-1], "ext": ".md", "is_current": True}
    base.update(overrides)
    return base


def _provider(tmp_path: Path) -> SQLiteVecProvider:
    cfg = DBConfig(provider="sqlite", path=tmp_path / "db", collection="test")
    return SQLiteVecProvider(cfg)


# ---------------------------------------------------------------------------
# vec0 savepoint behavior
# ---------------------------------------------------------------------------


def test_add_chunks_inside_savepoint_with_vec0(tmp_path: Path) -> None:
    """vec0 virtual table operations inside a savepoint must not destroy it.

    This is the exact pattern that caused 'no such savepoint: lsm_sp_N'
    when flush() wrapped add_chunks in an explicit BEGIN.  add_chunks
    internally uses ``with transaction(conn)`` which creates a savepoint
    when a transaction is already active.
    """
    provider = _provider(tmp_path)
    conn = provider.connection

    # Simulate the old flush() pattern: outer BEGIN → add_chunks (savepoint)
    conn.execute("BEGIN")
    try:
        # add_chunks will see in_transaction=True and create a savepoint
        provider.add_chunks(
            ids=["sp-1", "sp-2"],
            documents=["alpha", "beta"],
            metadatas=[_meta("/a.md"), _meta("/b.md")],
            embeddings=[_vector(1.0), _vector(2.0)],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    assert provider.count() == 2


def test_add_chunks_standalone_without_outer_transaction(tmp_path: Path) -> None:
    """add_chunks works correctly when no outer transaction is active."""
    provider = _provider(tmp_path)

    provider.add_chunks(
        ids=["a", "b"],
        documents=["alpha", "beta"],
        metadatas=[_meta("/a.md"), _meta("/b.md")],
        embeddings=[_vector(1.0), _vector(2.0)],
    )
    assert provider.count() == 2


def test_add_chunks_rollback_on_error_inside_savepoint(tmp_path: Path) -> None:
    """If add_chunks fails inside a savepoint, the savepoint rollback must work."""
    provider = _provider(tmp_path)
    conn = provider.connection

    # Seed one good chunk first
    provider.add_chunks(
        ids=["good"],
        documents=["good text"],
        metadatas=[_meta("/good.md")],
        embeddings=[_vector(0.5)],
    )

    conn.execute("BEGIN")
    try:
        # This should fail on the second item (missing source_path)
        with pytest.raises(ValueError, match="source_path"):
            provider.add_chunks(
                ids=["ok", "bad"],
                documents=["text", "text"],
                metadatas=[_meta("/ok.md"), {"source_name": "no-path"}],
                embeddings=[_vector(1.0), _vector(2.0)],
            )
    finally:
        conn.rollback()

    # Only the pre-existing chunk should remain
    assert provider.count() == 1


# ---------------------------------------------------------------------------
# Graph operations with transaction() context manager
# ---------------------------------------------------------------------------


def test_graph_insert_nodes_standalone(tmp_path: Path) -> None:
    """graph_insert_nodes commits via transaction() without errors."""
    provider = _provider(tmp_path)
    provider.graph_insert_nodes([
        {"node_id": "n1", "node_type": "heading", "label": "Chapter 1", "source_path": "/a.md"},
        {"node_id": "n2", "node_type": "heading", "label": "Chapter 2", "source_path": "/a.md"},
    ])

    count = provider.connection.execute(
        f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_nodes}"
    ).fetchone()[0]
    assert count == 2


def test_graph_insert_edges_standalone(tmp_path: Path) -> None:
    """graph_insert_edges commits via transaction() without errors."""
    provider = _provider(tmp_path)
    provider.graph_insert_nodes([
        {"node_id": "n1", "node_type": "heading", "label": "A", "source_path": "/a.md"},
        {"node_id": "n2", "node_type": "heading", "label": "B", "source_path": "/a.md"},
    ])
    provider.graph_insert_edges([
        {"src_id": "n1", "dst_id": "n2", "edge_type": "sibling", "weight": 1.0},
    ])

    count = provider.connection.execute(
        f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_edges}"
    ).fetchone()[0]
    assert count == 1


def test_graph_delete_source_removes_nodes_and_edges(tmp_path: Path) -> None:
    """graph_delete_source cleans up both nodes and edges."""
    provider = _provider(tmp_path)
    provider.graph_insert_nodes([
        {"node_id": "n1", "node_type": "heading", "label": "A", "source_path": "/a.md"},
        {"node_id": "n2", "node_type": "heading", "label": "B", "source_path": "/a.md"},
        {"node_id": "n3", "node_type": "heading", "label": "C", "source_path": "/other.md"},
    ])
    provider.graph_insert_edges([
        {"src_id": "n1", "dst_id": "n2", "edge_type": "sibling"},
        {"src_id": "n1", "dst_id": "n3", "edge_type": "cross_file"},
    ])

    provider.graph_delete_source("/a.md")

    nodes = provider.connection.execute(
        f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_nodes}"
    ).fetchone()[0]
    edges = provider.connection.execute(
        f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_edges}"
    ).fetchone()[0]
    # Only n3 remains; both edges (referencing n1) are gone
    assert nodes == 1
    assert edges == 0


def test_graph_operations_inside_outer_transaction(tmp_path: Path) -> None:
    """Graph operations use savepoints when inside an outer transaction."""
    provider = _provider(tmp_path)
    conn = provider.connection

    conn.execute("BEGIN")
    try:
        provider.graph_insert_nodes([
            {"node_id": "n1", "node_type": "heading", "label": "A", "source_path": "/a.md"},
        ])
        provider.graph_insert_edges([
            {"src_id": "n1", "dst_id": "n1", "edge_type": "self"},
        ])
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    nodes = conn.execute(f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_nodes}").fetchone()[0]
    assert nodes == 1


# ---------------------------------------------------------------------------
# Pipeline flush simulation (add_chunks + manifest in separate transactions)
# ---------------------------------------------------------------------------


def test_flush_pattern_chunks_then_manifest(tmp_path: Path) -> None:
    """Simulate the pipeline flush: add_chunks commits independently,
    then manifest is committed in a separate transaction."""
    provider = _provider(tmp_path)
    conn = provider.connection
    _ensure_manifest_table(conn)

    # Step 1: add_chunks manages its own transaction
    provider.add_chunks(
        ids=["c1", "c2"],
        documents=["hello", "world"],
        metadatas=[_meta("/a.md"), _meta("/b.md")],
        embeddings=[_vector(1.0), _vector(2.0)],
    )

    # Step 2: manifest in separate transaction
    upsert_manifest_entries(
        conn,
        {
            "/a.md": {"mtime_ns": 1000, "size": 100, "file_hash": "abc", "version": 1},
            "/b.md": {"mtime_ns": 2000, "size": 200, "file_hash": "def", "version": 1},
        },
        commit=True,
        skip_ensure_table=True,
    )

    assert provider.count() == 2
    manifest_count = conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0]
    assert manifest_count == 2


def test_flush_pattern_manifest_failure_leaves_chunks_intact(tmp_path: Path) -> None:
    """If manifest upsert fails after add_chunks, chunks remain committed."""
    provider = _provider(tmp_path)
    conn = provider.connection
    _ensure_manifest_table(conn)

    # add_chunks succeeds
    provider.add_chunks(
        ids=["c1"],
        documents=["hello"],
        metadatas=[_meta("/a.md")],
        embeddings=[_vector(1.0)],
    )

    # Simulate manifest failure (e.g., table dropped)
    try:
        conn.execute("DROP TABLE lsm_manifest")
        upsert_manifest_entries(conn, {"/a.md": {"mtime_ns": 1}}, commit=True)
    except Exception:
        pass  # expected failure

    # Chunks survived because they were in a separate transaction
    assert provider.count() == 1


# ---------------------------------------------------------------------------
# Interleaved graph + chunk operations (writer thread simulation)
# ---------------------------------------------------------------------------


def test_interleaved_graph_and_chunk_writes(tmp_path: Path) -> None:
    """Simulate the writer thread: update_metadatas → graph ops → flush,
    repeated for multiple files. No transaction conflicts."""
    provider = _provider(tmp_path)
    conn = provider.connection
    _ensure_manifest_table(conn)

    for i in range(3):
        source = f"/file_{i}.md"

        # Graph operations (standalone transactions)
        provider.graph_insert_nodes([
            {"node_id": f"n_{i}", "node_type": "heading", "label": f"Heading {i}", "source_path": source},
        ])

        # add_chunks (standalone transaction)
        provider.add_chunks(
            ids=[f"chunk_{i}"],
            documents=[f"content {i}"],
            metadatas=[_meta(source)],
            embeddings=[_vector(float(i + 1))],
        )

        # Manifest (standalone transaction)
        upsert_manifest_entries(
            conn,
            {source: {"mtime_ns": i * 1000, "size": 100, "file_hash": f"hash_{i}", "version": 1}},
            commit=True,
            skip_ensure_table=True,
        )

    assert provider.count() == 3
    nodes = conn.execute(f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_nodes}").fetchone()[0]
    assert nodes == 3
    manifest = conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0]
    assert manifest == 3


def test_graph_delete_then_reinsert_during_rechunk(tmp_path: Path) -> None:
    """Simulate rechunk: delete old graph data, insert new, then write chunks."""
    provider = _provider(tmp_path)
    conn = provider.connection

    # Initial state: file has graph data and chunks
    provider.graph_insert_nodes([
        {"node_id": "old_n1", "node_type": "heading", "label": "Old", "source_path": "/a.md"},
    ])
    provider.graph_insert_edges([
        {"src_id": "old_n1", "dst_id": "old_n1", "edge_type": "self"},
    ])
    provider.add_chunks(
        ids=["old_c1"],
        documents=["old content"],
        metadatas=[_meta("/a.md")],
        embeddings=[_vector(1.0)],
    )

    # Rechunk: delete old graph, insert new graph, write new chunks
    provider.graph_delete_source("/a.md")
    provider.graph_insert_nodes([
        {"node_id": "new_n1", "node_type": "heading", "label": "New", "source_path": "/a.md"},
        {"node_id": "new_n2", "node_type": "heading", "label": "New 2", "source_path": "/a.md"},
    ])
    provider.add_chunks(
        ids=["new_c1", "new_c2"],
        documents=["new content 1", "new content 2"],
        metadatas=[_meta("/a.md"), _meta("/a.md")],
        embeddings=[_vector(2.0), _vector(3.0)],
    )

    # Old graph gone, new graph present
    nodes = conn.execute(
        f"SELECT node_id FROM {DEFAULT_TABLE_NAMES.graph_nodes} ORDER BY node_id"
    ).fetchall()
    assert [r[0] for r in nodes] == ["new_n1", "new_n2"]

    # Old chunk + new chunks (upsert, old_c1 still exists)
    assert provider.count() == 3


def test_multiple_flushes_with_graph_operations_between(tmp_path: Path) -> None:
    """Multiple flush cycles with graph operations between them.

    Simulates processing many files where flush_threshold triggers
    multiple flush() calls within a single ingest run.
    """
    provider = _provider(tmp_path)
    conn = provider.connection
    _ensure_manifest_table(conn)

    # First batch
    for i in range(5):
        provider.graph_insert_nodes([
            {"node_id": f"n_a{i}", "node_type": "heading", "label": f"A{i}", "source_path": f"/batch1/{i}.md"},
        ])

    provider.add_chunks(
        ids=[f"c_a{i}" for i in range(5)],
        documents=[f"batch1 doc {i}" for i in range(5)],
        metadatas=[_meta(f"/batch1/{i}.md") for i in range(5)],
        embeddings=[_vector(float(i)) for i in range(5)],
    )
    upsert_manifest_entries(
        conn,
        {f"/batch1/{i}.md": {"mtime_ns": i, "size": i, "file_hash": f"h{i}", "version": 1} for i in range(5)},
        commit=True,
        skip_ensure_table=True,
    )

    # Second batch
    for i in range(5):
        provider.graph_insert_nodes([
            {"node_id": f"n_b{i}", "node_type": "heading", "label": f"B{i}", "source_path": f"/batch2/{i}.md"},
        ])

    provider.add_chunks(
        ids=[f"c_b{i}" for i in range(5)],
        documents=[f"batch2 doc {i}" for i in range(5)],
        metadatas=[_meta(f"/batch2/{i}.md") for i in range(5)],
        embeddings=[_vector(float(i + 5)) for i in range(5)],
    )
    upsert_manifest_entries(
        conn,
        {f"/batch2/{i}.md": {"mtime_ns": i, "size": i, "file_hash": f"h{i}", "version": 1} for i in range(5)},
        commit=True,
        skip_ensure_table=True,
    )

    assert provider.count() == 10
    nodes = conn.execute(f"SELECT COUNT(*) FROM {DEFAULT_TABLE_NAMES.graph_nodes}").fetchone()[0]
    assert nodes == 10
    manifest = conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0]
    assert manifest == 10
