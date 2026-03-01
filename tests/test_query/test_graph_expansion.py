"""Tests for graph-augmented retrieval (Phase 15.2)."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate
from lsm.query.stages.graph_expansion import expand_via_graph


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _create_test_db() -> sqlite3.Connection:
    """Create in-memory DB with graph tables and chunk tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE lsm_graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT,
            label TEXT,
            source_path TEXT,
            heading_path TEXT
        );
        CREATE TABLE lsm_graph_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_id TEXT REFERENCES lsm_graph_nodes(node_id),
            dst_id TEXT REFERENCES lsm_graph_nodes(node_id),
            edge_type TEXT,
            weight REAL DEFAULT 1.0
        );
        CREATE INDEX idx_edges_src ON lsm_graph_edges(src_id);
        CREATE INDEX idx_edges_dst ON lsm_graph_edges(dst_id);

        CREATE TABLE lsm_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_path TEXT,
            source_name TEXT,
            chunk_text TEXT,
            is_current INTEGER DEFAULT 1,
            node_type TEXT DEFAULT 'chunk',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
    """)
    return conn


class FakeGraphDB:
    """Fake DB provider with graph support via direct SQLite connection."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @property
    def connection(self):
        return self._conn

    def graph_traverse(self, start_ids, max_hops=2, edge_types=None):
        from lsm.vectordb.sqlite_vec import SQLiteVecProvider
        return SQLiteVecProvider.graph_traverse(self, start_ids, max_hops, edge_types)

    def get(self, ids=None, filters=None, limit=None, offset=0, include=None):
        """Retrieve chunks from lsm_chunks."""
        clauses = ["1=1"]
        params = []

        if filters:
            if "source_path" in filters:
                clauses.append("source_path = ?")
                params.append(filters["source_path"])
            if "is_current" in filters:
                clauses.append("is_current = ?")
                params.append(1 if filters["is_current"] else 0)
            if "node_type" in filters:
                clauses.append("node_type = ?")
                params.append(filters["node_type"])

        where = " AND ".join(clauses)
        sql = f"SELECT chunk_id, chunk_text, source_path, node_type FROM lsm_chunks WHERE {where}"
        if limit:
            sql += f" LIMIT {limit}"

        rows = self._conn.execute(sql, params).fetchall()

        result_ids = [r[0] for r in rows]
        result_docs = [r[1] or "" for r in rows] if include and "documents" in include else None
        result_metas = [
            {"source_path": r[2], "node_type": r[3]} for r in rows
        ] if include and "metadatas" in include else None

        return SimpleNamespace(ids=result_ids, documents=result_docs, metadatas=result_metas)


def _insert_graph_node(conn, node_id, node_type, label, source_path):
    conn.execute(
        "INSERT OR REPLACE INTO lsm_graph_nodes VALUES (?, ?, ?, ?, ?)",
        (node_id, node_type, label, source_path, None),
    )


def _insert_graph_edge(conn, src_id, dst_id, edge_type, weight=1.0):
    conn.execute(
        "INSERT INTO lsm_graph_edges (src_id, dst_id, edge_type, weight) VALUES (?, ?, ?, ?)",
        (src_id, dst_id, edge_type, weight),
    )


def _insert_chunk(conn, chunk_id, source_path, text="chunk text"):
    conn.execute(
        "INSERT INTO lsm_chunks (chunk_id, source_path, chunk_text, is_current, node_type) "
        "VALUES (?, ?, ?, 1, 'chunk')",
        (chunk_id, source_path, text),
    )


def _make_candidate(cid, source_path):
    return Candidate(
        cid=cid,
        text=f"Text of {cid}",
        meta={"source_path": source_path},
        distance=0.1,
        score_breakdown=ScoreBreakdown(dense_score=0.9),
    )


def _stable_id(source_path, *parts):
    """Mirror the graph_builder's stable ID generation."""
    import hashlib
    raw = "|".join([source_path] + list(parts))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestExpandViaGraph:
    def test_no_expansion_when_disabled(self):
        """Empty candidates returns empty."""
        result = expand_via_graph([], None)
        assert result == []

    def test_returns_original_when_no_graph(self):
        """When graph traversal finds nothing, returns originals."""
        conn = _create_test_db()
        conn.commit()
        db = FakeGraphDB(conn)

        candidates = [_make_candidate("c1", "/a.md")]
        result = expand_via_graph(candidates, db, max_hops=2)
        assert len(result) >= 1
        assert result[0].cid == "c1"

    def test_expands_via_graph_link(self):
        """Graph traversal finds related file, retrieves its chunks."""
        conn = _create_test_db()

        # Source file a.md has candidate c1
        file_a_id = _stable_id("/a.md", "file")
        file_b_id = _stable_id("/b.md", "file")

        _insert_graph_node(conn, file_a_id, "file", "a.md", "/a.md")
        _insert_graph_node(conn, file_b_id, "file", "b.md", "/b.md")
        _insert_graph_edge(conn, file_a_id, file_b_id, "references")

        # b.md has chunks that aren't in the initial results
        _insert_chunk(conn, "b_chunk_1", "/b.md", "content from b")
        conn.commit()

        db = FakeGraphDB(conn)
        candidates = [_make_candidate("c1", "/a.md")]

        result = expand_via_graph(candidates, db, max_hops=2)

        # Should have original + expanded chunk from b.md
        cids = [c.cid for c in result]
        assert "c1" in cids
        assert "b_chunk_1" in cids

    def test_deduplicates_expanded_candidates(self):
        """Don't add chunks that are already in the original candidates."""
        conn = _create_test_db()

        file_a_id = _stable_id("/a.md", "file")
        file_b_id = _stable_id("/b.md", "file")

        _insert_graph_node(conn, file_a_id, "file", "a.md", "/a.md")
        _insert_graph_node(conn, file_b_id, "file", "b.md", "/b.md")
        _insert_graph_edge(conn, file_a_id, file_b_id, "references")

        _insert_chunk(conn, "existing_chunk", "/b.md", "already in results")
        conn.commit()

        db = FakeGraphDB(conn)
        candidates = [
            _make_candidate("c1", "/a.md"),
            _make_candidate("existing_chunk", "/b.md"),
        ]

        result = expand_via_graph(candidates, db, max_hops=2)

        cid_counts = {}
        for c in result:
            cid_counts[c.cid] = cid_counts.get(c.cid, 0) + 1
        # No duplicates
        assert all(count == 1 for count in cid_counts.values())

    def test_expansion_score_present(self):
        """Expanded candidates have graph_expansion_score set."""
        conn = _create_test_db()

        file_a_id = _stable_id("/a.md", "file")
        file_b_id = _stable_id("/b.md", "file")

        _insert_graph_node(conn, file_a_id, "file", "a.md", "/a.md")
        _insert_graph_node(conn, file_b_id, "file", "b.md", "/b.md")
        _insert_graph_edge(conn, file_a_id, file_b_id, "references")

        _insert_chunk(conn, "b1", "/b.md")
        conn.commit()

        db = FakeGraphDB(conn)
        candidates = [_make_candidate("c1", "/a.md")]

        result = expand_via_graph(candidates, db, max_hops=2)

        expanded = [c for c in result if c.cid != "c1"]
        assert len(expanded) > 0
        for c in expanded:
            assert c.score_breakdown.graph_expansion_score is not None
            assert c.score_breakdown.graph_expansion_score > 0

    def test_respects_max_hops(self):
        """Nodes beyond max_hops are not reached."""
        conn = _create_test_db()

        file_a_id = _stable_id("/a.md", "file")
        file_b_id = _stable_id("/b.md", "file")
        file_c_id = _stable_id("/c.md", "file")

        _insert_graph_node(conn, file_a_id, "file", "a.md", "/a.md")
        _insert_graph_node(conn, file_b_id, "file", "b.md", "/b.md")
        _insert_graph_node(conn, file_c_id, "file", "c.md", "/c.md")

        # a→b→c chain
        _insert_graph_edge(conn, file_a_id, file_b_id, "references")
        _insert_graph_edge(conn, file_b_id, file_c_id, "references")

        _insert_chunk(conn, "b1", "/b.md")
        _insert_chunk(conn, "c1", "/c.md")
        conn.commit()

        db = FakeGraphDB(conn)
        candidates = [_make_candidate("a1", "/a.md")]

        # 1 hop should reach b but not c
        result = expand_via_graph(candidates, db, max_hops=1)
        cids = [c.cid for c in result]
        assert "b1" in cids
        assert "c1" not in cids

    def test_no_crash_on_missing_graph_tables(self):
        """Gracefully handles provider without graph support."""
        db = SimpleNamespace(
            graph_traverse=lambda *a, **kw: [],
            _conn=None,
            connection=None,
        )
        candidates = [_make_candidate("c1", "/a.md")]
        result = expand_via_graph(candidates, db, max_hops=2)
        # Should return originals without error
        assert len(result) == 1


class TestGraphExpansionConfig:
    def test_config_fields(self):
        from lsm.config.models.query import QueryConfig

        config = QueryConfig(
            graph_expansion_enabled=True,
            graph_expansion_hops=3,
        )
        assert config.graph_expansion_enabled is True
        assert config.graph_expansion_hops == 3

    def test_config_defaults(self):
        from lsm.config.models.query import QueryConfig

        config = QueryConfig()
        assert config.graph_expansion_enabled is False
        assert config.graph_expansion_hops == 2

    def test_config_validation(self):
        from lsm.config.models.query import QueryConfig

        config = QueryConfig(graph_expansion_hops=0)
        with pytest.raises(ValueError, match="graph_expansion_hops"):
            config.validate()
