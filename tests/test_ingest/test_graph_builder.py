"""Tests for graph builder (Phase 15.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from lsm.ingest.graph_builder import (
    DBGraphEdge,
    DBGraphNode,
    _extract_links,
    _stable_id,
    build_graph_from_file_graph,
)


# ------------------------------------------------------------------
# Fake FileGraph for testing
# ------------------------------------------------------------------


@dataclass(frozen=True)
class FakeGraphNode:
    id: str
    node_type: str
    name: str
    start_line: int = 1
    end_line: int = 10
    start_char: int = 0
    end_char: int = 100
    depth: int = 0
    parent_id: Optional[str] = None
    children: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = None
    line_hash: str = "abc123"

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class FakeFileGraph:
    path: str
    nodes: Tuple[FakeGraphNode, ...]
    root_ids: Tuple[str, ...] = ()

    def node_map(self):
        return {n.id: n for n in self.nodes}


# ------------------------------------------------------------------
# Tests: _stable_id
# ------------------------------------------------------------------


class TestStableId:
    def test_deterministic(self):
        a = _stable_id("/path/file.md", "part1")
        b = _stable_id("/path/file.md", "part1")
        assert a == b

    def test_different_inputs(self):
        a = _stable_id("/path/a.md", "x")
        b = _stable_id("/path/b.md", "x")
        assert a != b

    def test_length(self):
        result = _stable_id("/path/file.md")
        assert len(result) == 16


# ------------------------------------------------------------------
# Tests: _extract_links
# ------------------------------------------------------------------


class TestExtractLinks:
    def test_wikilinks(self):
        text = "See [[Other Note]] and [[Another|display text]] for details."
        nodes, edges = _extract_links(text, "/source.md", "file_node_1")
        link_nodes = [n for n in nodes if n.node_type == "link_target"]
        assert len(link_nodes) == 2
        labels = {n.label for n in link_nodes}
        assert "Other Note" in labels
        assert "Another" in labels

        ref_edges = [e for e in edges if e.edge_type == "references"]
        assert len(ref_edges) == 2
        for e in ref_edges:
            assert e.src_id == "file_node_1"

    def test_markdown_internal_links(self):
        text = "Read [the guide](./guide.md) and [intro](../intro.md)."
        nodes, edges = _extract_links(text, "/source.md", "file_node_1")
        link_nodes = [n for n in nodes if n.node_type == "link_target"]
        assert len(link_nodes) == 2

    def test_markdown_external_links_ignored(self):
        text = "Visit [Google](https://google.com) and [mail](mailto:a@b.com)."
        nodes, edges = _extract_links(text, "/source.md", "file_node_1")
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_doi_references(self):
        text = "Based on doi:10.1234/test.2023 and https://doi.org/10.5678/paper.1"
        nodes, edges = _extract_links(text, "/source.md", "file_node_1")
        doi_nodes = [n for n in nodes if n.node_type == "doi"]
        assert len(doi_nodes) == 2
        cites_edges = [e for e in edges if e.edge_type == "cites"]
        assert len(cites_edges) == 2

    def test_deduplicates_links(self):
        text = "See [[Target]] and [[Target]] again."
        nodes, edges = _extract_links(text, "/source.md", "file_node_1")
        assert len(nodes) == 1
        assert len(edges) == 1

    def test_empty_text(self):
        nodes, edges = _extract_links("", "/source.md", "file_node_1")
        assert nodes == []
        assert edges == []

    def test_anchor_links_ignored(self):
        text = "Jump to [section](#heading)."
        nodes, edges = _extract_links(text, "/source.md", "file_node_1")
        assert len(nodes) == 0


# ------------------------------------------------------------------
# Tests: build_graph_from_file_graph
# ------------------------------------------------------------------


class TestBuildGraphFromFileGraph:
    def test_file_node_always_created(self):
        nodes, edges = build_graph_from_file_graph(None, "/test.md", "some text")
        file_nodes = [n for n in nodes if n.node_type == "file"]
        assert len(file_nodes) == 1
        assert file_nodes[0].source_path == "/test.md"
        assert file_nodes[0].label == "test.md"

    def test_heading_hierarchy(self):
        heading1 = FakeGraphNode(id="h1", node_type="heading", name="Introduction", depth=0)
        heading2 = FakeGraphNode(
            id="h2", node_type="heading", name="Details",
            depth=1, parent_id="h1",
        )
        fg = FakeFileGraph(
            path="/test.md",
            nodes=(heading1, heading2),
            root_ids=("h1",),
        )
        nodes, edges = build_graph_from_file_graph(fg, "/test.md", "text")

        # File node + 2 heading nodes
        assert len(nodes) == 3

        contains_edges = [e for e in edges if e.edge_type == "contains"]
        assert len(contains_edges) == 2  # file→h1 and h1→h2

    def test_root_heading_connects_to_file(self):
        heading = FakeGraphNode(id="h1", node_type="heading", name="Title", depth=0)
        fg = FakeFileGraph(path="/test.md", nodes=(heading,))
        nodes, edges = build_graph_from_file_graph(fg, "/test.md", "text")

        file_node = next(n for n in nodes if n.node_type == "file")
        heading_node = next(n for n in nodes if n.node_type == "heading")

        root_edges = [
            e for e in edges
            if e.src_id == file_node.node_id and e.dst_id == heading_node.node_id
        ]
        assert len(root_edges) == 1
        assert root_edges[0].edge_type == "contains"

    def test_child_heading_connects_to_parent(self):
        parent = FakeGraphNode(id="h1", node_type="heading", name="Parent")
        child = FakeGraphNode(
            id="h2", node_type="heading", name="Child",
            depth=1, parent_id="h1",
        )
        fg = FakeFileGraph(path="/test.md", nodes=(parent, child))
        nodes, edges = build_graph_from_file_graph(fg, "/test.md", "text")

        parent_db_id = _stable_id("/test.md", "heading", "h1")
        child_db_id = _stable_id("/test.md", "heading", "h2")

        parent_child_edges = [
            e for e in edges
            if e.src_id == parent_db_id and e.dst_id == child_db_id
        ]
        assert len(parent_child_edges) == 1

    def test_links_extracted_from_text(self):
        text = "See [[Other Note]] and doi:10.1234/paper"
        nodes, edges = build_graph_from_file_graph(None, "/test.md", text)

        link_nodes = [n for n in nodes if n.node_type == "link_target"]
        doi_nodes = [n for n in nodes if n.node_type == "doi"]
        assert len(link_nodes) == 1
        assert len(doi_nodes) == 1

    def test_no_file_graph(self):
        nodes, edges = build_graph_from_file_graph(None, "/plain.txt", "Plain text.")
        assert len(nodes) == 1  # Only file node
        assert len(edges) == 0

    def test_heading_path_built(self):
        parent = FakeGraphNode(id="h1", node_type="heading", name="Part 1")
        child = FakeGraphNode(
            id="h2", node_type="heading", name="Section A",
            depth=1, parent_id="h1",
        )
        fg = FakeFileGraph(path="/test.md", nodes=(parent, child))
        nodes, edges = build_graph_from_file_graph(fg, "/test.md", "text")

        child_node = [n for n in nodes if n.label == "Section A"][0]
        assert child_node.heading_path == "Part 1 / Section A"


# ------------------------------------------------------------------
# Tests: graph provider methods (SQLiteVecProvider)
# ------------------------------------------------------------------


class TestGraphProviderMethods:
    def _create_test_db(self):
        """Create an in-memory SQLite DB with graph tables."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE lsm_graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT,
                label TEXT,
                source_path TEXT,
                heading_path TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE lsm_graph_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_id TEXT REFERENCES lsm_graph_nodes(node_id),
                dst_id TEXT REFERENCES lsm_graph_nodes(node_id),
                edge_type TEXT,
                weight REAL DEFAULT 1.0
            )
        """)
        conn.execute("CREATE INDEX idx_edges_src ON lsm_graph_edges(src_id)")
        conn.execute("CREATE INDEX idx_edges_dst ON lsm_graph_edges(dst_id)")
        conn.commit()
        return conn

    def _make_fake_provider(self, conn):
        """Create a minimal fake provider with graph methods."""
        from types import SimpleNamespace
        provider = SimpleNamespace(_conn=conn)

        # Bind the methods from SQLiteVecProvider
        from lsm.vectordb.sqlite_vec import SQLiteVecProvider
        provider.graph_insert_nodes = lambda nodes: SQLiteVecProvider.graph_insert_nodes(provider, nodes)
        provider.graph_insert_edges = lambda edges: SQLiteVecProvider.graph_insert_edges(provider, edges)
        provider.graph_traverse = lambda start_ids, max_hops=2, edge_types=None: SQLiteVecProvider.graph_traverse(provider, start_ids, max_hops, edge_types)
        return provider

    def test_insert_and_query_nodes(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)

        nodes = [
            {"node_id": "n1", "node_type": "file", "label": "test.md", "source_path": "/test.md"},
            {"node_id": "n2", "node_type": "heading", "label": "Intro", "source_path": "/test.md"},
        ]
        provider.graph_insert_nodes(nodes)

        rows = conn.execute("SELECT COUNT(*) FROM lsm_graph_nodes").fetchone()
        assert rows[0] == 2

    def test_insert_and_query_edges(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)

        provider.graph_insert_nodes([
            {"node_id": "n1", "node_type": "file", "label": "test.md", "source_path": "/test.md"},
            {"node_id": "n2", "node_type": "heading", "label": "Intro", "source_path": "/test.md"},
        ])
        provider.graph_insert_edges([
            {"src_id": "n1", "dst_id": "n2", "edge_type": "contains"},
        ])

        rows = conn.execute("SELECT COUNT(*) FROM lsm_graph_edges").fetchone()
        assert rows[0] == 1

    def test_traverse_single_hop(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)

        provider.graph_insert_nodes([
            {"node_id": "a", "node_type": "file", "label": "a", "source_path": "/a.md"},
            {"node_id": "b", "node_type": "heading", "label": "b", "source_path": "/a.md"},
            {"node_id": "c", "node_type": "heading", "label": "c", "source_path": "/a.md"},
        ])
        provider.graph_insert_edges([
            {"src_id": "a", "dst_id": "b", "edge_type": "contains"},
            {"src_id": "b", "dst_id": "c", "edge_type": "contains"},
        ])

        # Start at "a", 1 hop should reach "b" but not "c"
        reachable = provider.graph_traverse(["a"], max_hops=1)
        assert "a" in reachable
        assert "b" in reachable
        assert "c" not in reachable

    def test_traverse_multi_hop(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)

        provider.graph_insert_nodes([
            {"node_id": "a", "node_type": "file", "label": "a", "source_path": "/a.md"},
            {"node_id": "b", "node_type": "heading", "label": "b", "source_path": "/a.md"},
            {"node_id": "c", "node_type": "heading", "label": "c", "source_path": "/a.md"},
        ])
        provider.graph_insert_edges([
            {"src_id": "a", "dst_id": "b", "edge_type": "contains"},
            {"src_id": "b", "dst_id": "c", "edge_type": "contains"},
        ])

        # 2 hops should reach all
        reachable = provider.graph_traverse(["a"], max_hops=2)
        assert set(reachable) == {"a", "b", "c"}

    def test_traverse_edge_type_filter(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)

        provider.graph_insert_nodes([
            {"node_id": "a", "node_type": "file", "label": "a", "source_path": "/a.md"},
            {"node_id": "b", "node_type": "heading", "label": "b", "source_path": "/a.md"},
            {"node_id": "c", "node_type": "link_target", "label": "c", "source_path": "/c.md"},
        ])
        provider.graph_insert_edges([
            {"src_id": "a", "dst_id": "b", "edge_type": "contains"},
            {"src_id": "a", "dst_id": "c", "edge_type": "references"},
        ])

        # Only follow "contains" edges
        reachable = provider.graph_traverse(["a"], max_hops=2, edge_types=["contains"])
        assert "b" in reachable
        assert "c" not in reachable

    def test_traverse_empty_start(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)
        assert provider.graph_traverse([]) == []

    def test_upsert_nodes(self):
        conn = self._create_test_db()
        provider = self._make_fake_provider(conn)

        nodes = [{"node_id": "n1", "node_type": "file", "label": "old", "source_path": "/a.md"}]
        provider.graph_insert_nodes(nodes)

        # Upsert with new label
        nodes[0]["label"] = "new"
        provider.graph_insert_nodes(nodes)

        row = conn.execute("SELECT label FROM lsm_graph_nodes WHERE node_id = 'n1'").fetchone()
        assert row[0] == "new"
        # Should still be 1 row
        count = conn.execute("SELECT COUNT(*) FROM lsm_graph_nodes").fetchone()[0]
        assert count == 1
