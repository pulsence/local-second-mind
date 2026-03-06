"""Tests for the post-migration chunk enrichment pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lsm.db import enrichment
from lsm.db import migration as migration_mod
from lsm.db.enrichment import (
    ALL_STAGE_NAMES,
    STAGE_ALIASES,
    resolve_stage_names,
)
from lsm.db.schema import ensure_application_schema
from lsm.db.tables import DEFAULT_TABLE_NAMES
from lsm.ingest.dedup_hash import compute_simhash


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with full application schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_application_schema(conn)
    conn.commit()
    return conn


def _insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str = "c1",
    source_path: str = "/tmp/test.md",
    chunk_text: str = "hello world",
    *,
    heading: str | None = None,
    heading_path: str | None = None,
    version: int | None = 1,
    is_current: int = 1,
    node_type: str | None = "chunk",
    simhash: int | None = None,
    root_tags: str | None = None,
    folder_tags: str | None = None,
    content_type: str | None = None,
    cluster_id: int | None = None,
    start_char: int | None = None,
    end_char: int | None = None,
    chunk_length: int | None = None,
    chunk_index: int | None = 0,
) -> None:
    tn = DEFAULT_TABLE_NAMES
    conn.execute(
        f"""INSERT INTO {tn.chunks}
            (chunk_id, source_path, chunk_text, heading, heading_path,
             version, is_current, node_type, simhash, root_tags,
             folder_tags, content_type, cluster_id, start_char,
             end_char, chunk_length, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            source_path,
            chunk_text,
            heading,
            heading_path,
            version,
            is_current,
            node_type,
            simhash,
            root_tags,
            folder_tags,
            content_type,
            cluster_id,
            start_char,
            end_char,
            chunk_length,
            chunk_index,
        ),
    )
    conn.commit()


def _fake_config(**kwargs):
    """Build a minimal config-like object."""
    ingest = kwargs.pop("ingest", SimpleNamespace(roots=[]))
    query = kwargs.pop("query", SimpleNamespace(cluster_enabled=False, retrieval_profile="hybrid_rrf"))
    return SimpleNamespace(ingest=ingest, query=query, **kwargs)


# ==================================================================
# detect_stale_chunks
# ==================================================================


class TestDetectStaleChunks:
    def test_empty_database(self):
        conn = _make_conn()
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["simhash_null_count"] == 0
        assert result["tier2"]["heading_path_null_count"] == 0
        assert result["tier3"]["needs_reingest"] is False

    def test_counts_null_simhash(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", simhash=None)
        _insert_chunk(conn, "c2", simhash=12345)
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["simhash_null_count"] == 1

    def test_counts_null_version(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", version=None)
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["version_null_count"] == 1

    def test_counts_null_node_type(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", node_type=None)
        _insert_chunk(conn, "c2", node_type="")
        _insert_chunk(conn, "c3", node_type="chunk")
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["node_type_null_count"] == 2

    def test_counts_missing_tags(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", root_tags=None, content_type=None)
        _insert_chunk(conn, "c2", root_tags='["tag"]', content_type="notes")
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["tags_missing_count"] == 1

    def test_counts_null_heading_path(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", heading="Introduction", heading_path=None)
        _insert_chunk(conn, "c2", heading="Chapter 2", heading_path='["Chapter 2"]')
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier2"]["heading_path_null_count"] == 1

    def test_counts_null_positions(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", start_char=None)
        _insert_chunk(conn, "c2", start_char=0, end_char=100, chunk_length=100)
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier2"]["positions_null_count"] == 1

    def test_source_paths_for_tier2(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", source_path="/a.md", heading="H", heading_path=None)
        _insert_chunk(conn, "c2", source_path="/b.md", start_char=None)
        result = enrichment.detect_stale_chunks(conn)
        assert set(result["tier2"]["source_paths"]) == {"/a.md", "/b.md"}

    def test_cluster_rebuild_needed(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", cluster_id=None)
        result = enrichment.detect_stale_chunks(conn, cluster_enabled=True)
        assert result["tier2"]["cluster_rebuild_needed"] is True

    def test_cluster_not_needed_when_disabled(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", cluster_id=None)
        result = enrichment.detect_stale_chunks(conn, cluster_enabled=False)
        assert result["tier2"]["cluster_rebuild_needed"] is False

    def test_fully_enriched_returns_zeros(self):
        conn = _make_conn()
        _insert_chunk(
            conn,
            "c1",
            simhash=123,
            version=1,
            node_type="chunk",
            root_tags='["t"]',
            content_type="notes",
            heading="H",
            heading_path='["H"]',
            start_char=0,
            end_char=11,
            chunk_length=11,
            cluster_id=0,
        )
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["simhash_null_count"] == 0
        assert result["tier1"]["version_null_count"] == 0
        assert result["tier1"]["node_type_null_count"] == 0
        assert result["tier1"]["tags_missing_count"] == 0
        assert result["tier2"]["heading_path_null_count"] == 0
        assert result["tier2"]["positions_null_count"] == 0

    def test_missing_summaries_flagged(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", source_path="/f.md", node_type="chunk")
        # No section_summary or file_summary rows
        result = enrichment.detect_stale_chunks(conn, summaries_enabled=True)
        assert result["tier3"]["needs_reingest"] is True
        assert "/f.md" in result["tier3"]["missing_section_summary_files"]
        assert "/f.md" in result["tier3"]["missing_file_summary_files"]

    def test_drifted_source_paths_detected(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", source_path="/a.md", start_char=-1, end_char=-1, chunk_length=10)
        _insert_chunk(conn, "c2", source_path="/a.md", start_char=-1, end_char=-1, chunk_length=20)
        _insert_chunk(conn, "c3", source_path="/b.md", start_char=-1, end_char=-1, chunk_length=15)
        _insert_chunk(conn, "c4", source_path="/c.md", start_char=0, end_char=10, chunk_length=10)
        result = enrichment.detect_stale_chunks(conn)
        assert set(result["tier3"]["drifted_source_paths"]) == {"/a.md", "/b.md"}

    def test_needs_reingest_true_when_drifted_paths_exist(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", source_path="/a.md", start_char=-1, end_char=-1, chunk_length=10)
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier3"]["needs_reingest"] is True

    def test_no_table_returns_empty(self):
        conn = sqlite3.connect(":memory:")
        result = enrichment.detect_stale_chunks(conn)
        assert result["tier1"]["simhash_null_count"] == 0


# ==================================================================
# Tier 1
# ==================================================================


class TestTier1Enrichment:
    def test_simhash_backfill(self):
        conn = _make_conn()
        text = "some text for hashing"
        _insert_chunk(conn, "c1", chunk_text=text, simhash=None)
        cfg = _fake_config()

        updated = enrichment.run_tier1_enrichment(conn, cfg)
        assert updated >= 1

        row = conn.execute(
            f"SELECT simhash FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert row[0] == compute_simhash(text)

    def test_version_defaults(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", version=None, is_current=None)
        cfg = _fake_config()

        enrichment.run_tier1_enrichment(conn, cfg)

        row = conn.execute(
            f"SELECT version, is_current FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 1

    def test_node_type_defaults(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", node_type=None)
        _insert_chunk(conn, "c2", node_type="")
        cfg = _fake_config()

        enrichment.run_tier1_enrichment(conn, cfg)

        for cid in ("c1", "c2"):
            row = conn.execute(
                f"SELECT node_type FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = ?",
                (cid,),
            ).fetchone()
            assert row[0] == "chunk"

    def test_tag_enrichment(self, tmp_path):
        conn = _make_conn()
        # Create a test file in a root directory
        root = tmp_path / "notes"
        root.mkdir()
        test_file = root / "test.md"
        test_file.write_text("content", encoding="utf-8")

        root_cfg = SimpleNamespace(
            path=root,
            tags=["research"],
            content_type="notes",
        )
        _insert_chunk(
            conn,
            "c1",
            source_path=str(test_file),
            root_tags=None,
            content_type=None,
        )

        cfg = _fake_config(ingest=SimpleNamespace(roots=[root_cfg]))
        enrichment.run_tier1_enrichment(conn, cfg)

        row = conn.execute(
            f"SELECT root_tags, content_type FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert json.loads(row[0]) == ["research"]
        assert row[1] == "notes"

    def test_idempotent(self):
        conn = _make_conn()
        text = "idempotent test text"
        _insert_chunk(conn, "c1", chunk_text=text, simhash=None, version=None, node_type=None)
        cfg = _fake_config()

        result1 = enrichment.run_tier1_enrichment(conn, cfg)
        result2 = enrichment.run_tier1_enrichment(conn, cfg)

        assert result1 > 0
        assert result2 == 0  # Nothing left to update


# ==================================================================
# Tier 2
# ==================================================================


class TestTier2Enrichment:
    def test_backfill_positions_returns_structured_counters(self, tmp_path):
        conn = _make_conn()
        test_file = tmp_path / "positions.md"
        test_file.write_text("hello world", encoding="utf-8")

        _insert_chunk(
            conn,
            "c1",
            source_path=str(test_file),
            chunk_text="hello world",
            start_char=None,
            end_char=None,
            chunk_length=None,
        )

        counters = enrichment._backfill_positions(
            conn,
            DEFAULT_TABLE_NAMES,
            str(test_file),
            test_file,
        )

        assert counters.matched_updates == 1
        assert counters.sentinel_updates == 0
        assert counters.files_with_writes == 1
        assert counters.chunks_examined == 1

    def test_position_backfill(self, tmp_path):
        conn = _make_conn()
        test_file = tmp_path / "test.md"
        content = "hello world this is a test document"
        test_file.write_text(content, encoding="utf-8")

        _insert_chunk(
            conn,
            "c1",
            source_path=str(test_file),
            chunk_text=content,
            start_char=None,
            end_char=None,
            chunk_length=None,
        )
        cfg = _fake_config()

        updated, skipped = enrichment.run_tier2_enrichment(conn, cfg)
        assert updated >= 1
        assert len(skipped) == 0

        row = conn.execute(
            f"SELECT start_char, end_char, chunk_length FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert row[0] == 0  # starts at beginning
        assert row[1] == len(content)
        assert row[2] == len(content)

    def test_missing_source_files_skipped(self):
        conn = _make_conn()
        _insert_chunk(
            conn,
            "c1",
            source_path="/nonexistent/file.md",
            heading="H",
            heading_path=None,
        )
        cfg = _fake_config()

        updated, skipped = enrichment.run_tier2_enrichment(conn, cfg)
        assert updated == 0
        assert "/nonexistent/file.md" in skipped

    def test_idempotent(self, tmp_path):
        conn = _make_conn()
        test_file = tmp_path / "test.md"
        content = "idempotent tier 2 test"
        test_file.write_text(content, encoding="utf-8")

        _insert_chunk(
            conn,
            "c1",
            source_path=str(test_file),
            chunk_text=content,
            start_char=None,
        )
        cfg = _fake_config()

        u1, _ = enrichment.run_tier2_enrichment(conn, cfg)
        u2, _ = enrichment.run_tier2_enrichment(conn, cfg)
        assert u1 >= 1
        assert u2 == 0  # Already enriched

    def test_boundary_drift_sets_sentinel(self, tmp_path):
        """Chunks that can't be matched get start_char=-1 so future runs skip them."""
        conn = _make_conn()
        test_file = tmp_path / "test.md"
        test_file.write_text("completely different content", encoding="utf-8")

        # Chunk text does NOT appear in the file (boundary drift)
        _insert_chunk(
            conn,
            "c1",
            source_path=str(test_file),
            chunk_text="this text is not in the file at all",
            start_char=None,
            end_char=None,
            chunk_length=None,
        )
        cfg = _fake_config()

        u1, _ = enrichment.run_tier2_enrichment(conn, cfg)
        # u1 may include graph backfill count; the key check is the sentinel below

        row = conn.execute(
            f"SELECT start_char, end_char FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert row[0] == -1  # sentinel: attempted but unmatchable
        assert row[1] == -1

        # Second run should find zero files to process (sentinel excluded)
        u2, _ = enrichment.run_tier2_enrichment(conn, cfg)
        assert u2 == 0

    def test_position_backfill_logs_matched_and_deferred_progress(self, tmp_path, monkeypatch, caplog):
        conn = _make_conn()
        caplog.set_level("INFO")
        file_a = tmp_path / "a.md"
        file_b = tmp_path / "b.md"
        file_a.write_text("matched content", encoding="utf-8")
        file_b.write_text("different", encoding="utf-8")

        _insert_chunk(conn, "c1", source_path=str(file_a), chunk_text="matched content", start_char=None)
        _insert_chunk(conn, "c2", source_path=str(file_b), chunk_text="missing chunk text", start_char=None)

        monkeypatch.setattr(enrichment, "_POSITION_FLUSH_FILE_THRESHOLD", 1)
        monkeypatch.setattr(enrichment, "_POSITION_FLUSH_ROW_THRESHOLD", 1)
        monkeypatch.setattr(enrichment, "_POSITION_CHECKPOINT_EVERY_BATCHES", 99)

        report = enrichment.run_enrichment_pipeline(conn, _fake_config(), only_stages={"enrich_tier2_positions"})

        assert report.tier2_updated == 2
        assert any(
            "Position backfill:" in record.message
            and "matched" in record.message
            and "deferred" in record.message
            and "written" in record.message
            for record in caplog.records
        )


# ==================================================================
# _build_heading_map_from_graph  /  _backfill_graph  (unit tests)
# ==================================================================


class TestBuildHeadingMapFromGraph:
    """Direct tests for _build_heading_map_from_graph using real GraphNode objects."""

    @staticmethod
    def _gn(id: str, node_type: str, name: str, parent_id=None):
        """Convenience builder for a GraphNode."""
        from lsm.utils.file_graph import GraphNode

        return GraphNode(
            id=id,
            node_type=node_type,
            name=name,
            start_line=1,
            end_line=1,
            start_char=0,
            end_char=1,
            depth=0,
            parent_id=parent_id,
            children=(),
            metadata={},
            line_hash="abc",
        )

    def test_flat_headings(self):
        from lsm.utils.file_graph import FileGraph

        fg = FileGraph(
            path="test.md",
            content_hash="aaa",
            nodes=(
                self._gn("h1", "heading", "Introduction"),
                self._gn("h2", "heading", "Conclusion"),
            ),
            root_ids=("h1", "h2"),
        )
        heading_map = enrichment._build_heading_map_from_graph(fg)
        assert heading_map["Introduction"] == ["Introduction"]
        assert heading_map["Conclusion"] == ["Conclusion"]

    def test_nested_headings_build_path(self):
        from lsm.utils.file_graph import FileGraph

        fg = FileGraph(
            path="test.md",
            content_hash="aaa",
            nodes=(
                self._gn("h1", "heading", "Chapter 1"),
                self._gn("h2", "heading", "Section A", parent_id="h1"),
                self._gn("h3", "heading", "Subsection i", parent_id="h2"),
            ),
            root_ids=("h1",),
        )
        heading_map = enrichment._build_heading_map_from_graph(fg)
        assert heading_map["Chapter 1"] == ["Chapter 1"]
        assert heading_map["Section A"] == ["Chapter 1", "Section A"]
        assert heading_map["Subsection i"] == ["Chapter 1", "Section A", "Subsection i"]

    def test_non_heading_parents_skipped_in_path(self):
        from lsm.utils.file_graph import FileGraph

        fg = FileGraph(
            path="test.md",
            content_hash="aaa",
            nodes=(
                self._gn("root", "document", "doc"),
                self._gn("h1", "heading", "Title", parent_id="root"),
            ),
            root_ids=("root",),
        )
        heading_map = enrichment._build_heading_map_from_graph(fg)
        # "document" parent should not appear in the heading path
        assert heading_map["Title"] == ["Title"]

    def test_empty_graph_returns_empty(self):
        heading_map = enrichment._build_heading_map_from_graph(SimpleNamespace(nodes=[]))
        assert heading_map == {}

    def test_no_nodes_attr_returns_empty(self):
        heading_map = enrichment._build_heading_map_from_graph(object())
        assert heading_map == {}


class TestBackfillGraph:
    """Direct tests for _backfill_graph inserting nodes/edges from source files."""

    def test_inserts_graph_nodes_and_edges(self, tmp_path):
        conn = _make_conn()
        tn = DEFAULT_TABLE_NAMES

        # Create a markdown file with heading structure
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Heading\n\nSome content here.\n", encoding="utf-8")

        # Insert a chunk referencing this file, with no graph nodes yet
        _insert_chunk(
            conn,
            "c1",
            source_path=str(test_file),
            chunk_text="Some content here.",
        )
        conn.commit()

        updated = enrichment._backfill_graph(conn, tn)
        assert updated == 1

        # Verify graph nodes were created
        node_rows = conn.execute(
            f"SELECT node_id, node_type, label, source_path FROM {tn.graph_nodes}"
        ).fetchall()
        assert len(node_rows) >= 1
        # All nodes should reference this source file
        for row in node_rows:
            assert row["source_path"] == str(test_file)

    def test_skips_files_already_in_graph(self, tmp_path):
        conn = _make_conn()
        tn = DEFAULT_TABLE_NAMES

        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello\n\nWorld.\n", encoding="utf-8")

        _insert_chunk(conn, "c1", source_path=str(test_file), chunk_text="World.")

        # Pre-insert a graph node for this source — should cause skip
        conn.execute(
            f"INSERT INTO {tn.graph_nodes} (node_id, node_type, label, source_path) "
            f"VALUES (?, ?, ?, ?)",
            ("existing", "heading", "Hello", str(test_file)),
        )
        conn.commit()

        updated = enrichment._backfill_graph(conn, tn)
        assert updated == 0

    def test_skips_nonexistent_files(self):
        conn = _make_conn()
        tn = DEFAULT_TABLE_NAMES

        _insert_chunk(conn, "c1", source_path="/no/such/file.md", chunk_text="test")
        conn.commit()

        updated = enrichment._backfill_graph(conn, tn)
        assert updated == 0

    def test_no_graph_tables_returns_zero(self):
        """If graph_nodes table doesn't exist, should return 0 gracefully."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create just the chunks table, no graph tables
        conn.execute(
            """CREATE TABLE lsm_chunks (
                chunk_id TEXT PRIMARY KEY,
                chunk_text TEXT,
                source_path TEXT,
                node_type TEXT DEFAULT 'chunk',
                is_current INTEGER DEFAULT 1
            )"""
        )
        conn.execute(
            "INSERT INTO lsm_chunks (chunk_id, chunk_text, source_path) VALUES (?, ?, ?)",
            ("c1", "test", "/tmp/test.md"),
        )
        conn.commit()

        updated = enrichment._backfill_graph(conn, DEFAULT_TABLE_NAMES)
        assert updated == 0

    def test_skips_binary_extensions(self, tmp_path):
        conn = _make_conn()
        tn = DEFAULT_TABLE_NAMES

        # Create a PDF file (binary) — should be skipped
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake binary content")

        _insert_chunk(conn, "c1", source_path=str(pdf_file), chunk_text="parsed text")
        conn.commit()

        updated = enrichment._backfill_graph(conn, tn)
        assert updated == 0

    def test_idempotent(self, tmp_path):
        conn = _make_conn()
        tn = DEFAULT_TABLE_NAMES

        test_file = tmp_path / "doc.md"
        test_file.write_text("# Title\n\nParagraph.\n", encoding="utf-8")

        _insert_chunk(conn, "c1", source_path=str(test_file), chunk_text="Paragraph.")
        conn.commit()

        u1 = enrichment._backfill_graph(conn, tn)
        assert u1 == 1

        u2 = enrichment._backfill_graph(conn, tn)
        assert u2 == 0  # Nodes already exist, nothing to do

    def test_backfill_graph_does_not_create_source_path_index(self):
        conn = _make_conn()
        tn = DEFAULT_TABLE_NAMES

        conn.execute(f"DROP INDEX IF EXISTS idx_{tn.graph_nodes}_source_path")
        conn.commit()

        enrichment._backfill_graph(conn, tn)

        idx = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?",
            (f"idx_{tn.graph_nodes}_source_path",),
        ).fetchone()[0]
        assert idx == 0

    def test_sqlite_checkpoint_helper_is_noop_for_non_sqlite(self):
        assert enrichment._maybe_checkpoint_position_batch(object(), batches_committed=1) is None


# ==================================================================
# Tier 2b (cluster rebuild)
# ==================================================================


class TestTier2bClusterEnrichment:
    def test_skips_when_cluster_disabled(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", cluster_id=None)
        cfg = _fake_config(query=SimpleNamespace(cluster_enabled=False))

        result = enrichment.run_tier2_cluster_enrichment(conn, cfg)
        assert result == 0

    def test_skips_when_no_null_clusters(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", cluster_id=0)
        cfg = _fake_config(query=SimpleNamespace(cluster_enabled=True, cluster_algorithm="kmeans", cluster_k=5))

        result = enrichment.run_tier2_cluster_enrichment(conn, cfg)
        assert result == 0

    def test_postgresql_uses_provider_api_without_sqlite_vec_table(self, monkeypatch):
        conn = _make_conn()
        _insert_chunk(conn, "c1", cluster_id=None)
        _insert_chunk(conn, "c2", cluster_id=None)

        updates_seen = []

        class _FakeProvider:
            name = "postgresql"

            def __init__(self, connection):
                self._connection = connection

            def get_embeddings(self, filters=None, only_current=True):
                _ = filters, only_current
                return ["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]]

            def update_cluster_assignments(self, updates):
                updates_seen.extend(updates)
                for chunk_id, cluster_id in updates:
                    self._connection.execute(
                        f"UPDATE {DEFAULT_TABLE_NAMES.chunks} SET cluster_id = ? WHERE chunk_id = ?",
                        (int(cluster_id), str(chunk_id)),
                    )
                self._connection.commit()

            def _get_conn(self):
                class _Ctx:
                    def __enter__(inner_self):
                        return self._connection

                    def __exit__(inner_self, exc_type, exc, tb):
                        return False

                return _Ctx()

        monkeypatch.setattr(
            "lsm.vectordb.create_vectordb_provider",
            lambda _db_cfg: _FakeProvider(conn),
        )

        cfg = _fake_config(
            query=SimpleNamespace(cluster_enabled=True, cluster_algorithm="kmeans", cluster_k=2),
            db=SimpleNamespace(provider="postgresql"),
        )

        result = enrichment.run_tier2_cluster_enrichment(conn, cfg)

        assert result == 2
        assert len(updates_seen) == 2
        cluster_ids = {
            row[0] for row in conn.execute(
                f"SELECT cluster_id FROM {DEFAULT_TABLE_NAMES.chunks} ORDER BY chunk_id"
            ).fetchall()
        }
        assert cluster_ids == {0, 1}


# ==================================================================
# Full pipeline
# ==================================================================


class TestEnrichmentPipeline:
    def test_full_pipeline_report(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", simhash=None, version=None, node_type=None)
        cfg = _fake_config()

        report = enrichment.run_enrichment_pipeline(conn, cfg)
        assert isinstance(report, enrichment.EnrichmentReport)
        assert report.tier1_updated > 0
        assert len(report.errors) == 0

    def test_skip_tier2(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", simhash=None, start_char=None, source_path="/missing.md")
        cfg = _fake_config()

        report = enrichment.run_enrichment_pipeline(conn, cfg, skip_tier2=True)
        assert report.tier1_updated > 0
        assert report.tier2_updated == 0

    def test_stage_tracker_called(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", simhash=None)
        cfg = _fake_config()

        stages: list[tuple[str, str]] = []
        def tracker(name, status):
            stages.append((name, status))

        enrichment.run_enrichment_pipeline(conn, cfg, stage_tracker=tracker)
        assert ("enrich_tier1", "in_progress") in stages
        assert ("enrich_tier1", "completed") in stages

    def test_stage_tracker_completes_after_commit(self, monkeypatch):
        conn = _make_conn()
        _insert_chunk(conn, "c1", simhash=None)
        cfg = _fake_config()
        events: list[str] = []
        original_commit = enrichment.commit

        def tracked_commit(connection):
            events.append("commit")
            return original_commit(connection)

        def tracker(name, status):
            if name == "enrich_tier1_simhash" and status == "completed":
                events.append("completed")

        monkeypatch.setattr(enrichment, "commit", tracked_commit)

        enrichment.run_enrichment_pipeline(conn, cfg, stage_tracker=tracker, only_stages={"enrich_tier1_simhash"})

        assert "commit" in events
        assert "completed" in events
        assert events.index("commit") < events.index("completed")

    def test_migration_stage_tracker_records_completion_after_commit(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "enrich.db"))
        conn.row_factory = sqlite3.Row
        ensure_application_schema(conn)
        migration_mod._ensure_migration_progress_table(conn, DEFAULT_TABLE_NAMES)
        _insert_chunk(conn, "c1", simhash=None)
        cfg = _fake_config()
        tracker = migration_mod._build_enrichment_stage_tracker(
            conn=conn,
            run_id="run-1",
            source_type="sqlite",
            target_type="sqlite",
            tn=DEFAULT_TABLE_NAMES,
        )

        enrichment.run_enrichment_pipeline(
            conn,
            cfg,
            stage_tracker=tracker,
            only_stages={"enrich_tier1_simhash"},
        )

        row = conn.execute(
            f"SELECT status FROM {DEFAULT_TABLE_NAMES.migration_progress} "
            f"WHERE stage = 'enrich_tier1_simhash' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == "completed"

    def test_tier3_advisory_in_report(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", node_type="chunk", source_path="/f.md")
        cfg = _fake_config(
            query=SimpleNamespace(cluster_enabled=False, retrieval_profile="multi_vector")
        )

        report = enrichment.run_enrichment_pipeline(conn, cfg)
        # Should report missing summaries as tier3 needed
        assert len(report.tier3_needed) > 0

    def test_drifted_source_paths_in_report(self):
        conn = _make_conn()
        _insert_chunk(
            conn, "c1", source_path="/drifted.md",
            start_char=-1, end_char=-1, chunk_length=10,
        )
        _insert_chunk(
            conn, "c2", source_path="/ok.md",
            start_char=0, end_char=10, chunk_length=10,
        )
        cfg = _fake_config()

        report = enrichment.run_enrichment_pipeline(conn, cfg)
        assert "/drifted.md" in report.drifted_source_paths
        assert "/ok.md" not in report.drifted_source_paths
        assert any("boundary drifted" in t for t in report.tier3_needed)


# ==================================================================
# CLI Integration Tests
# ==================================================================


class TestCLIEnrichmentFlags:
    """Test that --enrich and --skip-enrich flags are parsed correctly."""

    def test_enrich_flag_parses(self):
        from lsm.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["migrate", "--enrich"])
        assert args.enrich is True
        assert args.skip_enrich is False

    def test_skip_enrich_flag_parses(self):
        from lsm.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "migrate", "--skip-enrich", "--from-db", "sqlite", "--to-db", "sqlite"
        ])
        assert args.skip_enrich is True
        assert args.enrich is False

    def test_mutual_exclusion_in_cli(self):
        from lsm.ui.shell.cli import run_migrate_cli

        # run_migrate_cli should reject both --enrich and --skip-enrich
        code = run_migrate_cli(
            "/nonexistent/config.json",
            enrich=True,
            skip_enrich=True,
        )
        assert code == 2

    def test_enrich_rejects_from_db(self):
        from lsm.ui.shell.cli import run_migrate_cli

        code = run_migrate_cli(
            "/nonexistent/config.json",
            enrich=True,
            from_db="sqlite",
        )
        assert code == 2

    def test_rechunk_flag_parses(self):
        from lsm.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["migrate", "--rechunk"])
        assert args.rechunk is True
        assert args.skip_rechunk is False

    def test_skip_rechunk_flag_parses(self):
        from lsm.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["migrate", "--skip-rechunk"])
        assert args.skip_rechunk is True
        assert args.rechunk is False

    def test_rechunk_mutual_exclusion_in_cli(self):
        from lsm.ui.shell.cli import run_migrate_cli

        code = run_migrate_cli(
            "/nonexistent/config.json",
            rechunk=True,
            skip_rechunk=True,
        )
        assert code == 2

    def test_handle_rechunk_offer_no_drifted(self, capsys):
        from lsm.ui.shell.cli import _handle_rechunk_offer
        from lsm.db.enrichment import EnrichmentReport

        report = EnrichmentReport(drifted_source_paths=())
        _handle_rechunk_offer(None, report, rechunk=True)
        captured = capsys.readouterr()
        # Should produce no output when no drifted paths
        assert "rechunk" not in captured.out.lower()

    def test_handle_rechunk_offer_skip_flag(self, capsys):
        from lsm.ui.shell.cli import _handle_rechunk_offer
        from lsm.db.enrichment import EnrichmentReport

        report = EnrichmentReport(drifted_source_paths=("/nonexistent/a.md",))
        _handle_rechunk_offer(None, report, skip_rechunk=True)
        captured = capsys.readouterr()
        assert "skipping rechunk" in captured.out.lower()

    def test_handle_rechunk_offer_nonexistent_paths(self, capsys):
        from lsm.ui.shell.cli import _handle_rechunk_offer
        from lsm.db.enrichment import EnrichmentReport

        report = EnrichmentReport(
            drifted_source_paths=("/nonexistent/a.md", "/nonexistent/b.md"),
        )
        _handle_rechunk_offer(None, report, rechunk=True)
        captured = capsys.readouterr()
        assert "none exist on disk" in captured.out.lower()

    def test_stage_flag_parses(self):
        from lsm.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["migrate", "--enrich", "--stage", "tier1", "--stage", "graph"])
        assert args.stages == ["tier1", "graph"]

    def test_stage_without_enrich_returns_error(self):
        from lsm.ui.shell.cli import run_migrate_cli

        code = run_migrate_cli(
            "/nonexistent/config.json",
            stages=["tier1"],
        )
        assert code == 2

    def test_invalid_stage_name_returns_error(self):
        from lsm.ui.shell.cli import run_migrate_cli

        code = run_migrate_cli(
            "/nonexistent/config.json",
            enrich=True,
            stages=["bogus"],
        )
        assert code == 2


# ==================================================================
# Stage name mapping
# ==================================================================


class TestStageNameMapping:
    def test_resolve_tier1(self):
        result = resolve_stage_names(["tier1"])
        assert result == STAGE_ALIASES["tier1"]

    def test_resolve_individual_graph(self):
        result = resolve_stage_names(["graph"])
        assert result == {"enrich_tier2_graph"}

    def test_resolve_multiple_combined(self):
        result = resolve_stage_names(["tier1", "graph"])
        expected = STAGE_ALIASES["tier1"] | {"enrich_tier2_graph"}
        assert result == expected

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown stage name 'bogus'"):
            resolve_stage_names(["bogus"])

    def test_case_insensitive(self):
        result = resolve_stage_names(["Tier1", "GRAPH"])
        expected = STAGE_ALIASES["tier1"] | {"enrich_tier2_graph"}
        assert result == expected

    def test_all_stage_names_covers_all_aliases(self):
        combined = set()
        for stages in STAGE_ALIASES.values():
            combined |= stages
        assert combined == ALL_STAGE_NAMES


# ==================================================================
# only_stages filtering
# ==================================================================


class TestOnlyStages:
    def test_only_simhash_runs_just_simhash(self):
        conn = _make_conn()
        _insert_chunk(conn, "c1", simhash=None, version=None, node_type=None)
        cfg = _fake_config()

        report = enrichment.run_enrichment_pipeline(
            conn, cfg, only_stages={"enrich_tier1_simhash"},
        )
        assert report.tier1_updated > 0

        # version and node_type should remain NULL (those stages were skipped)
        row = conn.execute(
            f"SELECT version, node_type FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert row[0] is None  # version not backfilled
        assert row[1] is None  # node_type not backfilled

    def test_only_tier1_skips_tier2(self):
        conn = _make_conn()
        _insert_chunk(
            conn, "c1", simhash=None, start_char=None, source_path="/missing.md",
        )
        cfg = _fake_config()

        report = enrichment.run_enrichment_pipeline(
            conn, cfg, only_stages=STAGE_ALIASES["tier1"],
        )
        assert report.tier1_updated > 0
        assert report.tier2_updated == 0

    def test_only_stages_and_skip_stages_raises(self):
        conn = _make_conn()
        cfg = _fake_config()

        with pytest.raises(ValueError, match="mutually exclusive"):
            enrichment.run_enrichment_pipeline(
                conn, cfg,
                only_stages={"enrich_tier1_simhash"},
                skip_stages={"enrich_tier1_tags"},
            )
