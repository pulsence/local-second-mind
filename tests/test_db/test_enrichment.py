"""Tests for the post-migration chunk enrichment pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lsm.db import enrichment
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
        assert u1 == 0  # No positions matched

        row = conn.execute(
            f"SELECT start_char, end_char FROM {DEFAULT_TABLE_NAMES.chunks} WHERE chunk_id = 'c1'"
        ).fetchone()
        assert row[0] == -1  # sentinel: attempted but unmatchable
        assert row[1] == -1

        # Second run should find zero files to process (sentinel excluded)
        u2, _ = enrichment.run_tier2_enrichment(conn, cfg)
        assert u2 == 0


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
