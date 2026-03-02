"""Tests for lsm.db.health — database health check module."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lsm.db.health import (
    DBHealthReport,
    check_db_health,
    _check_db_reachable,
    _check_legacy_provider,
    _check_schema_version,
    _check_required_tables,
    _check_partial_migration,
    _check_stale_chunks,
)
from lsm.db.schema import ensure_application_schema
from lsm.db.tables import DEFAULT_TABLE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, *, provider: str = "sqlite") -> SimpleNamespace:
    db_path = tmp_path / "data"
    db_path.mkdir(exist_ok=True)
    return SimpleNamespace(
        db=SimpleNamespace(
            provider=provider,
            path=db_path,
            connection_string=None,
            host=None,
            port=None,
            database=None,
            user=None,
            password=None,
        ),
        global_settings=SimpleNamespace(global_folder=tmp_path / "global"),
    )


def _create_db(tmp_path: Path) -> Path:
    """Create a valid lsm.db with full application schema."""
    db_path = tmp_path / "data" / "lsm.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    ensure_application_schema(conn)
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Test: clean database → "ok"
# ---------------------------------------------------------------------------

class TestHealthyDatabase:
    def test_clean_database_returns_ok(self, tmp_path: Path) -> None:
        _create_db(tmp_path)
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert report.status == "ok"

    def test_ok_report_is_not_blocking(self, tmp_path: Path) -> None:
        _create_db(tmp_path)
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert report.blocking is False


# ---------------------------------------------------------------------------
# Test: no database file → "missing"
# ---------------------------------------------------------------------------

class TestMissingDatabase:
    def test_no_db_file_returns_missing(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert report.status == "missing"
        assert "not found" in report.details

    def test_missing_is_not_blocking(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert report.blocking is False

    def test_missing_suggests_ingest(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert "ingest" in report.suggested_action.lower()


# ---------------------------------------------------------------------------
# Test: PostgreSQL unreachable → "corrupt" with connection guidance
# ---------------------------------------------------------------------------

class TestPostgresUnreachable:
    def test_pg_unreachable_returns_corrupt(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, provider="postgresql")
        config.db.connection_string = "postgresql://bad:bad@localhost:1/nonexistent"
        report = _check_db_reachable(config)
        assert report is not None
        assert report.status == "corrupt"
        assert report.blocking is True

    def test_pg_missing_psycopg2_returns_corrupt(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, provider="postgresql")
        config.db.connection_string = "postgresql://test@localhost/db"
        with patch.dict("sys.modules", {"psycopg2": None}):
            report = _check_db_reachable(config)
        assert report is not None
        assert report.status == "corrupt"
        assert "psycopg2" in report.details


# ---------------------------------------------------------------------------
# Test: .chroma/ directory present → "legacy_detected"
# ---------------------------------------------------------------------------

class TestLegacyDetection:
    def test_chromadb_provider_returns_legacy_detected(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        config.db.provider = "chromadb"
        report = _check_legacy_provider(config)
        assert report is not None
        assert report.status == "legacy_detected"
        assert report.blocking is True
        assert "chromadb" in report.details.lower()

    def test_chroma_provider_returns_legacy_detected(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        config.db.provider = "chroma"
        report = _check_legacy_provider(config)
        assert report is not None
        assert report.status == "legacy_detected"

    def test_sqlite_provider_returns_none(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        report = _check_legacy_provider(config)
        assert report is None

    def test_chroma_dir_on_disk_does_not_block(self, tmp_path: Path) -> None:
        """Leftover .chroma/ directory should NOT block when config uses sqlite."""
        config = _make_config(tmp_path)
        chroma_dir = tmp_path / "global" / ".chroma"
        chroma_dir.mkdir(parents=True)
        report = _check_legacy_provider(config)
        assert report is None

    def test_no_db_config_returns_none(self) -> None:
        config = SimpleNamespace()
        report = _check_legacy_provider(config)
        assert report is None


# ---------------------------------------------------------------------------
# Test: schema version mismatch → "mismatch" with correct field diff
# ---------------------------------------------------------------------------

class TestSchemaVersionMismatch:
    def test_mismatch_returns_correct_diff(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Record a schema version
        from lsm.db.schema_version import record_schema_version
        record_schema_version(conn, {
            "lsm_version": "0.6.0",
            "embedding_model": "test-model",
            "embedding_dim": 384,
            "chunking_strategy": "structure",
            "chunk_size": 800,
            "chunk_overlap": 120,
        })

        # Check with different config
        new_config = {
            "lsm_version": "0.7.0",
            "embedding_model": "test-model",
            "embedding_dim": 384,
            "chunking_strategy": "structure",
            "chunk_size": 1200,
            "chunk_overlap": 120,
        }

        report = _check_schema_version(conn, new_config, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is not None
        assert report.status == "mismatch"
        assert report.blocking is True
        assert "lsm_version" in report.schema_diff
        assert "chunk_size" in report.schema_diff

    def test_compatible_returns_none(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        from lsm.db.schema_version import record_schema_version
        schema = {
            "lsm_version": "0.7.0",
            "embedding_model": "test-model",
            "embedding_dim": 384,
            "chunking_strategy": "structure",
            "chunk_size": 800,
            "chunk_overlap": 120,
        }
        record_schema_version(conn, schema)
        report = _check_schema_version(conn, schema, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is None


# ---------------------------------------------------------------------------
# Test: missing application tables → "corrupt"
# ---------------------------------------------------------------------------

class TestMissingTables:
    def test_missing_tables_returns_corrupt(self, tmp_path: Path) -> None:
        db_path = tmp_path / "data" / "lsm.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create DB with only one table (not full schema)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.close()

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        report = _check_required_tables(conn, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is not None
        assert report.status == "corrupt"
        assert report.blocking is True
        assert "lsm_chunks" in report.details

    def test_all_tables_present_returns_none(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        report = _check_required_tables(conn, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is None


# ---------------------------------------------------------------------------
# Test: partial migration in progress → "partial_migration"
# ---------------------------------------------------------------------------

class TestPartialMigration:
    def test_no_migration_table_returns_none(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        report = _check_partial_migration(conn, DEFAULT_TABLE_NAMES)
        conn.close()
        assert report is None

    def test_stuck_migration_returns_partial(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Create migration_progress table with a stuck row
        conn.execute(f"""
            CREATE TABLE {DEFAULT_TABLE_NAMES.migration_progress} (
                id INTEGER PRIMARY KEY,
                task_name TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        conn.execute(
            f"INSERT INTO {DEFAULT_TABLE_NAMES.migration_progress} "
            f"(task_name, status, started_at) VALUES (?, ?, ?)",
            ("migrate_chunks", "in_progress", "2025-01-01T00:00:00"),
        )
        conn.commit()

        report = _check_partial_migration(conn, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is not None
        assert report.status == "partial_migration"
        assert report.blocking is True
        assert "resume" in report.suggested_action.lower()

    def test_completed_migration_returns_none(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        conn.execute(f"""
            CREATE TABLE {DEFAULT_TABLE_NAMES.migration_progress} (
                id INTEGER PRIMARY KEY,
                task_name TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        conn.execute(
            f"INSERT INTO {DEFAULT_TABLE_NAMES.migration_progress} "
            f"(task_name, status, started_at, completed_at) VALUES (?, ?, ?, ?)",
            ("migrate_chunks", "completed", "2025-01-01T00:00:00", "2025-01-01T00:01:00"),
        )
        conn.commit()

        report = _check_partial_migration(conn, DEFAULT_TABLE_NAMES)
        conn.close()
        assert report is None


# ---------------------------------------------------------------------------
# Test: stale chunks → "stale_chunks" (non-blocking)
# ---------------------------------------------------------------------------

class TestStaleChunks:
    def test_chunks_with_null_simhash(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Insert a chunk with NULL simhash
        conn.execute(
            f"INSERT INTO {DEFAULT_TABLE_NAMES.chunks} "
            f"(chunk_id, chunk_text, source_path, simhash) VALUES (?, ?, ?, ?)",
            ("c1", "hello world", "/docs/a.md", None),
        )
        conn.commit()

        report = _check_stale_chunks(conn, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is not None
        assert report.status == "stale_chunks"
        assert report.blocking is False
        assert "simhash" in report.details.lower()

    def test_chunks_with_null_node_type(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        conn.execute(
            f"INSERT INTO {DEFAULT_TABLE_NAMES.chunks} "
            f"(chunk_id, chunk_text, source_path, simhash, node_type) VALUES (?, ?, ?, ?, ?)",
            ("c1", "hello world", "/docs/a.md", 12345, None),
        )
        conn.commit()

        report = _check_stale_chunks(conn, DEFAULT_TABLE_NAMES)
        conn.close()

        assert report is not None
        assert report.status == "stale_chunks"
        assert "node_type" in report.details.lower()

    def test_fully_enriched_chunks_returns_none(self, tmp_path: Path) -> None:
        db_path = _create_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        conn.execute(
            f"INSERT INTO {DEFAULT_TABLE_NAMES.chunks} "
            f"(chunk_id, chunk_text, source_path, simhash, node_type) VALUES (?, ?, ?, ?, ?)",
            ("c1", "hello world", "/docs/a.md", 12345, "section"),
        )
        conn.commit()

        report = _check_stale_chunks(conn, DEFAULT_TABLE_NAMES)
        conn.close()
        assert report is None

    def test_no_chunks_table_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        report = _check_stale_chunks(conn, DEFAULT_TABLE_NAMES)
        conn.close()
        assert report is None


# ---------------------------------------------------------------------------
# Test: integration — full check_db_health() flow
# ---------------------------------------------------------------------------

class TestCheckDbHealthIntegration:
    def test_full_healthy_flow(self, tmp_path: Path) -> None:
        _create_db(tmp_path)
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert report.status == "ok"

    def test_missing_db_is_non_blocking(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        report = check_db_health(config)
        assert report.status == "missing"
        assert report.blocking is False

    def test_legacy_chromadb_config_blocks(self, tmp_path: Path) -> None:
        _create_db(tmp_path)
        config = _make_config(tmp_path)
        config.db.provider = "chromadb"
        report = check_db_health(config)
        assert report.status == "legacy_detected"
        assert report.blocking is True

    def test_chroma_dir_on_disk_does_not_block(self, tmp_path: Path) -> None:
        _create_db(tmp_path)
        config = _make_config(tmp_path)
        chroma_dir = tmp_path / "global" / ".chroma"
        chroma_dir.mkdir(parents=True)
        report = check_db_health(config)
        assert report.status == "ok"

    def test_no_config_db_returns_ok(self) -> None:
        config = SimpleNamespace(global_settings=SimpleNamespace(global_folder=None))
        report = check_db_health(config)
        assert report.status == "ok"


# ---------------------------------------------------------------------------
# Test: DBHealthReport dataclass
# ---------------------------------------------------------------------------

class TestDBHealthReport:
    def test_frozen(self) -> None:
        report = DBHealthReport(status="ok")
        with pytest.raises(AttributeError):
            report.status = "bad"

    def test_defaults(self) -> None:
        report = DBHealthReport(status="ok")
        assert report.details == ""
        assert report.suggested_action == ""
        assert report.schema_diff == {}
        assert report.blocking is False
