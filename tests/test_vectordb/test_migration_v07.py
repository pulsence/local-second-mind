"""Tests for legacy v0.7 -> v0.8 migration path."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lsm.db import migration as migration_mod


class _LegacyTargetProvider:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM lsm_chunks").fetchone()
        return int(row[0] or 0) if row is not None else 0


def _target_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migration_mod._ensure_aux_tables(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lsm_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_path TEXT,
            chunk_text TEXT,
            is_current INTEGER
        )
        """
    )
    conn.commit()
    return conn


def _write_memories_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            type TEXT,
            key TEXT,
            value_json TEXT,
            scope TEXT,
            tags_json TEXT,
            confidence REAL,
            created_at TEXT,
            last_used_at TEXT,
            expires_at TEXT,
            run_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE memory_candidates (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            provenance TEXT,
            rationale TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO memories (
            id, type, key, value_json, scope, tags_json,
            confidence, created_at, last_used_at, expires_at, run_id
        ) VALUES (
            'm-1', 'project_fact', 'fact_key', '{"answer": 42}', 'project', '["x","y"]',
            0.9, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', NULL, 'run-legacy'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO memory_candidates (
            id, memory_id, provenance, rationale, status, created_at, updated_at
        ) VALUES (
            'c-1', 'm-1', 'legacy', 'kept', 'pending',
            '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
        )
        """
    )
    conn.commit()
    conn.close()


def _runtime_config() -> dict[str, object]:
    return {
        "vectordb": {"provider": "sqlite"},
        "global": {"embed_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dimension": 384},
        "ingest": {"chunking_strategy": "structure", "chunk_size": 1800, "chunk_overlap": 200},
    }


def test_v07_manifest_import_with_100_plus_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "legacy"
    source_dir.mkdir(parents=True)
    manifest = {
        f"/docs/file_{idx}.md": {
            "mtime_ns": idx,
            "size": idx + 10,
            "file_hash": f"h{idx}",
            "version": 1,
            "embedding_model": "test-model",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        for idx in range(120)
    }
    (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_memories_db(source_dir / "memories.db")
    (source_dir / "schedules.json").write_text("[]", encoding="utf-8")

    target_conn = _target_connection()
    target_provider = _LegacyTargetProvider(target_conn)
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    result = migration_mod.migrate("v0.7", "v0.8", {"source_dir": source_dir}, _runtime_config())
    assert result["source"] == "v0.7"
    assert target_conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0] == 120


def test_v07_memories_import_preserves_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "legacy"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
    _write_memories_db(source_dir / "memories.db")
    (source_dir / "schedules.json").write_text("[]", encoding="utf-8")

    target_conn = _target_connection()
    target_provider = _LegacyTargetProvider(target_conn)
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    migration_mod.migrate("v0.7", "v0.8", {"source_dir": source_dir}, _runtime_config())

    row = target_conn.execute(
        """
        SELECT memory_type, memory_key, value_json, scope, tags_json, confidence, source_run_id
        FROM lsm_agent_memories
        WHERE id = 'm-1'
        """
    ).fetchone()
    assert row is not None
    assert row["memory_type"] == "project_fact"
    assert row["memory_key"] == "fact_key"
    assert row["value_json"] == '{"answer": 42}'
    assert row["scope"] == "project"
    assert row["tags_json"] == '["x","y"]'
    assert float(row["confidence"]) == pytest.approx(0.9)
    assert row["source_run_id"] == "run-legacy"

    candidate = target_conn.execute(
        "SELECT provenance, rationale, status FROM lsm_agent_memory_candidates WHERE id = 'c-1'"
    ).fetchone()
    assert candidate is not None
    assert candidate["provenance"] == "legacy"
    assert candidate["rationale"] == "kept"
    assert candidate["status"] == "pending"


def test_v07_schedules_import_preserves_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "legacy"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text("{}", encoding="utf-8")
    _write_memories_db(source_dir / "memories.db")
    schedules_payload = [
        {
            "schedule_id": "sched-1",
            "agent_name": "general",
            "last_run_at": "2026-01-01T00:00:00+00:00",
            "next_run_at": "2026-01-02T00:00:00+00:00",
            "last_status": "ok",
            "last_error": None,
            "queued_runs": 2,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    (source_dir / "schedules.json").write_text(json.dumps(schedules_payload), encoding="utf-8")

    target_conn = _target_connection()
    target_provider = _LegacyTargetProvider(target_conn)
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    migration_mod.migrate("v0.7", "v0.8", {"source_dir": source_dir}, _runtime_config())

    row = target_conn.execute(
        """
        SELECT agent_name, next_run_at, last_status, queued_runs
        FROM lsm_agent_schedules
        WHERE schedule_id = 'sched-1'
        """
    ).fetchone()
    assert row is not None
    assert row["agent_name"] == "general"
    assert row["next_run_at"] == "2026-01-02T00:00:00+00:00"
    assert row["last_status"] == "ok"
    assert int(row["queued_runs"]) == 2


def test_v07_missing_files_warn_and_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "legacy"
    source_dir.mkdir(parents=True)

    target_conn = _target_connection()
    target_provider = _LegacyTargetProvider(target_conn)
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    missing_paths: list[Path] = []
    monkeypatch.setattr(
        migration_mod,
        "_warn_legacy_missing",
        lambda path: missing_paths.append(Path(path)),
    )

    result = migration_mod.migrate("v0.7", "v0.8", {"source_dir": source_dir}, _runtime_config())

    assert result["source"] == "v0.7"
    assert len(missing_paths) >= 3
    assert target_conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0] == 0
