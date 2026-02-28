"""Tests for the migration framework."""

from __future__ import annotations

import sqlite3

import pytest

from lsm.config.models import VectorDBConfig
from lsm.db import migration as migration_mod
from lsm.vectordb.base import VectorDBGetResult


class _FakeProvider:
    def __init__(
        self,
        rows: list[dict[str, object]] | None = None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._rows: dict[str, dict[str, object]] = {
            str(item["id"]): dict(item) for item in (rows or [])
        }
        self.connection = connection

    def count(self) -> int:
        return len(self._rows)

    def get(self, *, limit=None, offset=0, include=None, **kwargs):  # noqa: ANN001
        _ = include, kwargs
        keys = sorted(self._rows.keys())
        if offset:
            keys = keys[offset:]
        if limit is not None:
            keys = keys[:limit]
        docs = [str(self._rows[key]["document"]) for key in keys]
        metas = [dict(self._rows[key]["metadata"]) for key in keys]
        embeddings = [list(self._rows[key]["embedding"]) for key in keys]
        return VectorDBGetResult(
            ids=keys,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

    def add_chunks(self, ids, documents, metadatas, embeddings):  # noqa: ANN001
        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self._rows[str(chunk_id)] = {
                "id": str(chunk_id),
                "document": str(document),
                "metadata": dict(metadata or {}),
                "embedding": list(embedding or []),
            }
            if self.connection is not None:
                self.connection.execute(
                    """
                    INSERT INTO lsm_chunks(chunk_id, source_path, chunk_text, is_current)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source_path=excluded.source_path,
                        chunk_text=excluded.chunk_text,
                        is_current=excluded.is_current
                    """,
                    (
                        str(chunk_id),
                        str((metadata or {}).get("source_path", "")),
                        str(document),
                    ),
                )
        if self.connection is not None:
            self.connection.commit()


def _aux_connection() -> sqlite3.Connection:
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


def _seed_aux_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO lsm_schema_versions (
            id, manifest_version, lsm_version, embedding_model, embedding_dim,
            chunking_strategy, chunk_size, chunk_overlap, created_at, last_ingest_at
        ) VALUES (1, NULL, '0.8.0', 'test-model', 384, 'structure', 1800, 200, 't0', 't0')
        """
    )
    conn.execute(
        """
        INSERT INTO lsm_manifest (
            source_path, mtime_ns, file_size, file_hash, version, embedding_model, schema_version_id, updated_at
        ) VALUES ('/tmp/a.md', 1, 2, 'h1', 1, 'test-model', 1, 't0')
        """
    )
    conn.execute(
        """
        INSERT INTO lsm_agent_memories (
            id, memory_type, memory_key, value_json, scope, tags_json,
            confidence, created_at, last_used_at, expires_at, source_run_id
        ) VALUES ('m1', 'project_fact', 'k', '{}', 'project', '[]', 1.0, 't0', 't0', NULL, 'run-1')
        """
    )
    conn.execute(
        """
        INSERT INTO lsm_agent_memory_candidates (
            id, memory_id, provenance, rationale, status, created_at, updated_at
        ) VALUES ('c1', 'm1', 'prov', 'rat', 'pending', 't0', 't0')
        """
    )
    conn.execute(
        """
        INSERT INTO lsm_agent_schedules (
            schedule_id, agent_name, last_run_at, next_run_at, last_status, last_error, queued_runs, updated_at
        ) VALUES ('s1', 'general', NULL, 't1', 'idle', NULL, 0, 't0')
        """
    )
    conn.execute(
        """
        INSERT INTO lsm_stats_cache (cache_key, cached_at, chunk_count, stats_json)
        VALUES ('collection_stats', 1.0, 1, '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO lsm_remote_cache (cache_key, provider, response_json, created_at, expires_at)
        VALUES ('query:test:q1', 'test', '{}', 't0', NULL)
        """
    )
    conn.commit()


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "document": "alpha document",
            "metadata": {"source_path": "/tmp/a.md", "chunk_index": 0, "is_current": 1},
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "id": "b",
            "document": "beta document",
            "metadata": {"source_path": "/tmp/b.md", "chunk_index": 0, "is_current": 1},
            "embedding": [0.4, 0.5, 0.6],
        },
    ]


@pytest.mark.parametrize(
    ("source_kind", "target_kind"),
    [
        ("chroma", "sqlite"),
        ("sqlite", "postgresql"),
        ("postgresql", "sqlite"),
    ],
)
def test_framework_migration_preserves_vectors_and_state(
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    target_kind: str,
) -> None:
    source_conn = _aux_connection()
    target_conn = _aux_connection()
    _seed_aux_tables(source_conn)
    source_provider = _FakeProvider(_sample_rows(), connection=source_conn)
    target_provider = _FakeProvider([], connection=target_conn)

    monkeypatch.setattr(
        migration_mod,
        "_provider_from_source",
        lambda *_args, **_kwargs: source_provider,
    )
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    result = migration_mod.migrate(
        source_kind,
        target_kind,
        {"vectordb": {"provider": source_kind}},
        {
            "vectordb": {"provider": target_kind},
            "global": {"embed_model": "test-model", "embedding_dimension": 384},
            "ingest": {"chunking_strategy": "structure", "chunk_size": 1800, "chunk_overlap": 200},
        },
    )

    assert result["migrated_vectors"] == 2
    assert result["total_vectors"] == 2
    assert target_provider.count() == 2
    assert target_provider.get(limit=10, include=["documents"]).documents == [
        "alpha document",
        "beta document",
    ]
    copied_manifest_count = target_conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0]
    assert copied_manifest_count == 1


def test_framework_validation_detects_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenTargetProvider(_FakeProvider):
        def add_chunks(self, ids, documents, metadatas, embeddings):  # noqa: ANN001
            super().add_chunks(ids[:1], documents[:1], metadatas[:1], embeddings[:1])

    source_conn = _aux_connection()
    target_conn = _aux_connection()
    _seed_aux_tables(source_conn)
    source_provider = _FakeProvider(_sample_rows(), connection=source_conn)
    target_provider = _BrokenTargetProvider([], connection=target_conn)

    monkeypatch.setattr(
        migration_mod,
        "_provider_from_source",
        lambda *_args, **_kwargs: source_provider,
    )
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    with pytest.raises(migration_mod.MigrationValidationError, match="vector_rows"):
        migration_mod.migrate(
            "sqlite",
            "sqlite",
            {"vectordb": {"provider": "sqlite"}},
            {
                "vectordb": {"provider": "sqlite"},
                "global": {"embed_model": "test-model", "embedding_dimension": 384},
                "ingest": {"chunking_strategy": "structure", "chunk_size": 1800, "chunk_overlap": 200},
            },
        )


def test_validate_migration_supports_explicit_vector_table_key() -> None:
    conn = _aux_connection()
    conn.execute(
        """
        INSERT INTO lsm_chunks(chunk_id, source_path, chunk_text, is_current)
        VALUES ('a', '/tmp/a.md', 'alpha', 1), ('b', '/tmp/b.md', 'beta', 1)
        """
    )
    conn.commit()
    migration_mod._record_validation_counts(conn, {"vector_rows:lsm_chunks": 2})

    result = migration_mod.validate_migration(conn)
    assert result["checked"] >= 1
    assert result["mismatches"] == {}


def test_count_vector_rows_explicit_key_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cursor:
        @staticmethod
        def fetchone() -> tuple[int]:
            return (5,)

    execute_calls: list[str] = []

    def _fake_execute(_conn, query: str, _params=()):  # noqa: ANN001
        execute_calls.append(query)
        return _Cursor()

    def _fake_table_exists(_conn, table_name: str) -> bool:  # noqa: ANN001
        return table_name == "lsm_chunks"

    monkeypatch.setattr(migration_mod, "_execute", _fake_execute)
    monkeypatch.setattr(migration_mod, "_table_exists", _fake_table_exists)

    # Explicit table key should not fallback to the generic lsm_chunks row count path.
    assert migration_mod._count_vector_rows(object(), "vector_rows:chunks_target") == 0
    assert execute_calls == []


def test_framework_migration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    source_conn = _aux_connection()
    target_conn = _aux_connection()
    _seed_aux_tables(source_conn)
    source_provider = _FakeProvider(_sample_rows(), connection=source_conn)
    target_provider = _FakeProvider([], connection=target_conn)

    monkeypatch.setattr(
        migration_mod,
        "_provider_from_source",
        lambda *_args, **_kwargs: source_provider,
    )
    monkeypatch.setattr(
        migration_mod,
        "_provider_from_target",
        lambda *_args, **_kwargs: target_provider,
    )

    runtime_config = {
        "vectordb": {"provider": "sqlite"},
        "global": {"embed_model": "test-model", "embedding_dimension": 384},
        "ingest": {"chunking_strategy": "structure", "chunk_size": 1800, "chunk_overlap": 200},
    }
    migration_mod.migrate("sqlite", "sqlite", {"vectordb": {"provider": "sqlite"}}, runtime_config)
    migration_mod.migrate("sqlite", "sqlite", {"vectordb": {"provider": "sqlite"}}, runtime_config)

    assert target_provider.count() == 2
    assert target_conn.execute("SELECT COUNT(*) FROM lsm_manifest").fetchone()[0] == 1
