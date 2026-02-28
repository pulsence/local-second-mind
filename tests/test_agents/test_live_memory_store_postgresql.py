"""Live PostgreSQL backend tests for agent memory storage."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from lsm.agents.memory import Memory, PostgreSQLMemoryStore, SQLiteMemoryStore
from lsm.agents.memory.migrations import migrate_memory_store
from lsm.agents.memory.models import now_utc
from lsm.config.models.agents import MemoryConfig


pytestmark = [pytest.mark.live, pytest.mark.live_vectordb]


def _create_live_postgres_store(
    dsn: str,
    *,
    table_prefix: str,
) -> PostgreSQLMemoryStore:
    memory_config = MemoryConfig(
        storage_backend="postgresql",
        postgres_connection_string=dsn,
        postgres_table_prefix=table_prefix,
    )
    return PostgreSQLMemoryStore(
        connection_string=dsn,
        memory_config=memory_config,
        table_prefix=table_prefix,
    )


def _drop_live_memory_tables(dsn: str, table_prefix: str) -> None:
    try:
        import psycopg2
    except Exception:
        return
    memories_table = f"{table_prefix}_memories"
    candidates_table = f"{table_prefix}_candidates"

    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {candidates_table}")
                cur.execute(f"DROP TABLE IF EXISTS {memories_table}")
    finally:
        conn.close()


def test_live_postgresql_memory_store_crud_promote_search_delete(
    live_postgres_connection_string: str,
) -> None:
    table_prefix = f"live_mem_{uuid4().hex[:12]}"
    store = _create_live_postgres_store(
        live_postgres_connection_string,
        table_prefix=table_prefix,
    )
    try:
        memory = Memory(
            type="project_fact",
            key="aquinas_eucharist",
            value={"summary": "Instrumental causality account"},
            scope="project",
            tags=["theology", "thomism"],
            confidence=0.9,
            source_run_id="live-run-001",
        )
        candidate_id = store.put_candidate(
            memory=memory,
            provenance="research_agent",
            rationale="Repeatedly used framing",
        )
        assert candidate_id

        # Pending candidates are not returned by search.
        assert store.search(scope="project", tags=["theology"]) == []

        promoted = store.promote(candidate_id)
        assert promoted.id == memory.id
        assert promoted.key == "aquinas_eucharist"

        fetched = store.get(memory.id)
        assert fetched.scope == "project"
        assert fetched.type == "project_fact"

        searched = store.search(
            scope="project",
            tags=["theology"],
            memory_type="project_fact",
            limit=10,
        )
        assert len(searched) == 1
        assert searched[0].id == memory.id

        promoted_candidates = store.list_candidates(status="promoted", limit=20)
        assert any(item.id == candidate_id for item in promoted_candidates)

        before = store.get(memory.id)
        updated_count = store.mark_used([memory.id])
        assert updated_count == 1
        after = store.get(memory.id)
        assert after.last_used_at >= before.last_used_at

        store.delete(memory.id)
        with pytest.raises(KeyError, match="Memory not found"):
            store.get(memory.id)
    finally:
        store.close()
        _drop_live_memory_tables(live_postgres_connection_string, table_prefix)


def test_live_postgresql_memory_store_reject_expire_and_ttl_caps(
    live_postgres_connection_string: str,
) -> None:
    table_prefix = f"live_mem_{uuid4().hex[:12]}"
    store = _create_live_postgres_store(
        live_postgres_connection_string,
        table_prefix=table_prefix,
    )
    try:
        rejected_memory = Memory(
            type="task_state",
            key="draft_state",
            value={"status": "incomplete"},
            scope="agent",
            tags=["todo"],
            source_run_id="live-run-002",
        )
        rejected_candidate = store.put_candidate(
            memory=rejected_memory,
            provenance="curator_agent",
            rationale="Interim run artifact",
        )
        store.reject(rejected_candidate)
        assert store.search(scope="agent", tags=["todo"]) == []
        rejected_rows = store.list_candidates(status="rejected", limit=20)
        assert any(item.id == rejected_candidate for item in rejected_rows)

        expired_memory = Memory(
            type="cache",
            key="stale_query_cache",
            value={"query": "old"},
            scope="global",
            tags=["cache"],
            expires_at=now_utc() - timedelta(hours=2),
            source_run_id="live-run-003",
        )
        expired_candidate = store.put_candidate(
            memory=expired_memory,
            provenance="query_agent",
            rationale="Transient cache",
        )
        store.promote(expired_candidate)
        assert store.expire() == 1
        with pytest.raises(KeyError, match="Memory not found"):
            store.get(expired_memory.id)

        ttl_capped_memory = Memory(
            type="cache",
            key="long_cache_ttl",
            value={"key": "value"},
            scope="project",
            tags=["cache"],
            expires_at=now_utc() + timedelta(days=10),
            source_run_id="live-run-004",
        )
        ttl_candidate = store.put_candidate(
            memory=ttl_capped_memory,
            provenance="query_agent",
            rationale="Cache entry",
        )
        store.promote(ttl_candidate)
        capped = store.get(ttl_capped_memory.id)
        assert capped.expires_at is not None
        assert capped.expires_at <= now_utc() + timedelta(hours=24, minutes=2)
    finally:
        store.close()
        _drop_live_memory_tables(live_postgres_connection_string, table_prefix)


def test_live_memory_migration_postgresql_to_sqlite(
    tmp_path: Path,
    live_postgres_connection_string: str,
) -> None:
    table_prefix = f"live_mem_{uuid4().hex[:12]}"
    source = _create_live_postgres_store(
        live_postgres_connection_string,
        table_prefix=table_prefix,
    )
    target_cfg = MemoryConfig(storage_backend="sqlite")
    target_conn = sqlite3.connect(str(tmp_path / "memory_target.sqlite3"))
    target = SQLiteMemoryStore(target_conn, target_cfg, owns_connection=True)
    try:
        promoted_memory = Memory(
            type="project_fact",
            key="promoted",
            value={"v": 1},
            scope="project",
            tags=["x"],
            source_run_id="live-run-a",
        )
        promoted_candidate = source.put_candidate(
            promoted_memory,
            provenance="research",
            rationale="Important",
        )
        source.promote(promoted_candidate)

        rejected_memory = Memory(
            type="task_state",
            key="rejected",
            value={"v": 2},
            scope="agent",
            tags=["y"],
            source_run_id="live-run-b",
        )
        rejected_candidate = source.put_candidate(
            rejected_memory,
            provenance="curator",
            rationale="Not stable",
        )
        source.reject(rejected_candidate)

        counts = migrate_memory_store(source, target, include_rejected=True)
        assert counts["migrated"] == 2
        assert counts["promoted"] == 1
        assert counts["rejected"] == 1

        promoted_results = target.search(scope="project", memory_type="project_fact")
        assert len(promoted_results) == 1
        assert promoted_results[0].key == "promoted"

        rejected_results = target.list_candidates(status="rejected")
        assert len(rejected_results) == 1
        assert rejected_results[0].memory.key == "rejected"
    finally:
        source.close()
        target.close()
        _drop_live_memory_tables(live_postgres_connection_string, table_prefix)


def test_live_memory_migration_sqlite_to_postgresql(
    tmp_path: Path,
    live_postgres_connection_string: str,
) -> None:
    source_cfg = MemoryConfig(storage_backend="sqlite")
    source_conn = sqlite3.connect(str(tmp_path / "memory_source.sqlite3"))
    source = SQLiteMemoryStore(source_conn, source_cfg, owns_connection=True)
    table_prefix = f"live_mem_{uuid4().hex[:12]}"
    target = _create_live_postgres_store(
        live_postgres_connection_string,
        table_prefix=table_prefix,
    )
    try:
        promoted_memory = Memory(
            type="project_fact",
            key="promoted",
            value={"v": 1},
            scope="project",
            tags=["x"],
            source_run_id="live-run-a",
        )
        promoted_candidate = source.put_candidate(
            promoted_memory,
            provenance="research",
            rationale="Important",
        )
        source.promote(promoted_candidate)

        rejected_memory = Memory(
            type="task_state",
            key="rejected",
            value={"v": 2},
            scope="agent",
            tags=["y"],
            source_run_id="live-run-b",
        )
        rejected_candidate = source.put_candidate(
            rejected_memory,
            provenance="curator",
            rationale="Not stable",
        )
        source.reject(rejected_candidate)

        counts = migrate_memory_store(source, target, include_rejected=True)
        assert counts["migrated"] == 2
        assert counts["promoted"] == 1
        assert counts["rejected"] == 1

        promoted_results = target.search(scope="project", memory_type="project_fact")
        assert len(promoted_results) == 1
        assert promoted_results[0].key == "promoted"

        rejected_results = target.list_candidates(status="rejected")
        assert len(rejected_results) == 1
        assert rejected_results[0].memory.key == "rejected"
    finally:
        source.close()
        target.close()
        _drop_live_memory_tables(live_postgres_connection_string, table_prefix)
