from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from lsm.agents.memory import (
    Memory,
    PostgreSQLMemoryStore,
    SQLiteMemoryStore,
    create_memory_store,
)
from lsm.agents.memory.migrations import migrate_memory_store
from lsm.agents.memory.models import now_utc
from lsm.config.models.agents import AgentConfig, MemoryConfig, SandboxConfig
from lsm.config.models.vectordb import VectorDBConfig


def _sqlite_store(tmp_path: Path, name: str = "memory.sqlite3") -> SQLiteMemoryStore:
    config = MemoryConfig(storage_backend="sqlite", sqlite_path=tmp_path / name)
    return SQLiteMemoryStore(config.sqlite_path, config)


def test_sqlite_memory_store_crud_promote_and_search(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        memory = Memory(
            type="project_fact",
            key="aquinas_eucharist",
            value={"summary": "Instrumental causality account"},
            scope="project",
            tags=["theology", "thomism"],
            confidence=0.9,
            source_run_id="run-001",
        )

        candidate_id = store.put_candidate(
            memory=memory,
            provenance="research_agent",
            rationale="Repeatedly used framing",
        )
        assert candidate_id
        assert store.search(scope="project") == []

        promoted = store.promote(candidate_id)
        assert promoted.id == memory.id

        fetched = store.get(memory.id)
        assert fetched.key == "aquinas_eucharist"
        assert fetched.scope == "project"

        results = store.search(
            scope="project",
            tags=["theology"],
            memory_type="project_fact",
            limit=10,
        )
        assert len(results) == 1
        assert results[0].id == memory.id

        store.delete(memory.id)
        with pytest.raises(KeyError, match="Memory not found"):
            store.get(memory.id)
    finally:
        store.close()


def test_sqlite_memory_store_reject_keeps_memory_unsearchable(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        memory = Memory(
            type="task_state",
            key="draft_state",
            value={"status": "incomplete"},
            scope="agent",
            tags=["todo"],
            source_run_id="run-002",
        )
        candidate_id = store.put_candidate(
            memory=memory,
            provenance="curator_agent",
            rationale="Interim run artifact",
        )
        store.reject(candidate_id)

        assert store.search(scope="agent", tags=["todo"]) == []
        rejected = store.list_candidates(status="rejected")
        assert len(rejected) == 1
        assert rejected[0].memory.id == memory.id
    finally:
        store.close()


def test_sqlite_memory_store_expire_removes_expired_memories(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        memory = Memory(
            type="cache",
            key="stale_query_cache",
            value={"query": "old"},
            scope="global",
            tags=["cache"],
            expires_at=now_utc() - timedelta(hours=2),
            source_run_id="run-003",
        )
        candidate_id = store.put_candidate(
            memory=memory,
            provenance="query_agent",
            rationale="Transient cache",
        )
        store.promote(candidate_id)
        assert store.expire() == 1
        with pytest.raises(KeyError, match="Memory not found"):
            store.get(memory.id)
    finally:
        store.close()


def test_sqlite_memory_store_enforces_ttl_caps(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        cache_memory = Memory(
            type="cache",
            key="long_cache",
            value={"k": "v"},
            scope="project",
            tags=["cache"],
            expires_at=now_utc() + timedelta(days=10),
            source_run_id="run-004",
        )
        cache_candidate = store.put_candidate(
            memory=cache_memory,
            provenance="query_agent",
            rationale="Cache entry",
        )
        store.promote(cache_candidate)
        cache_fetched = store.get(cache_memory.id)
        assert cache_fetched.expires_at is not None
        assert cache_fetched.expires_at <= now_utc() + timedelta(hours=24, minutes=2)

        pinned_memory = Memory(
            type="pinned",
            key="permanent_pref",
            value={"style": "concise"},
            scope="agent",
            tags=["preferences"],
            expires_at=now_utc() + timedelta(days=3),
            source_run_id="run-005",
        )
        pinned_candidate = store.put_candidate(
            memory=pinned_memory,
            provenance="user",
            rationale="Explicit pin",
        )
        store.promote(pinned_candidate)
        pinned_fetched = store.get(pinned_memory.id)
        assert pinned_fetched.expires_at is None
    finally:
        store.close()


def test_create_memory_store_auto_selects_backend(tmp_path: Path) -> None:
    agent_cfg = AgentConfig(
        enabled=True,
        agents_folder=tmp_path / "Agents",
        sandbox=SandboxConfig(),
        memory=MemoryConfig(
            storage_backend="auto",
            sqlite_path=tmp_path / "memory.sqlite3",
        ),
    )

    chroma_cfg = VectorDBConfig(
        provider="chromadb",
        persist_dir=tmp_path / ".chroma",
        collection="local_kb",
    )
    sqlite_store = create_memory_store(agent_cfg, chroma_cfg)
    assert isinstance(sqlite_store, SQLiteMemoryStore)
    sqlite_store.close()

    postgres_cfg = VectorDBConfig(
        provider="postgresql",
        collection="local_kb",
        connection_string="postgresql://user:pass@localhost:5432/lsm",
    )
    postgres_store = create_memory_store(agent_cfg, postgres_cfg)
    assert isinstance(postgres_store, PostgreSQLMemoryStore)
    postgres_store.close()


def test_migrate_memory_store_between_sqlite_backends(tmp_path: Path) -> None:
    source = _sqlite_store(tmp_path, "source.sqlite3")
    target = _sqlite_store(tmp_path, "target.sqlite3")
    try:
        promoted_memory = Memory(
            type="project_fact",
            key="promoted",
            value={"v": 1},
            scope="project",
            tags=["x"],
            source_run_id="run-a",
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
            source_run_id="run-b",
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
