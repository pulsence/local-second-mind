from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

from lsm.agents.memory import Memory, SQLiteMemoryStore
from lsm.agents.memory.api import (
    memory_expire,
    memory_promote,
    memory_put_candidate,
    memory_search,
)
from lsm.agents.memory.models import now_utc
from lsm.config.models.agents import MemoryConfig


def _sqlite_store(tmp_path: Path, name: str = "memory_api.sqlite3") -> SQLiteMemoryStore:
    config = MemoryConfig(storage_backend="sqlite")
    conn = sqlite3.connect(str(tmp_path / name))
    return SQLiteMemoryStore(conn, config, owns_connection=True)


def test_memory_api_put_promote_and_ranked_search(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        pinned = Memory(
            type="pinned",
            key="writing_style",
            value={"tone": "concise"},
            scope="project",
            tags=["aquinas", "writing"],
            confidence=0.5,
            last_used_at=now_utc() - timedelta(days=30),
            source_run_id="run-pinned",
        )
        recent = Memory(
            type="project_fact",
            key="eucharist_note",
            value={"claim": "instrumental causality"},
            scope="project",
            tags=["aquinas", "theology"],
            confidence=0.9,
            last_used_at=now_utc() - timedelta(days=1),
            source_run_id="run-recent",
        )

        pinned_candidate = memory_put_candidate(
            store,
            pinned,
            provenance="research_agent",
            rationale="Persistent writing preference",
        )
        recent_candidate = memory_put_candidate(
            store,
            recent,
            provenance="research_agent",
            rationale="Repeated project fact",
        )
        memory_promote(store, pinned_candidate)
        memory_promote(store, recent_candidate)

        before = store.get(pinned.id).last_used_at
        results = memory_search(
            store,
            scope="project",
            tags=["aquinas"],
            limit=5,
            update_last_used=False,
        )
        assert len(results) == 2
        assert results[0].id == pinned.id
        assert store.get(pinned.id).last_used_at == before

        updated = memory_search(
            store,
            scope="project",
            tags=["aquinas"],
            limit=5,
            update_last_used=True,
        )
        assert len(updated) == 2
        assert store.get(pinned.id).last_used_at >= before
    finally:
        store.close()


def test_memory_api_respects_token_budget(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        small = Memory(
            type="project_fact",
            key="small",
            value={"text": "brief note"},
            scope="project",
            tags=["token"],
            confidence=1.0,
            source_run_id="run-small",
        )
        large = Memory(
            type="project_fact",
            key="large",
            value={"text": "x" * 4000},
            scope="project",
            tags=["token"],
            confidence=0.7,
            source_run_id="run-large",
        )
        memory_promote(
            store,
            memory_put_candidate(store, small, provenance="agent", rationale="small"),
        )
        memory_promote(
            store,
            memory_put_candidate(store, large, provenance="agent", rationale="large"),
        )

        results = memory_search(
            store,
            scope="project",
            tags=["token"],
            limit=5,
            token_budget=50,
        )
        assert len(results) == 1
        assert results[0].key == "small"
    finally:
        store.close()


def test_memory_api_expire_cleanup(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        expired = Memory(
            type="cache",
            key="old_cache",
            value={"query": "stale"},
            scope="global",
            tags=["cache"],
            expires_at=now_utc() - timedelta(hours=2),
            source_run_id="run-expire",
        )
        candidate_id = memory_put_candidate(
            store,
            expired,
            provenance="query_agent",
            rationale="transient cache",
        )
        memory_promote(store, candidate_id)

        removed = memory_expire(store)
        assert removed == 1
    finally:
        store.close()

