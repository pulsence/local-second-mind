"""
Migration helpers for memory stores.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

from lsm.config.models.agents import MemoryConfig

from .store import BaseMemoryStore, PostgreSQLMemoryStore, SQLiteMemoryStore


def migrate_memory_store(
    source: BaseMemoryStore,
    target: BaseMemoryStore,
    *,
    include_pending: bool = True,
    include_rejected: bool = True,
) -> Dict[str, int]:
    """
    Migrate memory candidates and memories between stores.

    Returns:
        Dictionary with migration counts.
    """
    candidates = source.list_candidates(limit=100000)
    migrated = 0
    promoted = 0
    pending = 0
    rejected = 0
    skipped = 0

    for candidate in candidates:
        status = candidate.status
        if status == "pending" and not include_pending:
            skipped += 1
            continue
        if status == "rejected" and not include_rejected:
            skipped += 1
            continue

        try:
            target.delete(candidate.memory.id)
        except KeyError:
            pass

        new_candidate_id = target.put_candidate(
            candidate.memory,
            candidate.provenance,
            candidate.rationale,
        )
        if status == "promoted":
            target.promote(new_candidate_id)
            promoted += 1
        elif status == "rejected":
            target.reject(new_candidate_id)
            rejected += 1
        else:
            pending += 1
        migrated += 1

    return {
        "migrated": migrated,
        "promoted": promoted,
        "pending": pending,
        "rejected": rejected,
        "skipped": skipped,
    }


def migrate_sqlite_to_postgresql(
    sqlite_path: Path,
    postgres_connection_string: str,
    memory_config: MemoryConfig,
) -> Dict[str, int]:
    """
    Migrate memories from SQLite to PostgreSQL.
    """
    source_conn = sqlite3.connect(str(sqlite_path))
    target = PostgreSQLMemoryStore(postgres_connection_string, memory_config)
    source = SQLiteMemoryStore(source_conn, memory_config, owns_connection=True)
    try:
        return migrate_memory_store(source, target)
    finally:
        source.close()
        target.close()


def migrate_postgresql_to_sqlite(
    postgres_connection_string: str,
    sqlite_path: Path,
    memory_config: MemoryConfig,
) -> Dict[str, int]:
    """
    Migrate memories from PostgreSQL to SQLite.
    """
    target_conn = sqlite3.connect(str(sqlite_path))
    source = PostgreSQLMemoryStore(postgres_connection_string, memory_config)
    target = SQLiteMemoryStore(target_conn, memory_config, owns_connection=True)
    try:
        return migrate_memory_store(source, target)
    finally:
        source.close()
        target.close()
