"""
Storage backends for agent memory.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from lsm.config.models.agents import AgentConfig, MemoryConfig
from lsm.config.models.vectordb import VectorDBConfig

from .models import Memory, MemoryCandidate, now_utc

PSYCOPG2_AVAILABLE = False
psycopg2 = None
RealDictCursor = None
Json = None


def _ensure_postgres_dependencies() -> None:
    """Import psycopg2 lazily."""
    global PSYCOPG2_AVAILABLE, psycopg2, RealDictCursor, Json
    if psycopg2 is not None and RealDictCursor is not None and Json is not None:
        return
    try:
        import psycopg2 as psycopg2_module  # type: ignore[import-not-found]
        from psycopg2.extras import RealDictCursor as real_dict_cursor  # type: ignore[import-not-found]
        from psycopg2.extras import Json as json_wrapper  # type: ignore[import-not-found]

        psycopg2 = psycopg2_module
        RealDictCursor = real_dict_cursor
        Json = json_wrapper
        PSYCOPG2_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - dependency/environment specific
        PSYCOPG2_AVAILABLE = False
        raise RuntimeError("psycopg2 dependency is not available") from exc


class BaseMemoryStore(ABC):
    """
    Abstract memory store interface.
    """

    def __init__(self, memory_config: MemoryConfig) -> None:
        self.memory_config = memory_config

    @abstractmethod
    def put_candidate(self, memory: Memory, provenance: str, rationale: str) -> str:
        """Create a pending memory candidate and return candidate ID."""

    @abstractmethod
    def promote(self, candidate_id: str) -> Memory:
        """Promote a pending candidate and return the associated memory."""

    @abstractmethod
    def reject(self, candidate_id: str) -> None:
        """Reject a candidate."""

    @abstractmethod
    def expire(self) -> int:
        """Delete expired memories and return number deleted."""

    @abstractmethod
    def get(self, memory_id: str) -> Memory:
        """Fetch memory by ID."""

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete memory by ID."""

    @abstractmethod
    def search(
        self,
        scope: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
        limit: int = 20,
        token_budget: Optional[int] = None,
    ) -> List[Memory]:
        """
        Search promoted memories.
        """

    @abstractmethod
    def mark_used(
        self,
        memory_ids: Sequence[str],
        *,
        used_at: Optional[datetime] = None,
    ) -> int:
        """
        Update `last_used_at` for memory IDs and return updated row count.
        """

    @abstractmethod
    def list_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[MemoryCandidate]:
        """List memory candidates."""

    def close(self) -> None:
        """Close any underlying resources."""

    def _apply_ttl_cap(self, memory: Memory) -> Memory:
        """
        Apply memory TTL caps and defaults.
        """
        normalized = replace(memory)
        now = now_utc()
        ttl_cap = self.memory_config.ttl_cap_for_type(normalized.type)

        if normalized.type == "pinned":
            normalized.expires_at = None
            return normalized

        if normalized.expires_at is None and ttl_cap is not None:
            normalized.expires_at = now + ttl_cap
            return normalized

        if normalized.expires_at is not None and ttl_cap is not None:
            max_expiry = now + ttl_cap
            if normalized.expires_at > max_expiry:
                normalized.expires_at = max_expiry
        return normalized

    @staticmethod
    def _estimate_tokens(memory: Memory) -> int:
        payload = {
            "key": memory.key,
            "value": memory.value,
            "tags": memory.tags,
            "type": memory.type,
            "scope": memory.scope,
        }
        return max(1, len(json.dumps(payload, default=str)) // 4)

    @staticmethod
    def _normalize_tags(tags: Optional[Sequence[str]]) -> List[str]:
        if not tags:
            return []
        normalized = [str(tag).strip() for tag in tags if str(tag).strip()]
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _matches_tags(memory_tags: Sequence[str], required_tags: Sequence[str]) -> bool:
        if not required_tags:
            return True
        memory_tag_set = {str(tag).strip() for tag in memory_tags if str(tag).strip()}
        return all(tag in memory_tag_set for tag in required_tags)


class SQLiteMemoryStore(BaseMemoryStore):
    """
    SQLite memory store implementation.
    """

    def __init__(self, db_path: Path, memory_config: MemoryConfig) -> None:
        super().__init__(memory_config)
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                scope TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                expires_at TEXT NULL,
                source_run_id TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_candidates (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL UNIQUE,
                provenance TEXT NOT NULL,
                rationale TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_candidates_status
            ON memory_candidates(status);
            CREATE INDEX IF NOT EXISTS idx_memories_scope_type
            ON memories(scope, memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_expires_at
            ON memories(expires_at);
            """
        )
        self._conn.commit()

    def put_candidate(self, memory: Memory, provenance: str, rationale: str) -> str:
        now = now_utc()
        candidate_id = str(uuid4())
        prepared = self._apply_ttl_cap(memory)
        prepared.validate()

        try:
            self._conn.execute(
                """
                INSERT INTO memories (
                    id, memory_type, memory_key, value_json, scope, tags_json,
                    confidence, created_at, last_used_at, expires_at, source_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prepared.id,
                    prepared.type,
                    prepared.key,
                    json.dumps(prepared.value),
                    prepared.scope,
                    json.dumps(prepared.tags),
                    prepared.confidence,
                    prepared.created_at.isoformat(),
                    prepared.last_used_at.isoformat(),
                    prepared.expires_at.isoformat() if prepared.expires_at else None,
                    prepared.source_run_id,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO memory_candidates (
                    id, memory_id, provenance, rationale, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    prepared.id,
                    str(provenance).strip(),
                    str(rationale).strip(),
                    "pending",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            self._conn.commit()
            return candidate_id
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Failed to store memory candidate: {exc}") from exc

    def promote(self, candidate_id: str) -> Memory:
        row = self._conn.execute(
            "SELECT memory_id FROM memory_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Candidate not found: {candidate_id}")
        now = now_utc().isoformat()
        self._conn.execute(
            "UPDATE memory_candidates SET status = ?, updated_at = ? WHERE id = ?",
            ("promoted", now, candidate_id),
        )
        self._conn.commit()
        return self.get(str(row["memory_id"]))

    def reject(self, candidate_id: str) -> None:
        row = self._conn.execute(
            "SELECT id FROM memory_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Candidate not found: {candidate_id}")
        now = now_utc().isoformat()
        self._conn.execute(
            "UPDATE memory_candidates SET status = ?, updated_at = ? WHERE id = ?",
            ("rejected", now, candidate_id),
        )
        self._conn.commit()

    def expire(self) -> int:
        now = now_utc().isoformat()
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        self._conn.commit()
        return int(cursor.rowcount or 0)

    def get(self, memory_id: str) -> Memory:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory not found: {memory_id}")
        return self._memory_from_row(row)

    def delete(self, memory_id: str) -> None:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()
        if int(cursor.rowcount or 0) == 0:
            raise KeyError(f"Memory not found: {memory_id}")

    def search(
        self,
        scope: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
        limit: int = 20,
        token_budget: Optional[int] = None,
    ) -> List[Memory]:
        limit = max(1, int(limit))
        normalized_tags = self._normalize_tags(tags)
        where_clauses = ["c.status = 'promoted'"]
        params: list[Any] = []
        if scope:
            where_clauses.append("m.scope = ?")
            params.append(str(scope).strip().lower())
        if memory_type:
            where_clauses.append("m.memory_type = ?")
            params.append(str(memory_type).strip().lower())

        sql_limit = max(limit, limit * 5)
        rows = self._conn.execute(
            f"""
            SELECT m.*
            FROM memories m
            JOIN memory_candidates c ON c.memory_id = m.id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY
                CASE WHEN m.memory_type = 'pinned' THEN 0 ELSE 1 END ASC,
                m.last_used_at DESC
            LIMIT ?
            """,
            (*params, sql_limit),
        ).fetchall()

        results: list[Memory] = []
        consumed_tokens = 0
        for row in rows:
            memory = self._memory_from_row(row)
            if not self._matches_tags(memory.tags, normalized_tags):
                continue
            estimated = self._estimate_tokens(memory)
            if token_budget is not None and token_budget > 0:
                if consumed_tokens + estimated > int(token_budget):
                    break
                consumed_tokens += estimated
            results.append(memory)
            if len(results) >= limit:
                break
        return results

    def list_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[MemoryCandidate]:
        limit = max(1, int(limit))
        params: list[Any] = []
        where_clause = ""
        if status:
            where_clause = "WHERE c.status = ?"
            params.append(str(status).strip().lower())
        rows = self._conn.execute(
            f"""
            SELECT
                c.id AS candidate_id,
                c.provenance,
                c.rationale,
                c.status AS candidate_status,
                m.*
            FROM memory_candidates c
            JOIN memories m ON m.id = c.memory_id
            {where_clause}
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

        candidates: list[MemoryCandidate] = []
        for row in rows:
            memory = self._memory_from_row(row)
            candidates.append(
                MemoryCandidate(
                    id=str(row["candidate_id"]),
                    memory=memory,
                    provenance=str(row["provenance"]),
                    rationale=str(row["rationale"]),
                    status=str(row["candidate_status"]),
                )
            )
        return candidates

    def mark_used(
        self,
        memory_ids: Sequence[str],
        *,
        used_at: Optional[datetime] = None,
    ) -> int:
        normalized_ids = [str(item).strip() for item in memory_ids if str(item).strip()]
        if not normalized_ids:
            return 0
        timestamp = (used_at or now_utc()).isoformat()
        placeholders = ", ".join(["?"] * len(normalized_ids))
        cursor = self._conn.execute(
            f"""
            UPDATE memories
            SET last_used_at = ?
            WHERE id IN ({placeholders})
            """,
            (timestamp, *normalized_ids),
        )
        self._conn.commit()
        return int(cursor.rowcount or 0)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> Memory:
        expires_at_raw = row["expires_at"]
        return Memory(
            id=str(row["id"]),
            type=str(row["memory_type"]),
            key=str(row["memory_key"]),
            value=json.loads(str(row["value_json"])),
            scope=str(row["scope"]),
            tags=json.loads(str(row["tags_json"])),
            confidence=float(row["confidence"]),
            created_at=_parse_iso_datetime(str(row["created_at"])),
            last_used_at=_parse_iso_datetime(str(row["last_used_at"])),
            expires_at=(
                _parse_iso_datetime(str(expires_at_raw))
                if expires_at_raw
                else None
            ),
            source_run_id=str(row["source_run_id"]),
        )


class PostgreSQLMemoryStore(BaseMemoryStore):
    """
    PostgreSQL memory store implementation.
    """

    def __init__(
        self,
        connection_string: str,
        memory_config: MemoryConfig,
        table_prefix: str = "agent_memory",
    ) -> None:
        super().__init__(memory_config)
        self.connection_string = str(connection_string).strip()
        if not self.connection_string:
            raise ValueError("PostgreSQL memory store requires connection_string")
        safe_prefix = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in str(table_prefix).strip().lower()
        ).strip("_")
        self.table_prefix = safe_prefix or "agent_memory"
        self.memories_table = f"{self.table_prefix}_memories"
        self.candidates_table = f"{self.table_prefix}_candidates"
        self._schema_initialized = False

    @contextmanager
    def _conn(self):
        _ensure_postgres_dependencies()
        assert psycopg2 is not None  # for type checkers
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._schema_initialized:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.memories_table} (
                        id TEXT PRIMARY KEY,
                        memory_type TEXT NOT NULL,
                        memory_key TEXT NOT NULL,
                        value_json JSONB NOT NULL,
                        scope TEXT NOT NULL,
                        tags_json JSONB NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        last_used_at TIMESTAMPTZ NOT NULL,
                        expires_at TIMESTAMPTZ NULL,
                        source_run_id TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.candidates_table} (
                        id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL UNIQUE REFERENCES {self.memories_table}(id) ON DELETE CASCADE,
                        provenance TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.candidates_table}_status ON {self.candidates_table}(status);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.memories_table}_scope_type ON {self.memories_table}(scope, memory_type);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.memories_table}_expires ON {self.memories_table}(expires_at);"
                )
        self._schema_initialized = True

    def put_candidate(self, memory: Memory, provenance: str, rationale: str) -> str:
        self._ensure_schema()
        now = now_utc()
        candidate_id = str(uuid4())
        prepared = self._apply_ttl_cap(memory)
        prepared.validate()

        assert Json is not None  # for type checkers
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.memories_table} (
                        id, memory_type, memory_key, value_json, scope, tags_json,
                        confidence, created_at, last_used_at, expires_at, source_run_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        prepared.id,
                        prepared.type,
                        prepared.key,
                        Json(prepared.value),
                        prepared.scope,
                        Json(prepared.tags),
                        prepared.confidence,
                        prepared.created_at,
                        prepared.last_used_at,
                        prepared.expires_at,
                        prepared.source_run_id,
                    ),
                )
                cur.execute(
                    f"""
                    INSERT INTO {self.candidates_table} (
                        id, memory_id, provenance, rationale, status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        candidate_id,
                        prepared.id,
                        str(provenance).strip(),
                        str(rationale).strip(),
                        "pending",
                        now,
                        now,
                    ),
                )
        return candidate_id

    def promote(self, candidate_id: str) -> Memory:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(
                    f"SELECT memory_id FROM {self.candidates_table} WHERE id = %s",
                    (candidate_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise KeyError(f"Candidate not found: {candidate_id}")
                cur.execute(
                    f"UPDATE {self.candidates_table} SET status = %s, updated_at = %s WHERE id = %s",
                    ("promoted", now_utc(), candidate_id),
                )
                memory_id = str(row["memory_id"])
        return self.get(memory_id)

    def reject(self, candidate_id: str) -> None:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.candidates_table} SET status = %s, updated_at = %s WHERE id = %s",
                    ("rejected", now_utc(), candidate_id),
                )
                if int(cur.rowcount or 0) == 0:
                    raise KeyError(f"Candidate not found: {candidate_id}")

    def expire(self) -> int:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.memories_table} WHERE expires_at IS NOT NULL AND expires_at <= %s",
                    (now_utc(),),
                )
                return int(cur.rowcount or 0)

    def get(self, memory_id: str) -> Memory:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(
                    f"SELECT * FROM {self.memories_table} WHERE id = %s",
                    (memory_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise KeyError(f"Memory not found: {memory_id}")
                return self._memory_from_row(row)

    def delete(self, memory_id: str) -> None:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.memories_table} WHERE id = %s",
                    (memory_id,),
                )
                if int(cur.rowcount or 0) == 0:
                    raise KeyError(f"Memory not found: {memory_id}")

    def search(
        self,
        scope: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
        limit: int = 20,
        token_budget: Optional[int] = None,
    ) -> List[Memory]:
        self._ensure_schema()
        limit = max(1, int(limit))
        normalized_tags = self._normalize_tags(tags)

        where_clauses = ["c.status = %s"]
        params: list[Any] = ["promoted"]
        if scope:
            where_clauses.append("m.scope = %s")
            params.append(str(scope).strip().lower())
        if memory_type:
            where_clauses.append("m.memory_type = %s")
            params.append(str(memory_type).strip().lower())
        params.append(max(limit, limit * 5))

        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(
                    f"""
                    SELECT m.*
                    FROM {self.memories_table} m
                    JOIN {self.candidates_table} c ON c.memory_id = m.id
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY
                        CASE WHEN m.memory_type = 'pinned' THEN 0 ELSE 1 END ASC,
                        m.last_used_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()

        results: list[Memory] = []
        consumed_tokens = 0
        for row in rows:
            memory = self._memory_from_row(row)
            if not self._matches_tags(memory.tags, normalized_tags):
                continue
            estimated = self._estimate_tokens(memory)
            if token_budget is not None and token_budget > 0:
                if consumed_tokens + estimated > int(token_budget):
                    break
                consumed_tokens += estimated
            results.append(memory)
            if len(results) >= limit:
                break
        return results

    def list_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> List[MemoryCandidate]:
        self._ensure_schema()
        limit = max(1, int(limit))
        params: list[Any] = []
        where_clause = ""
        if status:
            where_clause = "WHERE c.status = %s"
            params.append(str(status).strip().lower())
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(
                    f"""
                    SELECT
                        c.id AS candidate_id,
                        c.provenance,
                        c.rationale,
                        c.status AS candidate_status,
                        m.*
                    FROM {self.candidates_table} c
                    JOIN {self.memories_table} m ON m.id = c.memory_id
                    {where_clause}
                    ORDER BY c.created_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()

        candidates: list[MemoryCandidate] = []
        for row in rows:
            memory = self._memory_from_row(row)
            candidates.append(
                MemoryCandidate(
                    id=str(row["candidate_id"]),
                    memory=memory,
                    provenance=str(row["provenance"]),
                    rationale=str(row["rationale"]),
                    status=str(row["candidate_status"]),
                )
            )
        return candidates

    def mark_used(
        self,
        memory_ids: Sequence[str],
        *,
        used_at: Optional[datetime] = None,
    ) -> int:
        self._ensure_schema()
        normalized_ids = [str(item).strip() for item in memory_ids if str(item).strip()]
        if not normalized_ids:
            return 0
        placeholders = ", ".join(["%s"] * len(normalized_ids))
        timestamp = used_at or now_utc()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self.memories_table}
                    SET last_used_at = %s
                    WHERE id IN ({placeholders})
                    """,
                    (timestamp, *normalized_ids),
                )
                return int(cur.rowcount or 0)

    @staticmethod
    def _memory_from_row(row: Dict[str, Any]) -> Memory:
        return Memory(
            id=str(row["id"]),
            type=str(row["memory_type"]),
            key=str(row["memory_key"]),
            value=row["value_json"],
            scope=str(row["scope"]),
            tags=list(row["tags_json"] or []),
            confidence=float(row["confidence"]),
            created_at=_parse_datetime(row["created_at"]),
            last_used_at=_parse_datetime(row["last_used_at"]),
            expires_at=(
                _parse_datetime(row["expires_at"])
                if row.get("expires_at") is not None
                else None
            ),
            source_run_id=str(row["source_run_id"]),
        )


def create_memory_store(
    agent_config: AgentConfig,
    vectordb_config: VectorDBConfig,
) -> BaseMemoryStore:
    """
    Create the appropriate memory store backend.
    """
    memory_cfg = agent_config.memory
    backend = (memory_cfg.storage_backend or "auto").strip().lower()

    if backend == "auto":
        provider = str(vectordb_config.provider or "chromadb").strip().lower()
        backend = "postgresql" if provider == "postgresql" else "sqlite"

    if backend == "sqlite":
        return SQLiteMemoryStore(memory_cfg.sqlite_path, memory_cfg)
    if backend == "postgresql":
        conn_string = (
            memory_cfg.postgres_connection_string
            or vectordb_config.connection_string
            or _build_postgres_connection_string(vectordb_config)
        )
        if not conn_string:
            raise ValueError(
                "PostgreSQL memory backend requires agents.memory.postgres_connection_string "
                "or vectordb.connection_string (or host/database/user settings)."
            )
        return PostgreSQLMemoryStore(conn_string, memory_cfg, memory_cfg.postgres_table_prefix)

    raise ValueError("Unsupported memory backend. Use 'auto', 'sqlite', or 'postgresql'.")


def _build_postgres_connection_string(vectordb_config: VectorDBConfig) -> Optional[str]:
    if not vectordb_config.host or not vectordb_config.database or not vectordb_config.user:
        return None
    password = vectordb_config.password or ""
    port = vectordb_config.port or 5432
    return (
        f"postgresql://{vectordb_config.user}:{password}"
        f"@{vectordb_config.host}:{port}/{vectordb_config.database}"
    )


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return _parse_iso_datetime(str(value))
