"""Database migration framework for cross-backend state transfer."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from lsm import __version__ as LSM_VERSION
from lsm.config.models import VectorDBConfig
from lsm.logging import get_logger
from lsm.vectordb import create_vectordb_provider
from lsm.vectordb.chromadb import ChromaDBProvider

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int, int, str], None]


class MigrationSource(str, Enum):
    """Supported migration source backends."""

    CHROMA = "chroma"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    V07_LEGACY = "v0.7"


class MigrationTarget(str, Enum):
    """Supported migration target backends."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class MigrationValidationError(RuntimeError):
    """Raised when migration row-count validation detects mismatches."""


@dataclass(frozen=True)
class AuxiliaryTableSpec:
    name: str
    primary_key: str
    sqlite_ddl: str
    postgres_ddl: str


_AUX_TABLE_SPECS: tuple[AuxiliaryTableSpec, ...] = (
    AuxiliaryTableSpec(
        name="lsm_schema_versions",
        primary_key="id",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_schema_versions (
                id                INTEGER PRIMARY KEY,
                manifest_version  INTEGER,
                lsm_version       TEXT,
                embedding_model   TEXT,
                embedding_dim     INTEGER,
                chunking_strategy TEXT,
                chunk_size        INTEGER,
                chunk_overlap     INTEGER,
                created_at        TEXT,
                last_ingest_at    TEXT
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_schema_versions (
                id                BIGINT PRIMARY KEY,
                manifest_version  BIGINT,
                lsm_version       TEXT,
                embedding_model   TEXT,
                embedding_dim     BIGINT,
                chunking_strategy TEXT,
                chunk_size        BIGINT,
                chunk_overlap     BIGINT,
                created_at        TEXT,
                last_ingest_at    TEXT
            )
        """,
    ),
    AuxiliaryTableSpec(
        name="lsm_manifest",
        primary_key="source_path",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_manifest (
                source_path       TEXT PRIMARY KEY,
                mtime_ns          INTEGER,
                file_size         INTEGER,
                file_hash         TEXT,
                version           INTEGER,
                embedding_model   TEXT,
                schema_version_id INTEGER,
                updated_at        TEXT
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_manifest (
                source_path       TEXT PRIMARY KEY,
                mtime_ns          BIGINT,
                file_size         BIGINT,
                file_hash         TEXT,
                version           BIGINT,
                embedding_model   TEXT,
                schema_version_id BIGINT,
                updated_at        TEXT
            )
        """,
    ),
    AuxiliaryTableSpec(
        name="lsm_agent_memories",
        primary_key="id",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_agent_memories (
                id               TEXT PRIMARY KEY,
                memory_type      TEXT NOT NULL,
                memory_key       TEXT NOT NULL,
                value_json       TEXT NOT NULL,
                scope            TEXT NOT NULL,
                tags_json        TEXT NOT NULL,
                confidence       REAL NOT NULL,
                created_at       TEXT NOT NULL,
                last_used_at     TEXT NOT NULL,
                expires_at       TEXT NULL,
                source_run_id    TEXT NOT NULL
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_agent_memories (
                id               TEXT PRIMARY KEY,
                memory_type      TEXT NOT NULL,
                memory_key       TEXT NOT NULL,
                value_json       TEXT NOT NULL,
                scope            TEXT NOT NULL,
                tags_json        TEXT NOT NULL,
                confidence       DOUBLE PRECISION NOT NULL,
                created_at       TEXT NOT NULL,
                last_used_at     TEXT NOT NULL,
                expires_at       TEXT NULL,
                source_run_id    TEXT NOT NULL
            )
        """,
    ),
    AuxiliaryTableSpec(
        name="lsm_agent_memory_candidates",
        primary_key="id",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_agent_memory_candidates (
                id            TEXT PRIMARY KEY,
                memory_id     TEXT NOT NULL UNIQUE,
                provenance    TEXT NOT NULL,
                rationale     TEXT NOT NULL,
                status        TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_agent_memory_candidates (
                id            TEXT PRIMARY KEY,
                memory_id     TEXT NOT NULL UNIQUE,
                provenance    TEXT NOT NULL,
                rationale     TEXT NOT NULL,
                status        TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
        """,
    ),
    AuxiliaryTableSpec(
        name="lsm_agent_schedules",
        primary_key="schedule_id",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_agent_schedules (
                schedule_id   TEXT PRIMARY KEY,
                agent_name    TEXT NOT NULL,
                last_run_at   TEXT,
                next_run_at   TEXT NOT NULL,
                last_status   TEXT DEFAULT 'idle',
                last_error    TEXT,
                queued_runs   INTEGER DEFAULT 0,
                updated_at    TEXT NOT NULL
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_agent_schedules (
                schedule_id   TEXT PRIMARY KEY,
                agent_name    TEXT NOT NULL,
                last_run_at   TEXT,
                next_run_at   TEXT NOT NULL,
                last_status   TEXT DEFAULT 'idle',
                last_error    TEXT,
                queued_runs   BIGINT DEFAULT 0,
                updated_at    TEXT NOT NULL
            )
        """,
    ),
    AuxiliaryTableSpec(
        name="lsm_stats_cache",
        primary_key="cache_key",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_stats_cache (
                cache_key      TEXT PRIMARY KEY,
                cached_at      REAL NOT NULL,
                chunk_count    INTEGER NOT NULL,
                stats_json     TEXT NOT NULL
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_stats_cache (
                cache_key      TEXT PRIMARY KEY,
                cached_at      DOUBLE PRECISION NOT NULL,
                chunk_count    BIGINT NOT NULL,
                stats_json     TEXT NOT NULL
            )
        """,
    ),
    AuxiliaryTableSpec(
        name="lsm_remote_cache",
        primary_key="cache_key",
        sqlite_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_remote_cache (
                cache_key      TEXT PRIMARY KEY,
                provider       TEXT NOT NULL,
                response_json  TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                expires_at     TEXT
            )
        """,
        postgres_ddl="""
            CREATE TABLE IF NOT EXISTS lsm_remote_cache (
                cache_key      TEXT PRIMARY KEY,
                provider       TEXT NOT NULL,
                response_json  TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                expires_at     TEXT
            )
        """,
    ),
)

_VALIDATION_TABLE = "lsm_migration_validation"
_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def migrate(
    source: MigrationSource | str,
    target: MigrationTarget | str,
    source_config: Any,
    target_config: Any,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """Migrate vectors + auxiliary state between supported backends."""
    source_enum = _coerce_source(source)
    target_enum = _coerce_target(target)

    if source_enum == MigrationSource.V07_LEGACY:
        target_provider = _provider_from_target(target_enum, target_config)
        with ExitStack() as stack:
            target_conn = stack.enter_context(_connection_context(target_provider))
            if target_conn is None:
                raise RuntimeError("Migration target does not expose a writable DB connection.")
            _ensure_aux_tables(target_conn)
            source_dir = _resolve_v07_source_dir(source_config)
            imported_counts = _migrate_v07_legacy(
                source_dir=source_dir,
                target_conn=target_conn,
                progress_callback=progress_callback,
            )
            _record_derived_schema_version(
                target_conn=target_conn,
                runtime_config=target_config,
                inferred_embedding_dim=None,
            )
            vector_key = _vector_validation_key(target_provider, target_conn)
            expected_counts = {vector_key: _count_vector_rows(target_conn, vector_key)}
            expected_counts.update(imported_counts)
            if _table_exists(target_conn, "lsm_schema_versions"):
                expected_counts["lsm_schema_versions"] = _count_table_rows(
                    target_conn, "lsm_schema_versions"
                )
            _record_validation_counts(target_conn, expected_counts)
            validation_result = validate_migration(target_conn)

        return {
            "source": source_enum.value,
            "target": target_enum.value,
            "total_vectors": expected_counts[vector_key],
            "migrated_vectors": expected_counts[vector_key],
            "validated_tables": int(validation_result.get("checked", 0)),
            "legacy_source_dir": str(source_dir),
        }

    source_provider = _provider_from_source(source_enum, source_config)
    target_provider = _provider_from_target(target_enum, target_config)

    _emit_progress(progress_callback, "migrate", 0, 0, "Copying vectors and chunk metadata.")
    migrated, total, inferred_dim = _copy_vectors(
        source_provider=source_provider,
        target_provider=target_provider,
        batch_size=max(1, int(batch_size)),
        progress_callback=progress_callback,
    )

    validation_checked = 0
    with ExitStack() as stack:
        source_conn = stack.enter_context(_connection_context(source_provider))
        target_conn = stack.enter_context(_connection_context(target_provider))

        if target_conn is not None:
            _ensure_aux_tables(target_conn)

        vector_key = (
            _vector_validation_key(target_provider, target_conn)
            if target_conn is not None
            else "vector_rows"
        )
        expected_counts: dict[str, int] = {vector_key: int(total)}
        if source_conn is not None and target_conn is not None:
            expected_counts.update(
                _copy_auxiliary_state(
                    source_conn=source_conn,
                    target_conn=target_conn,
                    progress_callback=progress_callback,
                )
            )
            _record_derived_schema_version(
                target_conn=target_conn,
                runtime_config=target_config,
                inferred_embedding_dim=inferred_dim,
            )
            if _table_exists(target_conn, "lsm_schema_versions"):
                expected_counts["lsm_schema_versions"] = _count_table_rows(
                    target_conn, "lsm_schema_versions"
                )
            _record_validation_counts(target_conn, expected_counts)
            validation_result = validate_migration(target_conn)
            validation_checked = int(validation_result.get("checked", 0))

    _emit_progress(progress_callback, "migrate", migrated, total, "Migration complete.")
    return {
        "source": source_enum.value,
        "target": target_enum.value,
        "total_vectors": int(total),
        "migrated_vectors": int(migrated),
        "validated_tables": validation_checked,
    }


def validate_migration(target_conn: Any) -> Dict[str, Any]:
    """Validate migration counts previously recorded on the target connection."""
    _ensure_aux_tables(target_conn)
    _ensure_validation_table(target_conn)
    rows = _fetch_table_rows(target_conn, _VALIDATION_TABLE)
    if not rows:
        return {"checked": 0, "mismatches": {}}

    mismatches: dict[str, dict[str, int]] = {}
    for row in rows:
        table_name = str(row.get("table_name") or "").strip()
        if not table_name:
            continue
        expected = int(row.get("expected_count") or 0)
        if table_name.startswith("vector_rows"):
            actual = _count_vector_rows(target_conn, table_name)
        else:
            actual = _count_table_rows(target_conn, table_name)
        if actual != expected:
            mismatches[table_name] = {"expected": expected, "actual": actual}

    if mismatches:
        details = ", ".join(
            f"{table} expected={values['expected']} actual={values['actual']}"
            for table, values in mismatches.items()
        )
        raise MigrationValidationError(f"Migration validation failed: {details}")

    return {"checked": len(rows), "mismatches": {}}


def _copy_vectors(
    *,
    source_provider: Any,
    target_provider: Any,
    batch_size: int,
    progress_callback: Optional[ProgressCallback],
) -> tuple[int, int, Optional[int]]:
    total = int(source_provider.count())
    migrated = 0
    offset = 0
    inferred_dim: Optional[int] = None
    last_page_ids: Optional[tuple[str, ...]] = None

    while True:
        page = source_provider.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = list(page.ids or [])
        if not ids:
            break

        page_ids = tuple(str(value) for value in ids)
        if last_page_ids is not None and page_ids == last_page_ids:
            raise RuntimeError(
                "Source provider returned identical pagination page twice; "
                "aborting to prevent duplicate migration loop."
            )
        last_page_ids = page_ids

        documents = list(page.documents or [])
        metadatas = list(page.metadatas or [])
        embeddings = _normalize_embeddings(page.embeddings)
        if inferred_dim is None and embeddings:
            inferred_dim = len(embeddings[0])

        target_provider.add_chunks(ids, documents, metadatas, embeddings)
        migrated += len(ids)
        offset += len(ids)
        _emit_progress(
            progress_callback,
            "vectors",
            migrated,
            total,
            f"Migrated {migrated}/{total} vectors.",
        )

    return migrated, total, inferred_dim


def _copy_auxiliary_state(
    *,
    source_conn: Any,
    target_conn: Any,
    progress_callback: Optional[ProgressCallback],
) -> Dict[str, int]:
    counts: dict[str, int] = {}
    _ensure_aux_tables(target_conn)

    for spec in _AUX_TABLE_SPECS:
        if not _table_exists(source_conn, spec.name):
            continue
        rows = _fetch_table_rows(source_conn, spec.name)
        counts[spec.name] = len(rows)
        if rows:
            _upsert_rows(target_conn, spec.name, spec.primary_key, rows)
        _emit_progress(
            progress_callback,
            "state",
            len(rows),
            len(rows),
            f"Copied auxiliary table '{spec.name}' ({len(rows)} rows).",
        )

    return counts


def _record_derived_schema_version(
    *,
    target_conn: Any,
    runtime_config: Any,
    inferred_embedding_dim: Optional[int],
) -> None:
    _ensure_aux_tables(target_conn)
    payload = _derive_schema_payload(runtime_config, inferred_embedding_dim)

    rows = _fetch_query_rows(
        target_conn,
        """
        SELECT
            lsm_version,
            embedding_model,
            embedding_dim,
            chunking_strategy,
            chunk_size,
            chunk_overlap
        FROM lsm_schema_versions
        ORDER BY id DESC
        LIMIT 1
        """,
    )
    if rows:
        latest = rows[0]
        if (
            str(latest.get("lsm_version") or "") == str(payload["lsm_version"])
            and str(latest.get("embedding_model") or "") == str(payload["embedding_model"])
            and int(latest.get("embedding_dim") or 0) == int(payload["embedding_dim"])
            and str(latest.get("chunking_strategy") or "") == str(payload["chunking_strategy"])
            and int(latest.get("chunk_size") or 0) == int(payload["chunk_size"])
            and int(latest.get("chunk_overlap") or 0) == int(payload["chunk_overlap"])
        ):
            return

    now = _utcnow_iso()
    rows_to_insert = [
        {
            "id": _next_schema_version_id(target_conn),
            "manifest_version": None,
            "lsm_version": payload["lsm_version"],
            "embedding_model": payload["embedding_model"],
            "embedding_dim": payload["embedding_dim"],
            "chunking_strategy": payload["chunking_strategy"],
            "chunk_size": payload["chunk_size"],
            "chunk_overlap": payload["chunk_overlap"],
            "created_at": now,
            "last_ingest_at": now,
        }
    ]
    _upsert_rows(target_conn, "lsm_schema_versions", "id", rows_to_insert)


def _record_validation_counts(target_conn: Any, counts: Mapping[str, int]) -> None:
    _ensure_validation_table(target_conn)
    now = _utcnow_iso()
    rows = [
        {
            "table_name": table_name,
            "expected_count": int(expected),
            "recorded_at": now,
        }
        for table_name, expected in counts.items()
    ]
    _upsert_rows(target_conn, _VALIDATION_TABLE, "table_name", rows)


def _derive_schema_payload(runtime_config: Any, inferred_embedding_dim: Optional[int]) -> Dict[str, Any]:
    embed_model = _runtime_value(
        runtime_config,
        "embedding_model",
        "embed_model",
        default="",
    )
    chunking_strategy = _runtime_value(runtime_config, "chunking_strategy", default="")
    chunk_size = _runtime_value(runtime_config, "chunk_size", default=0)
    chunk_overlap = _runtime_value(runtime_config, "chunk_overlap", default=0)
    embed_dim = inferred_embedding_dim
    if embed_dim is None:
        embed_dim = _runtime_value(runtime_config, "embedding_dim", "embedding_dimension", default=0)

    return {
        "lsm_version": str(_runtime_value(runtime_config, "lsm_version", default=LSM_VERSION)),
        "embedding_model": str(embed_model or ""),
        "embedding_dim": int(embed_dim or 0),
        "chunking_strategy": str(chunking_strategy or ""),
        "chunk_size": int(chunk_size or 0),
        "chunk_overlap": int(chunk_overlap or 0),
    }


def _runtime_value(runtime_config: Any, *names: str, default: Any = None) -> Any:
    if isinstance(runtime_config, Mapping):
        for name in names:
            if name in runtime_config:
                return runtime_config[name]
        global_cfg = runtime_config.get("global")
        ingest_cfg = runtime_config.get("ingest")
        for name in names:
            if isinstance(global_cfg, Mapping) and name in global_cfg:
                return global_cfg[name]
            if isinstance(ingest_cfg, Mapping) and name in ingest_cfg:
                return ingest_cfg[name]
        return default

    for name in names:
        value = getattr(runtime_config, name, None)
        if value is not None:
            return value

    global_settings = getattr(runtime_config, "global_settings", None)
    ingest_settings = getattr(runtime_config, "ingest", None)
    if global_settings is not None:
        for name in names:
            value = getattr(global_settings, name, None)
            if value is not None:
                return value
    if ingest_settings is not None:
        for name in names:
            value = getattr(ingest_settings, name, None)
            if value is not None:
                return value
    return default


def _normalize_embeddings(embeddings: Any) -> list[list[float]]:
    if embeddings is None:
        return []
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    normalized: list[list[float]] = []
    for row in embeddings:
        if hasattr(row, "tolist"):
            row = row.tolist()
        normalized.append([float(value) for value in row])
    return normalized


def _coerce_source(value: MigrationSource | str) -> MigrationSource:
    if isinstance(value, MigrationSource):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"chroma", "chromadb"}:
        return MigrationSource.CHROMA
    if normalized in {"sqlite", "sqlite-vec", "sqlite_vec"}:
        return MigrationSource.SQLITE
    if normalized in {"postgres", "postgresql"}:
        return MigrationSource.POSTGRESQL
    if normalized in {"v0.7", "v07", "legacy"}:
        return MigrationSource.V07_LEGACY
    raise ValueError(f"Unsupported migration source: {value!r}")


def _coerce_target(value: MigrationTarget | str) -> MigrationTarget:
    if isinstance(value, MigrationTarget):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"sqlite", "sqlite-vec", "sqlite_vec", "v0.8", "v08"}:
        return MigrationTarget.SQLITE
    if normalized in {"postgres", "postgresql"}:
        return MigrationTarget.POSTGRESQL
    raise ValueError(f"Unsupported migration target: {value!r}")


def _provider_from_source(source: MigrationSource, source_config: Any) -> Any:
    if source == MigrationSource.CHROMA:
        config = _to_vectordb_config(source_config, provider_hint="chromadb")
        return ChromaDBProvider(config)
    config = _to_vectordb_config(
        source_config,
        provider_hint="sqlite" if source == MigrationSource.SQLITE else "postgresql",
    )
    return create_vectordb_provider(config)


def _provider_from_target(target: MigrationTarget, target_config: Any) -> Any:
    config = _to_vectordb_config(
        target_config,
        provider_hint="sqlite" if target == MigrationTarget.SQLITE else "postgresql",
    )
    return create_vectordb_provider(config)


def _to_vectordb_config(raw: Any, provider_hint: Optional[str] = None) -> VectorDBConfig:
    if isinstance(raw, VectorDBConfig):
        return replace(raw, provider=provider_hint or raw.provider)

    if hasattr(raw, "vectordb") and isinstance(raw.vectordb, VectorDBConfig):
        base = raw.vectordb
        return replace(base, provider=provider_hint or base.provider)

    if isinstance(raw, Mapping):
        vectordb_raw: Mapping[str, Any] = raw
        if "vectordb" in raw and isinstance(raw["vectordb"], Mapping):
            vectordb_raw = raw["vectordb"]
        provider = provider_hint or str(vectordb_raw.get("provider", "sqlite"))
        path_value = vectordb_raw.get("path", Path(".lsm"))
        collection = str(vectordb_raw.get("collection", "local_kb"))
        return VectorDBConfig(
            provider=provider,
            collection=collection,
            path=Path(path_value),
            connection_string=vectordb_raw.get("connection_string"),
            host=vectordb_raw.get("host"),
            port=vectordb_raw.get("port"),
            database=vectordb_raw.get("database"),
            user=vectordb_raw.get("user"),
            password=vectordb_raw.get("password"),
            index_type=vectordb_raw.get("index_type", "hnsw"),
            pool_size=int(vectordb_raw.get("pool_size", 5)),
        )

    raise ValueError("Unable to derive VectorDBConfig for migration source/target.")


def _emit_progress(
    callback: Optional[ProgressCallback],
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, int(current), int(total), message)
        return
    except TypeError:
        pass
    try:
        callback(str(message), int(current), int(total))  # type: ignore[misc]
        return
    except TypeError:
        pass
    callback(stage, int(current), int(total), message)  # re-raise consistent error


@contextmanager
def _connection_context(provider: Any):
    if provider is None:
        yield None
        return

    connection = getattr(provider, "connection", None)
    if connection is not None:
        yield connection
        return

    get_conn = getattr(provider, "_get_conn", None)
    if callable(get_conn):
        with get_conn() as conn:
            yield conn
        return

    yield None


def _ensure_aux_tables(conn: Any) -> None:
    dialect = _dialect(conn)
    for spec in _AUX_TABLE_SPECS:
        ddl = spec.sqlite_ddl if dialect == "sqlite" else spec.postgres_ddl
        _execute(conn, ddl)
    _ensure_validation_table(conn)
    _commit(conn)


def _ensure_validation_table(conn: Any) -> None:
    dialect = _dialect(conn)
    if dialect == "sqlite":
        _execute(
            conn,
            f"""
            CREATE TABLE IF NOT EXISTS {_VALIDATION_TABLE} (
                table_name TEXT PRIMARY KEY,
                expected_count INTEGER NOT NULL,
                recorded_at TEXT NOT NULL
            )
            """,
        )
    else:
        _execute(
            conn,
            f"""
            CREATE TABLE IF NOT EXISTS {_VALIDATION_TABLE} (
                table_name TEXT PRIMARY KEY,
                expected_count BIGINT NOT NULL,
                recorded_at TEXT NOT NULL
            )
            """,
        )
    _commit(conn)


def _table_exists(conn: Any, table_name: str) -> bool:
    safe_table = _safe_ident(table_name)
    if _dialect(conn) == "sqlite":
        row = _execute(
            conn,
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (safe_table,),
        ).fetchone()
        return row is not None

    row = _execute(conn, "SELECT to_regclass(%s)", (safe_table,)).fetchone()
    return bool(row and row[0])


def _fetch_table_rows(conn: Any, table_name: str) -> list[dict[str, Any]]:
    safe_table = _safe_ident(table_name)
    if not _table_exists(conn, safe_table):
        return []
    cursor = _execute(conn, f"SELECT * FROM {safe_table}")
    rows = cursor.fetchall()
    columns = [item[0] for item in (cursor.description or [])]
    output: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            output.append({column: row[column] for column in columns})
        elif isinstance(row, Mapping):
            output.append(dict(row))
        else:
            output.append(dict(zip(columns, row)))
    return output


def _fetch_query_rows(conn: Any, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    cursor = _execute(conn, query, tuple(params))
    rows = cursor.fetchall()
    columns = [item[0] for item in (cursor.description or [])]
    output: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            output.append({column: row[column] for column in columns})
        elif isinstance(row, Mapping):
            output.append(dict(row))
        else:
            output.append(dict(zip(columns, row)))
    return output


def _upsert_rows(conn: Any, table_name: str, primary_key: str, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return

    safe_table = _safe_ident(table_name)
    safe_pk = _safe_ident(primary_key)
    columns = list(rows_list[0].keys())
    for key in columns:
        _safe_ident(key)
    update_columns = [column for column in columns if column != safe_pk]

    dialect = _dialect(conn)
    if dialect == "sqlite":
        placeholders = ", ".join(["?"] * len(columns))
        assignments = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
        conflict_sql = (
            f"ON CONFLICT({safe_pk}) DO UPDATE SET {assignments}"
            if assignments
            else f"ON CONFLICT({safe_pk}) DO NOTHING"
        )
        sql_text = (
            f"INSERT INTO {safe_table} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) {conflict_sql}"
        )
    else:
        placeholders = ", ".join(["%s"] * len(columns))
        assignments = ", ".join(f"{column}=EXCLUDED.{column}" for column in update_columns)
        conflict_sql = (
            f"ON CONFLICT ({safe_pk}) DO UPDATE SET {assignments}"
            if assignments
            else f"ON CONFLICT ({safe_pk}) DO NOTHING"
        )
        sql_text = (
            f"INSERT INTO {safe_table} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) {conflict_sql}"
        )

    for row in rows_list:
        values = [row.get(column) for column in columns]
        _execute(conn, sql_text, tuple(values))
    _commit(conn)


def _count_table_rows(conn: Any, table_name: str) -> int:
    safe_table = _safe_ident(table_name)
    if not _table_exists(conn, safe_table):
        return 0
    row = _execute(conn, f"SELECT COUNT(*) FROM {safe_table}").fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _count_vector_rows_for_key(conn: Any, key: Optional[str]) -> int:
    preferred_table: Optional[str] = None
    if key and ":" in key:
        _prefix, table_name = key.split(":", 1)
        table_name = str(table_name or "").strip()
        if table_name:
            preferred_table = table_name
    if preferred_table:
        if _table_exists(conn, preferred_table):
            return _count_table_rows(conn, preferred_table)
        return 0

    if _table_exists(conn, "lsm_chunks"):
        row = _execute(conn, "SELECT COUNT(*) FROM lsm_chunks").fetchone()
        return int(row[0] or 0) if row is not None else 0

    if _dialect(conn) == "postgresql":
        rows = _fetch_query_rows(
            conn,
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = current_schema()
              AND tablename LIKE 'chunks_%'
            ORDER BY tablename
            LIMIT 1
            """,
        )
        if rows:
            table_name = str(rows[0].get("tablename") or "").strip()
            if table_name:
                return _count_table_rows(conn, table_name)
    return 0


def _count_vector_rows(conn: Any, key: Optional[str] = None) -> int:
    return _count_vector_rows_for_key(conn, key)


def _vector_validation_key(target_provider: Any, target_conn: Any) -> str:
    if _table_exists(target_conn, "lsm_chunks"):
        return "vector_rows:lsm_chunks"

    table_name = getattr(target_provider, "_table_name", None)
    if isinstance(table_name, str):
        candidate = table_name.strip()
        if candidate:
            try:
                _safe_ident(candidate)
            except ValueError:
                return "vector_rows"
            return f"vector_rows:{candidate}"

    return "vector_rows"


def _next_schema_version_id(conn: Any) -> int:
    row = _execute(conn, "SELECT COALESCE(MAX(id), 0) FROM lsm_schema_versions").fetchone()
    current = int(row[0] or 0) if row is not None else 0
    return current + 1


def _dialect(conn: Any) -> str:
    return "sqlite" if isinstance(conn, sqlite3.Connection) else "postgresql"


def _commit(conn: Any) -> None:
    commit = getattr(conn, "commit", None)
    if callable(commit):
        commit()


def _execute(conn: Any, query: str, params: Iterable[Any] = ()) -> Any:
    execute = getattr(conn, "execute", None)
    if callable(execute):
        return execute(query, tuple(params))
    cursor = conn.cursor()
    cursor.execute(query, tuple(params))
    return cursor


def _safe_ident(value: str) -> str:
    candidate = str(value).strip()
    if not _VALID_IDENTIFIER.match(candidate):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return candidate


def _resolve_v07_source_dir(source_config: Any) -> Path:
    candidate: Optional[Any] = None
    if isinstance(source_config, (str, Path)):
        candidate = source_config
    elif isinstance(source_config, Mapping):
        candidate = source_config.get("source_dir") or source_config.get("path")
    else:
        candidate = getattr(source_config, "source_dir", None)
        if candidate is None:
            candidate = getattr(source_config, "global_folder", None)
        if candidate is None:
            global_settings = getattr(source_config, "global_settings", None)
            if global_settings is not None:
                candidate = getattr(global_settings, "global_folder", None)

    if candidate is None:
        raise ValueError(
            "Legacy migration requires a source directory. "
            "Pass --source-dir <path> with manifest.json/memories.db/schedules.json."
        )

    resolved = Path(candidate).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Legacy source directory not found: {resolved}")
    return resolved


def _migrate_v07_legacy(
    *,
    source_dir: Path,
    target_conn: Any,
    progress_callback: Optional[ProgressCallback],
) -> Dict[str, int]:
    _ensure_aux_tables(target_conn)
    counts: dict[str, int] = {
        "lsm_manifest": 0,
        "lsm_agent_memories": 0,
        "lsm_agent_memory_candidates": 0,
        "lsm_agent_schedules": 0,
        "lsm_stats_cache": 0,
        "lsm_remote_cache": 0,
    }

    _emit_progress(
        progress_callback,
        "legacy",
        0,
        0,
        f"Importing legacy v0.7 state from {source_dir}.",
    )

    manifest_path = source_dir / "manifest.json"
    if manifest_path.exists():
        raw_manifest = _load_json_file(manifest_path)
        rows: list[dict[str, Any]] = []
        if isinstance(raw_manifest, Mapping):
            for source_path, value in raw_manifest.items():
                if not isinstance(value, Mapping):
                    continue
                rows.append(
                    {
                        "source_path": str(source_path),
                        "mtime_ns": value.get("mtime_ns"),
                        "file_size": value.get("size", value.get("file_size")),
                        "file_hash": value.get("file_hash"),
                        "version": int(value.get("version", 1) or 1),
                        "embedding_model": value.get("embedding_model"),
                        "schema_version_id": value.get("schema_version_id"),
                        "updated_at": value.get("updated_at") or _utcnow_iso(),
                    }
                )
        if rows:
            _upsert_rows(target_conn, "lsm_manifest", "source_path", rows)
        counts["lsm_manifest"] = len(rows)
    else:
        _warn_legacy_missing(manifest_path)

    memories_db_path = source_dir / "memories.db"
    if memories_db_path.exists():
        legacy_conn = sqlite3.connect(str(memories_db_path))
        legacy_conn.row_factory = sqlite3.Row
        try:
            memory_table = _first_existing_table(
                legacy_conn,
                ("lsm_agent_memories", "memories"),
            )
            candidate_table = _first_existing_table(
                legacy_conn,
                ("lsm_agent_memory_candidates", "memory_candidates"),
            )

            memory_rows: list[dict[str, Any]] = []
            if memory_table is not None:
                for row in _fetch_table_rows(legacy_conn, memory_table):
                    memory_rows.append(
                        {
                            "id": row.get("id"),
                            "memory_type": row.get("memory_type", row.get("type", "project_fact")),
                            "memory_key": row.get("memory_key", row.get("key", "")),
                            "value_json": _json_text(
                                row.get("value_json", row.get("value", {})),
                                default="{}",
                            ),
                            "scope": row.get("scope", "project"),
                            "tags_json": _json_text(
                                row.get("tags_json", row.get("tags", [])),
                                default="[]",
                            ),
                            "confidence": float(row.get("confidence", 1.0) or 1.0),
                            "created_at": row.get("created_at") or _utcnow_iso(),
                            "last_used_at": row.get("last_used_at") or row.get("created_at") or _utcnow_iso(),
                            "expires_at": row.get("expires_at"),
                            "source_run_id": row.get("source_run_id", row.get("run_id", "")),
                        }
                    )
                if memory_rows:
                    _upsert_rows(target_conn, "lsm_agent_memories", "id", memory_rows)
            counts["lsm_agent_memories"] = len(memory_rows)

            candidate_rows: list[dict[str, Any]] = []
            if candidate_table is not None:
                for row in _fetch_table_rows(legacy_conn, candidate_table):
                    candidate_rows.append(
                        {
                            "id": row.get("id"),
                            "memory_id": row.get("memory_id"),
                            "provenance": row.get("provenance", ""),
                            "rationale": row.get("rationale", ""),
                            "status": row.get("status", "pending"),
                            "created_at": row.get("created_at") or _utcnow_iso(),
                            "updated_at": row.get("updated_at") or row.get("created_at") or _utcnow_iso(),
                        }
                    )
                if candidate_rows:
                    _upsert_rows(
                        target_conn,
                        "lsm_agent_memory_candidates",
                        "id",
                        candidate_rows,
                    )
            counts["lsm_agent_memory_candidates"] = len(candidate_rows)
        finally:
            legacy_conn.close()
    else:
        _warn_legacy_missing(memories_db_path)

    schedules_path = source_dir / "schedules.json"
    if schedules_path.exists():
        raw_schedules = _load_json_file(schedules_path)
        schedule_rows: list[dict[str, Any]] = []
        for schedule in _iter_legacy_schedules(raw_schedules):
            schedule_id = str(
                schedule.get("schedule_id") or schedule.get("id") or ""
            ).strip()
            if not schedule_id:
                agent_name = str(schedule.get("agent_name") or "agent").strip()
                schedule_id = f"{agent_name}:{_legacy_hash(json.dumps(schedule, sort_keys=True))}"
            schedule_rows.append(
                {
                    "schedule_id": schedule_id,
                    "agent_name": str(schedule.get("agent_name") or "unknown"),
                    "last_run_at": schedule.get("last_run_at"),
                    "next_run_at": schedule.get("next_run_at") or _utcnow_iso(),
                    "last_status": str(schedule.get("last_status") or "idle"),
                    "last_error": schedule.get("last_error"),
                    "queued_runs": int(schedule.get("queued_runs", 0) or 0),
                    "updated_at": schedule.get("updated_at") or _utcnow_iso(),
                }
            )
        if schedule_rows:
            _upsert_rows(target_conn, "lsm_agent_schedules", "schedule_id", schedule_rows)
        counts["lsm_agent_schedules"] = len(schedule_rows)
    else:
        _warn_legacy_missing(schedules_path)

    stats_cache_path = source_dir / "stats_cache.json"
    if stats_cache_path.exists():
        raw_stats = _load_json_file(stats_cache_path)
        stats_rows: list[dict[str, Any]] = []
        if isinstance(raw_stats, Mapping) and "stats" in raw_stats:
            stats_rows.append(
                {
                    "cache_key": "collection_stats",
                    "cached_at": float(raw_stats.get("cached_at", 0.0) or 0.0),
                    "chunk_count": int(raw_stats.get("chunk_count", 0) or 0),
                    "stats_json": _json_text(raw_stats.get("stats", {}), default="{}"),
                }
            )
        elif isinstance(raw_stats, Mapping):
            for key, value in raw_stats.items():
                if not isinstance(value, Mapping):
                    continue
                stats_rows.append(
                    {
                        "cache_key": str(key),
                        "cached_at": float(value.get("cached_at", 0.0) or 0.0),
                        "chunk_count": int(value.get("chunk_count", 0) or 0),
                        "stats_json": _json_text(value.get("stats", {}), default="{}"),
                    }
                )
        if stats_rows:
            _upsert_rows(target_conn, "lsm_stats_cache", "cache_key", stats_rows)
        counts["lsm_stats_cache"] = len(stats_rows)
    else:
        _warn_legacy_missing(stats_cache_path)

    remote_rows: list[dict[str, Any]] = []
    for root in (source_dir / "Downloads", source_dir / "remote"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            payload = _load_json_file(path)
            if not isinstance(payload, Mapping):
                continue
            provider = _legacy_provider_name(payload, path)
            cache_key = _legacy_remote_cache_key(provider=provider, payload=payload, path=path, base_dir=source_dir)
            created_at = _legacy_created_at(payload, path)
            expires_at = payload.get("expires_at")
            remote_rows.append(
                {
                    "cache_key": cache_key,
                    "provider": provider,
                    "response_json": _json_text(payload, default="{}"),
                    "created_at": created_at,
                    "expires_at": str(expires_at) if expires_at is not None else None,
                }
            )
    if remote_rows:
        _upsert_rows(target_conn, "lsm_remote_cache", "cache_key", remote_rows)
    counts["lsm_remote_cache"] = len(remote_rows)

    _emit_progress(
        progress_callback,
        "legacy",
        1,
        1,
        (
            "Legacy import complete: "
            f"manifest={counts['lsm_manifest']}, memories={counts['lsm_agent_memories']}, "
            f"candidates={counts['lsm_agent_memory_candidates']}, schedules={counts['lsm_agent_schedules']}, "
            f"stats={counts['lsm_stats_cache']}, remote={counts['lsm_remote_cache']}."
        ),
    )
    return counts


def _first_existing_table(conn: Any, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if _table_exists(conn, name):
            return name
    return None


def _iter_legacy_schedules(raw: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                yield item
        return

    if isinstance(raw, Mapping):
        schedules = raw.get("schedules")
        if isinstance(schedules, list):
            for item in schedules:
                if isinstance(item, Mapping):
                    yield item
            return
        for key, value in raw.items():
            if isinstance(value, Mapping):
                payload = dict(value)
                payload.setdefault("schedule_id", str(key))
                yield payload


def _legacy_provider_name(payload: Mapping[str, Any], path: Path) -> str:
    provider = str(payload.get("provider") or path.parent.name or "unknown_provider")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", provider.strip())
    return cleaned or "unknown_provider"


def _legacy_remote_cache_key(
    *,
    provider: str,
    payload: Mapping[str, Any],
    path: Path,
    base_dir: Path,
) -> str:
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        return f"query:{provider}:{_legacy_hash(query)}"
    feed_url = payload.get("feed_url")
    if isinstance(feed_url, str) and feed_url.strip():
        return f"feed:rss:{_legacy_hash(feed_url)}"
    rel = str(path.relative_to(base_dir))
    return f"legacy:{_legacy_hash(rel)}"


def _legacy_created_at(payload: Mapping[str, Any], path: Path) -> str:
    saved_at = payload.get("saved_at")
    if isinstance(saved_at, (int, float)):
        return datetime.fromtimestamp(float(saved_at), tz=timezone.utc).isoformat()
    created_at = payload.get("created_at")
    if isinstance(created_at, str) and created_at.strip():
        return created_at
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _legacy_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _json_text(value: Any, *, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return default


def _warn_legacy_missing(path: Path) -> None:
    logger.warning("Legacy migration skipped missing file: %s", path)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
