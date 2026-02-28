"""Database migration framework for cross-backend state transfer."""

from __future__ import annotations

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
from lsm.vectordb import create_vectordb_provider
from lsm.vectordb.chromadb import ChromaDBProvider

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
        raise NotImplementedError(
            "Legacy v0.7 migration path is not available in the framework yet."
        )

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

        expected_counts: dict[str, int] = {"vector_rows": int(total)}
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
        if table_name == "vector_rows":
            actual = _count_vector_rows(target_conn)
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


def _count_vector_rows(conn: Any) -> int:
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
