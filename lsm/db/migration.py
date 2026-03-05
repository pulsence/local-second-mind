"""Database migration framework for cross-backend state transfer.

Migration Matrix
----------------
The following tables are handled during cross-backend migration:

**Migrated as data (copied row-by-row)**:
- ``lsm_schema_versions`` — schema version tracking
- ``lsm_manifest`` — ingest manifest
- ``lsm_agent_memories`` — agent long-term memory
- ``lsm_agent_memory_candidates`` — agent memory candidates
- ``lsm_agent_schedules`` — agent schedule persistence
- ``lsm_stats_cache`` — collection stats cache
- ``lsm_remote_cache`` — remote provider cache
- Vector data (``chunks`` + embeddings) — via provider API

**Rebuilt post-migration**:
- ``lsm_reranker_cache`` — cross-encoder score cache (ephemeral, rebuilt on demand)
- ``lsm_cluster_centroids`` — rebuilt via ``lsm cluster build``
- ``lsm_graph_nodes`` / ``lsm_graph_edges`` — rebuilt via ``lsm graph build-links``
- ``lsm_embedding_models`` — registry entries copied, models reloaded from path
- ``lsm_job_status`` — job tracking reset for new backend
- ``chunks_fts`` (SQLite) — FTS5 virtual table rebuilt from chunks after PG→SQLite
- ``tsvector`` column (PostgreSQL) — generated column created after SQLite→PG

**Intentionally excluded**:
- ``lsm_migration_progress`` — target-local operational table, recreated on target
- ``lsm_migration_validation`` — target-local operational table, recreated on target
- ``vec_chunks`` — SQLite-only virtual table, recreated by provider init

**Embedding format conversion**:
- SQLite vec0 stores embeddings as binary BLOBs (packed floats via ``struct.pack``)
- PostgreSQL pgvector uses the ``vector`` type (Python list of floats)
- Migration converts between formats via the provider API: ``source.get()`` returns
  Python float lists, ``target.add_chunks()`` converts to target format
- This is transparent and does not require explicit conversion code
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from lsm import __version__ as LSM_VERSION
from lsm.config.models import DBConfig
from lsm.db.compat import commit as compat_commit
from lsm.db.compat import count_rows as compat_count_rows
from lsm.db.compat import dialect as compat_dialect
from lsm.db.compat import execute as compat_execute
from lsm.db.compat import fetch_rows_as_dicts as compat_fetch_rows_as_dicts
from lsm.db.compat import safe_identifier
from lsm.db.compat import table_exists as compat_table_exists
from lsm.db.compat import upsert_rows as compat_upsert_rows
from lsm.db.connection import resolve_connection
from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames
from lsm.logging import get_logger
from lsm.vectordb import create_vectordb_provider

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


def _build_aux_table_specs(tn: TableNames) -> tuple[AuxiliaryTableSpec, ...]:
    """Build auxiliary table specs using resolved table names."""
    return (
        AuxiliaryTableSpec(
            name=tn.schema_versions,
            primary_key="id",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.schema_versions} (
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
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.schema_versions} (
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
            name=tn.manifest,
            primary_key="source_path",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.manifest} (
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
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.manifest} (
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
            name=tn.agent_memories,
            primary_key="id",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.agent_memories} (
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
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.agent_memories} (
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
            name=tn.agent_memory_candidates,
            primary_key="id",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.agent_memory_candidates} (
                    id            TEXT PRIMARY KEY,
                    memory_id     TEXT NOT NULL UNIQUE,
                    provenance    TEXT NOT NULL,
                    rationale     TEXT NOT NULL,
                    status        TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
            """,
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.agent_memory_candidates} (
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
            name=tn.agent_schedules,
            primary_key="schedule_id",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.agent_schedules} (
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
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.agent_schedules} (
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
            name=tn.stats_cache,
            primary_key="cache_key",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.stats_cache} (
                    cache_key      TEXT PRIMARY KEY,
                    cached_at      REAL NOT NULL,
                    chunk_count    INTEGER NOT NULL,
                    stats_json     TEXT NOT NULL
                )
            """,
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.stats_cache} (
                    cache_key      TEXT PRIMARY KEY,
                    cached_at      DOUBLE PRECISION NOT NULL,
                    chunk_count    BIGINT NOT NULL,
                    stats_json     TEXT NOT NULL
                )
            """,
        ),
        AuxiliaryTableSpec(
            name=tn.remote_cache,
            primary_key="cache_key",
            sqlite_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.remote_cache} (
                    cache_key      TEXT PRIMARY KEY,
                    provider       TEXT NOT NULL,
                    response_json  TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    expires_at     TEXT
                )
            """,
            postgres_ddl=f"""
                CREATE TABLE IF NOT EXISTS {tn.remote_cache} (
                    cache_key      TEXT PRIMARY KEY,
                    provider       TEXT NOT NULL,
                    response_json  TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    expires_at     TEXT
                )
            """,
        ),
    )

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def migrate(
    source: MigrationSource | str,
    target: MigrationTarget | str,
    source_config: Any,
    target_config: Any,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    batch_size: int = 1000,
    table_names: TableNames | None = None,
    resume: bool = False,
    skip_enrich: bool = False,
) -> Dict[str, Any]:
    """Migrate vectors + auxiliary state between supported backends."""
    import uuid

    tn = table_names or DEFAULT_TABLE_NAMES
    source_enum = _coerce_source(source)
    target_enum = _coerce_target(target)
    run_id = str(uuid.uuid4())

    if source_enum == MigrationSource.V07_LEGACY:
        target_provider = _provider_from_target(target_enum, target_config)
        enrichment_report: Optional[Any] = None
        with ExitStack() as stack:
            target_conn = stack.enter_context(_connection_context(target_provider))
            if target_conn is None:
                raise RuntimeError("Migration target does not expose a writable DB connection.")
            _ensure_aux_tables(target_conn, tn)
            source_dir = _resolve_v07_source_dir(source_config)
            imported_counts = _migrate_v07_legacy(
                source_dir=source_dir,
                target_conn=target_conn,
                progress_callback=progress_callback,
                tn=tn,
            )
            _record_derived_schema_version(
                target_conn=target_conn,
                runtime_config=target_config,
                inferred_embedding_dim=None,
                tn=tn,
            )

            # Post-import enrichment (best-effort; schema may lack enrichment columns)
            if not skip_enrich and isinstance(target_conn, sqlite3.Connection):
                try:
                    from lsm.db.enrichment import run_enrichment_pipeline

                    _emit_progress(progress_callback, "legacy", 0, 0, "Running post-migration enrichment.")
                    enrichment_report = run_enrichment_pipeline(
                        target_conn,
                        target_config,
                        table_names=tn,
                    )
                except Exception as exc:
                    logger.warning("Post-migration enrichment skipped: %s", exc)

            vector_key = _vector_validation_key(target_provider, target_conn, tn)
            expected_counts = {vector_key: _count_vector_rows(target_conn, vector_key, tn)}
            expected_counts.update(imported_counts)
            if compat_table_exists(target_conn, tn.schema_versions):
                expected_counts[tn.schema_versions] = compat_count_rows(
                    target_conn, tn.schema_versions
                )
            _record_validation_counts(target_conn, expected_counts, tn)
            validation_result = validate_migration(target_conn, tn)

        result: dict[str, Any] = {
            "source": source_enum.value,
            "target": target_enum.value,
            "total_vectors": expected_counts[vector_key],
            "migrated_vectors": expected_counts[vector_key],
            "validated_tables": int(validation_result.get("checked", 0)),
            "legacy_source_dir": str(source_dir),
            "imported_counts": imported_counts,
        }
        if enrichment_report is not None:
            result["enrichment"] = enrichment_report
        return result

    source_provider = _provider_from_source(source_enum, source_config)
    target_provider = _provider_from_target(target_enum, target_config)

    # Determine completed stages for resume.
    # Always check for a prior interrupted/in-progress run so we can
    # pick up where we left off instead of re-copying everything.
    completed_stages: set[str] = set()
    with _connection_context(target_provider) as tc:
        if tc is not None:
            _ensure_migration_progress_table(tc, tn)
            prev_run = _get_latest_run_id(tc, tn)
            if prev_run:
                has_incomplete = _has_incomplete_stages(tc, prev_run, tn)
                if resume or has_incomplete:
                    run_id = prev_run
                    completed_stages = _get_completed_stages(tc, prev_run, tn)

    validation_checked = 0
    total = int(source_provider.count())
    migrated = 0
    inferred_dim: Optional[int] = None
    enrichment_report: Optional[Any] = None

    with ExitStack() as stack:
        source_conn = stack.enter_context(_connection_context(source_provider))
        target_conn = stack.enter_context(_connection_context(target_provider))
        if target_conn is None:
            raise RuntimeError("Migration target does not expose a writable DB connection.")

        _ensure_aux_tables(target_conn, tn)
        _ensure_migration_progress_table(target_conn, tn)

        # Vector copy stage.
        _emit_progress(progress_callback, "migrate", 0, 0, "Copying vectors and chunk metadata.")
        vector_stage = "copy_vectors"
        if vector_stage in completed_stages:
            _emit_progress(progress_callback, "migrate", total, total, "Skipping copy_vectors (already completed).")
            migrated = total
        else:
            start_offset = _get_stage_rows_processed(target_conn, run_id, vector_stage, tn)
            if start_offset > 0:
                _emit_progress(
                    progress_callback, "migrate", start_offset, total,
                    f"Resuming from vector {start_offset:,}/{total:,}.",
                )
            stage_id = _begin_stage(
                target_conn,
                run_id,
                vector_stage,
                source_enum.value,
                target_enum.value,
                tn,
            )
            try:
                copied_count, total, inferred_dim = _copy_vectors(
                    source_provider=source_provider,
                    target_provider=target_provider,
                    batch_size=max(1, int(batch_size)),
                    progress_callback=progress_callback,
                    start_offset=start_offset,
                    rows_progress_callback=lambda rows: _set_stage_rows_processed(
                        target_conn,
                        stage_id,
                        rows,
                        tn=tn,
                    ),
                )
                migrated = int(start_offset + copied_count)
                _complete_stage(target_conn, stage_id, rows=migrated, tn=tn)
            except KeyboardInterrupt:
                _interrupt_stage(target_conn, stage_id, tn=tn)
                _emit_progress(
                    progress_callback, "migrate", 0, 0,
                    f"Interrupted. Progress saved. Run `lsm migrate --resume` to continue.",
                )
                return {
                    "source": source_enum.value,
                    "target": target_enum.value,
                    "total_vectors": total,
                    "migrated_vectors": migrated,
                    "validated_tables": 0,
                    "interrupted": True,
                }
            except Exception as exc:
                _fail_stage(target_conn, stage_id, str(exc), tn=tn)
                raise

        vector_key = _vector_validation_key(target_provider, target_conn, tn)
        expected_counts: dict[str, int] = {vector_key: int(total)}

        # Auxiliary copy stages (per-table granularity for resume).
        if source_conn is not None:
            for spec in _build_aux_table_specs(tn):
                stage_name = _aux_copy_stage_name(spec.name, tn)
                if stage_name in completed_stages:
                    _emit_progress(
                        progress_callback,
                        "migrate",
                        0,
                        0,
                        f"Skipping {stage_name} (already completed).",
                    )
                    expected_counts[spec.name] = compat_count_rows(target_conn, spec.name)
                    continue

                stage_id = _begin_stage(
                    target_conn,
                    run_id,
                    stage_name,
                    source_enum.value,
                    target_enum.value,
                    tn,
                )
                try:
                    copied = _copy_aux_table(
                        source_conn=source_conn,
                        target_conn=target_conn,
                        spec=spec,
                        progress_callback=progress_callback,
                    )
                    _complete_stage(target_conn, stage_id, rows=copied, tn=tn)
                    expected_counts[spec.name] = compat_count_rows(target_conn, spec.name)
                except Exception as exc:
                    _fail_stage(target_conn, stage_id, str(exc), tn=tn)
                    raise

        _record_derived_schema_version(
            target_conn=target_conn,
            runtime_config=target_config,
            inferred_embedding_dim=inferred_dim,
            tn=tn,
        )
        if compat_table_exists(target_conn, tn.schema_versions):
            expected_counts[tn.schema_versions] = compat_count_rows(target_conn, tn.schema_versions)

        # Schema evolution.
        evolve_stage = "schema_evolution"
        if evolve_stage not in completed_stages:
            stage_id = _begin_stage(
                target_conn,
                run_id,
                evolve_stage,
                source_enum.value,
                target_enum.value,
                tn,
            )
            try:
                _evolve_schema(target_conn, tn)
                _complete_stage(target_conn, stage_id, tn=tn)
            except Exception as exc:
                _fail_stage(target_conn, stage_id, str(exc), tn=tn)
                raise
        else:
            _emit_progress(progress_callback, "migrate", 0, 0, "Skipping schema_evolution (already completed).")

        # FTS rebuild after cross-backend migration.
        _rebuild_fts_if_needed(
            target_conn=target_conn,
            target_provider=target_provider,
            source_type=source_enum.value,
            target_type=target_enum.value,
            tn=tn,
            progress_callback=progress_callback,
        )

        # Enrichment integration (post-copy).
        if not skip_enrich and isinstance(target_conn, sqlite3.Connection):
            from lsm.db.enrichment import run_enrichment_pipeline

            _emit_progress(progress_callback, "migrate", 0, 0, "Running post-migration enrichment.")
            stage_tracker = _build_enrichment_stage_tracker(
                conn=target_conn,
                run_id=run_id,
                source_type=source_enum.value,
                target_type=target_enum.value,
                tn=tn,
            )
            enrichment_report = run_enrichment_pipeline(
                target_conn,
                target_config,
                table_names=tn,
                stage_tracker=stage_tracker,
                skip_stages=(completed_stages if resume else None),
            )
        elif not skip_enrich:
            _emit_progress(
                progress_callback,
                "migrate",
                0,
                0,
                "Skipping enrichment (requires SQLite target connection).",
            )

        _record_validation_counts(target_conn, expected_counts, tn)
        validation_result = validate_migration(target_conn, tn)
        validation_checked = int(validation_result.get("checked", 0))

    _emit_progress(progress_callback, "migrate", migrated, total, "Migration complete.")
    result: dict[str, Any] = {
        "source": source_enum.value,
        "target": target_enum.value,
        "total_vectors": int(total),
        "migrated_vectors": int(migrated),
        "validated_tables": validation_checked,
    }
    if enrichment_report is not None:
        result["enrichment"] = enrichment_report
    return result


def validate_migration(target_conn: Any, tn: TableNames = DEFAULT_TABLE_NAMES) -> Dict[str, Any]:
    """Validate migration counts previously recorded on the target connection."""
    _ensure_aux_tables(target_conn, tn)
    _ensure_validation_table(target_conn, tn)
    safe_validation_table = safe_identifier(tn.migration_validation)
    rows = (
        compat_fetch_rows_as_dicts(target_conn, f"SELECT * FROM {safe_validation_table}")
        if compat_table_exists(target_conn, safe_validation_table)
        else []
    )
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
            actual = compat_count_rows(target_conn, table_name)
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
    start_offset: int = 0,
    rows_progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[int, int, Optional[int]]:
    total = int(source_provider.count())
    migrated = 0
    offset = max(0, min(int(start_offset), total))
    inferred_dim: Optional[int] = None
    last_page_ids: Optional[tuple[str, ...]] = None

    # Track (timestamp, cumulative_offset) samples for moving-average ETA.
    # Keep enough samples to cover ~5000 vectors worth of batches.
    max_samples = max(2, 5000 // max(1, batch_size) + 1)
    timing_samples: deque[tuple[float, int]] = deque(maxlen=max_samples)
    timing_samples.append((time.monotonic(), offset))

    if rows_progress_callback is not None:
        rows_progress_callback(offset)

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
        if rows_progress_callback is not None:
            rows_progress_callback(offset)

        timing_samples.append((time.monotonic(), offset))
        eta_str = _format_eta(timing_samples, offset, total)

        _emit_progress(
            progress_callback,
            "vectors",
            offset,
            total,
            f"Migrated {offset:,}/{total:,} vectors. ({eta_str})",
        )

    return migrated, total, inferred_dim


def _format_eta(
    samples: deque[tuple[float, int]],
    current: int,
    total: int,
) -> str:
    """Compute an ETA string from a moving window of (timestamp, offset) samples."""
    remaining = total - current
    if remaining <= 0:
        return "done"
    if len(samples) < 2:
        return "estimating..."

    oldest_time, oldest_offset = samples[0]
    newest_time, newest_offset = samples[-1]
    elapsed = newest_time - oldest_time
    vectors_done = newest_offset - oldest_offset

    if elapsed <= 0 or vectors_done <= 0:
        return "estimating..."

    rate = vectors_done / elapsed  # vectors per second
    eta_seconds = remaining / rate

    if eta_seconds < 60:
        return f"ETA {int(eta_seconds)}s"
    if eta_seconds < 3600:
        minutes = int(eta_seconds) // 60
        secs = int(eta_seconds) % 60
        return f"ETA {minutes}m {secs:02d}s"
    hours = int(eta_seconds) // 3600
    minutes = (int(eta_seconds) % 3600) // 60
    return f"ETA {hours}h {minutes:02d}m"


def _aux_copy_stage_name(table_name: str, tn: TableNames) -> str:
    mapping = {
        tn.manifest: "copy_manifest",
        tn.agent_memories: "copy_memories",
        tn.agent_memory_candidates: "copy_memory_candidates",
        tn.agent_schedules: "copy_schedules",
        tn.stats_cache: "copy_stats_cache",
        tn.remote_cache: "copy_remote_cache",
        tn.schema_versions: "copy_schema_versions",
    }
    return mapping.get(table_name, f"copy_{table_name}")


def _copy_aux_table(
    *,
    source_conn: Any,
    target_conn: Any,
    spec: AuxiliaryTableSpec,
    progress_callback: Optional[ProgressCallback],
) -> int:
    if not compat_table_exists(source_conn, spec.name):
        return 0
    rows = compat_fetch_rows_as_dicts(
        source_conn, f"SELECT * FROM {safe_identifier(spec.name)}"
    )
    if rows:
        compat_upsert_rows(target_conn, spec.name, spec.primary_key, rows)
    _emit_progress(
        progress_callback,
        "state",
        len(rows),
        len(rows),
        f"Copied auxiliary table '{spec.name}' ({len(rows)} rows).",
    )
    return len(rows)


def _copy_auxiliary_state(
    *,
    source_conn: Any,
    target_conn: Any,
    progress_callback: Optional[ProgressCallback],
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> Dict[str, int]:
    counts: dict[str, int] = {}
    _ensure_aux_tables(target_conn, tn)

    for spec in _build_aux_table_specs(tn):
        if not compat_table_exists(source_conn, spec.name):
            continue
        rows = compat_fetch_rows_as_dicts(
            source_conn, f"SELECT * FROM {safe_identifier(spec.name)}"
        )
        counts[spec.name] = len(rows)
        if rows:
            compat_upsert_rows(target_conn, spec.name, spec.primary_key, rows)
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
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    _ensure_aux_tables(target_conn, tn)
    payload = _derive_schema_payload(runtime_config, inferred_embedding_dim)

    rows = compat_fetch_rows_as_dicts(
        target_conn,
        f"""
        SELECT
            lsm_version,
            embedding_model,
            embedding_dim,
            chunking_strategy,
            chunk_size,
            chunk_overlap
        FROM {tn.schema_versions}
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
            "id": _next_schema_version_id(target_conn, tn),
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
    compat_upsert_rows(target_conn, tn.schema_versions, "id", rows_to_insert)


def _record_validation_counts(
    target_conn: Any,
    counts: Mapping[str, int],
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    _ensure_validation_table(target_conn, tn)
    now = _utcnow_iso()
    rows = [
        {
            "table_name": table_name,
            "expected_count": int(expected),
            "recorded_at": now,
        }
        for table_name, expected in counts.items()
    ]
    compat_upsert_rows(target_conn, tn.migration_validation, "table_name", rows)


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
        from lsm.vectordb.chromadb import ChromaDBProvider

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


def _to_vectordb_config(raw: Any, provider_hint: Optional[str] = None) -> DBConfig:
    if isinstance(raw, DBConfig):
        return replace(raw, provider=provider_hint or raw.provider)

    if hasattr(raw, "db") and isinstance(raw.db, DBConfig):
        base = raw.db
        return replace(base, provider=provider_hint or base.provider)

    if isinstance(raw, Mapping):
        db_raw: Mapping[str, Any] = raw
        if "db" in raw and isinstance(raw["db"], Mapping):
            db_raw = raw["db"]
        vector_raw: Mapping[str, Any] = db_raw
        if "vector" in db_raw and isinstance(db_raw["vector"], Mapping):
            vector_raw = db_raw["vector"]
        provider = provider_hint or str(vector_raw.get("provider", "sqlite"))
        collection = str(vector_raw.get("collection", "local_kb"))
        db_kwargs: dict[str, Any] = {
            "provider": provider,
            "collection": collection,
            "connection_string": db_raw.get("connection_string"),
            "host": db_raw.get("host"),
            "port": db_raw.get("port"),
            "database": db_raw.get("database"),
            "user": db_raw.get("user"),
            "password": db_raw.get("password"),
            "index_type": vector_raw.get("index_type", "hnsw"),
            "pool_size": int(vector_raw.get("pool_size", 5)),
        }
        if db_raw.get("path") not in {None, ""}:
            db_kwargs["path"] = Path(db_raw.get("path"))
        return DBConfig(
            **db_kwargs,
        )

    raise ValueError("Unable to derive DBConfig for migration source/target.")


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


# ---------------------------------------------------------------------------
# Migration progress tracking
# ---------------------------------------------------------------------------

_MIGRATION_PROGRESS_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS {table} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_run TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    source_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    rows_processed INTEGER DEFAULT 0,
    error_message TEXT
)
"""

_MIGRATION_PROGRESS_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    migration_run TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    source_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    rows_processed INTEGER DEFAULT 0,
    error_message TEXT
)
"""


def _ensure_migration_progress_table(
    conn: Any,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Create the migration_progress table if it does not exist."""
    dialect = compat_dialect(conn)
    template = _MIGRATION_PROGRESS_DDL_SQLITE if dialect == "sqlite" else _MIGRATION_PROGRESS_DDL_POSTGRES
    ddl = template.format(table=tn.migration_progress)
    compat_execute(conn, ddl)
    compat_commit(conn)


def _begin_stage(
    conn: Any,
    run_id: str,
    stage: str,
    source_type: str,
    target_type: str,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> int:
    """Record the start of a migration stage. Returns the row id."""
    _ensure_migration_progress_table(conn, tn)
    now = datetime.now(timezone.utc).isoformat()
    params = (run_id, now, source_type, target_type, stage)
    if compat_dialect(conn) == "postgresql":
        cursor = compat_execute(
            conn,
            f"INSERT INTO {tn.migration_progress} "
            f"(migration_run, started_at, source_type, target_type, stage, status) "
            f"VALUES (?, ?, ?, ?, ?, 'in_progress') RETURNING id",
            params,
        )
        row = cursor.fetchone()
        compat_commit(conn)
        return row[0]
    else:
        cursor = compat_execute(
            conn,
            f"INSERT INTO {tn.migration_progress} "
            f"(migration_run, started_at, source_type, target_type, stage, status) "
            f"VALUES (?, ?, ?, ?, ?, 'in_progress')",
            params,
        )
        compat_commit(conn)
        return cursor.lastrowid


def _complete_stage(
    conn: Any,
    stage_id: int,
    rows: int = 0,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Mark a migration stage as completed."""
    now = datetime.now(timezone.utc).isoformat()
    compat_execute(
        conn,
        f"UPDATE {tn.migration_progress} "
        f"SET status = 'completed', completed_at = ?, rows_processed = ? "
        f"WHERE id = ?",
        (now, rows, stage_id),
    )
    compat_commit(conn)


def _set_stage_rows_processed(
    conn: Any,
    stage_id: int,
    rows: int,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Update in-progress row count for a stage checkpoint."""
    compat_execute(
        conn,
        f"UPDATE {tn.migration_progress} SET rows_processed = ? WHERE id = ?",
        (int(rows), stage_id),
    )
    compat_commit(conn)


def _fail_stage(
    conn: Any,
    stage_id: int,
    error: str,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Mark a migration stage as failed with an error message."""
    now = datetime.now(timezone.utc).isoformat()
    compat_execute(
        conn,
        f"UPDATE {tn.migration_progress} "
        f"SET status = 'failed', completed_at = ?, error_message = ? "
        f"WHERE id = ?",
        (now, error, stage_id),
    )
    compat_commit(conn)


def _interrupt_stage(
    conn: Any,
    stage_id: int,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Mark a migration stage as interrupted (user cancelled)."""
    now = datetime.now(timezone.utc).isoformat()
    compat_execute(
        conn,
        f"UPDATE {tn.migration_progress} "
        f"SET status = 'interrupted', completed_at = ? "
        f"WHERE id = ?",
        (now, stage_id),
    )
    compat_commit(conn)


def _has_incomplete_stages(
    conn: Any,
    run_id: str,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> bool:
    """Check whether a run has any interrupted, in-progress, or failed stages."""
    row = compat_execute(
        conn,
        f"SELECT COUNT(*) FROM {tn.migration_progress} "
        f"WHERE migration_run = ? AND status IN ('in_progress', 'interrupted', 'failed')",
        (run_id,),
    ).fetchone()
    return bool(row and row[0] > 0)


def _get_stage_rows_processed(
    conn: Any,
    run_id: str,
    stage: str,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> int:
    """Return the highest checkpoint row count for a stage in a run."""
    _ensure_migration_progress_table(conn, tn)
    row = compat_execute(
        conn,
        f"SELECT COALESCE(MAX(rows_processed), 0) FROM {tn.migration_progress} "
        f"WHERE migration_run = ? AND stage = ?",
        (run_id, stage),
    ).fetchone()
    return int((row[0] if row is not None else 0) or 0)


def _get_completed_stages(
    conn: Any,
    run_id: str,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> set[str]:
    """Get stage names that completed in a given migration run."""
    _ensure_migration_progress_table(conn, tn)
    rows = compat_execute(
        conn,
        f"SELECT stage FROM {tn.migration_progress} "
        f"WHERE migration_run = ? AND status = 'completed'",
        (run_id,),
    ).fetchall()
    return {row[0] for row in rows}


def _get_latest_run_id(
    conn: Any,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> Optional[str]:
    """Get the most recent migration_run id."""
    _ensure_migration_progress_table(conn, tn)
    row = compat_execute(
        conn,
        f"SELECT migration_run FROM {tn.migration_progress} "
        f"ORDER BY id DESC LIMIT 1",
    ).fetchone()
    return row[0] if row else None


def _build_enrichment_stage_tracker(
    *,
    conn: Any,
    run_id: str,
    source_type: str,
    target_type: str,
    tn: TableNames,
) -> Callable[[str, str], None]:
    """Create stage tracker callback backed by migration_progress rows."""
    stage_ids: dict[str, int] = {}

    def _tracker(stage_name: str, status: str) -> None:
        if status == "in_progress":
            stage_ids[stage_name] = _begin_stage(
                conn,
                run_id,
                stage_name,
                source_type,
                target_type,
                tn,
            )
            return
        stage_id = stage_ids.get(stage_name)
        if stage_id is None:
            return
        if status == "completed":
            _complete_stage(conn, stage_id, tn=tn)
            stage_ids.pop(stage_name, None)
            return
        if status == "failed":
            _fail_stage(conn, stage_id, "enrichment stage failed", tn=tn)
            stage_ids.pop(stage_name, None)
            return

    return _tracker


# ---------------------------------------------------------------------------
# Schema evolution
# ---------------------------------------------------------------------------

def _evolve_schema(
    conn: Any,
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Ensure target DB has all current-version tables and columns.

    Runs column-level evolution first (so new columns exist for indexes),
    then calls ensure_application_schema() for any new tables and indexes.
    """
    from lsm.db.schema import ensure_application_schema

    # Column-level evolution first (adds missing columns to existing tables)
    dialect = compat_dialect(conn)
    if dialect == "sqlite":
        _evolve_sqlite_columns(conn, tn)
    compat_commit(conn)

    # Now safe to create tables/indexes that reference the new columns
    try:
        ensure_application_schema(conn, table_names=tn)
    except Exception:
        logger.debug("Schema evolution: ensure_application_schema partial failure", exc_info=True)
    compat_commit(conn)


def _evolve_sqlite_columns(conn: Any, tn: TableNames) -> None:
    """Add missing columns to existing SQLite tables.

    SQLite does not support ADD COLUMN IF NOT EXISTS, so we check
    PRAGMA table_info() first.
    """
    from lsm.db.schema import ensure_application_schema

    # Get the expected schema by creating a temp in-memory DB
    import sqlite3 as _sqlite3

    ref_conn = _sqlite3.connect(":memory:")
    ensure_application_schema(ref_conn, table_names=tn)

    for table_name in tn.application_tables():
        try:
            existing_cols = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
        except Exception:
            continue  # table doesn't exist, ensure_application_schema handles creation

        try:
            ref_cols = {
                row[1]: row[2]  # name -> type
                for row in ref_conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
        except Exception:
            continue

        for col_name, col_type in ref_cols.items():
            if col_name not in existing_cols:
                try:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                    )
                except Exception:
                    pass  # column may have been added concurrently

    ref_conn.close()


def _rebuild_fts_if_needed(
    *,
    target_conn: Any,
    target_provider: Any,
    source_type: str,
    target_type: str,
    tn: TableNames,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Rebuild FTS indexes after cross-backend migration.

    - **PG → SQLite**: Rebuilds FTS5 virtual table and sync triggers by
      calling the SQLite provider's ``_ensure_fts_and_triggers()``.
    - **SQLite → PG**: Rebuilds tsvector column and GIN index by
      calling the PostgreSQL provider's ``_ensure_fts()``.

    Does nothing for same-backend migrations.
    """
    if source_type == target_type:
        return

    if target_type == "sqlite" and source_type in ("postgresql", "chroma"):
        # SQLite target: rebuild FTS5 from chunks data
        _emit_progress(progress_callback, "migrate", 0, 0, "Rebuilding FTS5 index...")
        try:
            fts_setup = getattr(target_provider, "_ensure_fts_and_triggers", None)
            if callable(fts_setup):
                fts_setup()
                logger.info("FTS5 index rebuilt after %s → SQLite migration.", source_type)
            else:
                logger.debug("Target provider has no _ensure_fts_and_triggers(); skipping FTS rebuild.")
        except Exception as exc:
            logger.warning("FTS5 rebuild failed (non-fatal): %s", exc)

    elif target_type == "postgresql" and source_type in ("sqlite", "chroma"):
        # PostgreSQL target: create tsvector column + GIN index
        _emit_progress(progress_callback, "migrate", 0, 0, "Rebuilding tsvector/GIN index...")
        try:
            fts_setup = getattr(target_provider, "_ensure_fts", None)
            if callable(fts_setup):
                with _connection_context(target_provider) as conn:
                    if conn is not None:
                        fts_setup(conn)
                        logger.info("tsvector/GIN index rebuilt after %s → PostgreSQL migration.", source_type)
            else:
                logger.debug("Target provider has no _ensure_fts(); skipping FTS rebuild.")
        except Exception as exc:
            logger.warning("tsvector/GIN rebuild failed (non-fatal): %s", exc)


@contextmanager
def _connection_context(provider: Any):
    if provider is None:
        yield None
        return
    conn_cm = resolve_connection(provider)
    entered = False
    try:
        with conn_cm as conn:
            entered = True
            yield conn
    except Exception:
        if entered:
            raise
        yield None


def _ensure_aux_tables(conn: Any, tn: TableNames = DEFAULT_TABLE_NAMES) -> None:
    dialect = compat_dialect(conn)
    for spec in _build_aux_table_specs(tn):
        ddl = spec.sqlite_ddl if dialect == "sqlite" else spec.postgres_ddl
        compat_execute(conn, ddl)
    _ensure_validation_table(conn, tn)
    compat_commit(conn)


def _ensure_validation_table(conn: Any, tn: TableNames = DEFAULT_TABLE_NAMES) -> None:
    dialect = compat_dialect(conn)
    table = tn.migration_validation
    if dialect == "sqlite":
        compat_execute(
            conn,
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                table_name TEXT PRIMARY KEY,
                expected_count INTEGER NOT NULL,
                recorded_at TEXT NOT NULL
            )
            """,
        )
    else:
        compat_execute(
            conn,
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                table_name TEXT PRIMARY KEY,
                expected_count BIGINT NOT NULL,
                recorded_at TEXT NOT NULL
            )
            """,
        )
    compat_commit(conn)

def _count_vector_rows_for_key(
    conn: Any, key: Optional[str], tn: TableNames = DEFAULT_TABLE_NAMES
) -> int:
    preferred_table: Optional[str] = None
    if key and ":" in key:
        _prefix, table_name = key.split(":", 1)
        table_name = str(table_name or "").strip()
        if table_name:
            preferred_table = table_name
    if preferred_table:
        if compat_table_exists(conn, preferred_table):
            return compat_count_rows(conn, preferred_table)
        return 0

    if compat_table_exists(conn, tn.chunks):
        row = compat_execute(conn, f"SELECT COUNT(*) FROM {tn.chunks}").fetchone()
        return int(row[0] or 0) if row is not None else 0

    if compat_dialect(conn) == "postgresql":
        rows = compat_fetch_rows_as_dicts(
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
                return compat_count_rows(conn, table_name)
    return 0


def _count_vector_rows(
    conn: Any, key: Optional[str] = None, tn: TableNames = DEFAULT_TABLE_NAMES
) -> int:
    return _count_vector_rows_for_key(conn, key, tn)


def _vector_validation_key(
    target_provider: Any, target_conn: Any, tn: TableNames = DEFAULT_TABLE_NAMES
) -> str:
    if compat_table_exists(target_conn, tn.chunks):
        return f"vector_rows:{tn.chunks}"

    table_name = getattr(target_provider, "_table_name", None)
    if isinstance(table_name, str):
        candidate = table_name.strip()
        if candidate:
            try:
                safe_identifier(candidate)
            except ValueError:
                return "vector_rows"
            return f"vector_rows:{candidate}"

    return "vector_rows"


def _next_schema_version_id(conn: Any, tn: TableNames = DEFAULT_TABLE_NAMES) -> int:
    row = compat_execute(
        conn, f"SELECT COALESCE(MAX(id), 0) FROM {tn.schema_versions}"
    ).fetchone()
    current = int(row[0] or 0) if row is not None else 0
    return current + 1


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


def _resolve_v07_file(source_dir: Path, filename: str) -> Optional[Path]:
    """Search for a v0.7 legacy file in standard subdirectories.

    Checks ``source_dir/``, then ``source_dir/.ingest/``, then
    ``source_dir/Agents/``.

    Returns:
        Path to the first match, or ``None`` if not found.
    """
    for subdir in ("", ".ingest", "Agents"):
        candidate = source_dir / subdir / filename if subdir else source_dir / filename
        if candidate.exists():
            return candidate
    return None


def auto_detect_migration(
    global_folder: Path,
    config: Any,
) -> dict:
    """Auto-detect migration source and target from filesystem heuristics.

    Returns:
        Dict with keys: ``from_db``, ``from_version``, ``to_db``, ``to_version``,
        ``source_dir`` (optional).
    """
    from lsm import __version__ as current_version
    from lsm.db.connection import resolve_db_path

    result: dict[str, Any] = {
        "from_db": None,
        "from_version": None,
        "to_db": getattr(getattr(config, "db", None), "provider", "sqlite"),
        "to_version": current_version,
        "source_dir": None,
    }

    folder = Path(global_folder).expanduser().resolve()

    # 1. .chroma/ exists → from_db = "chroma"
    #    ChromaDB was the v0.7 provider and has no schema version table,
    #    so assume v0.7 when no version can be read from the source.
    chroma_dir = folder / ".chroma"
    if chroma_dir.is_dir():
        result["from_db"] = "chroma"
        result["from_version"] = "v0.7"
        return result

    # 2. lsm.db exists → from_db = "sqlite", read schema version
    for data_subdir in ("data", ".ingest", ""):
        db_candidate = folder / data_subdir / "lsm.db" if data_subdir else folder / "lsm.db"
        if db_candidate.exists():
            result["from_db"] = "sqlite"
            try:
                conn = sqlite3.connect(str(db_candidate))
                conn.row_factory = sqlite3.Row
                from lsm.db.schema_version import get_active_schema_version
                version_row = get_active_schema_version(conn)
                if version_row:
                    result["from_version"] = version_row.get("lsm_version")
                conn.close()
            except Exception:
                pass
            return result

    # 3. PostgreSQL configured and reachable
    db_cfg = getattr(config, "db", None)
    if db_cfg and getattr(db_cfg, "provider", "") == "postgresql":
        conn_str = getattr(db_cfg, "connection_string", None)
        if conn_str:
            try:
                import psycopg2
                conn = psycopg2.connect(conn_str, connect_timeout=5)
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                conn.close()
                result["from_db"] = "postgresql"
                return result
            except Exception:
                pass

    # 4. manifest.json without lsm.db → v0.7 legacy
    manifest = _resolve_v07_file(folder, "manifest.json")
    if manifest is not None:
        result["from_version"] = "v0.7"
        result["source_dir"] = str(folder)
        return result

    return result


def _migrate_v07_legacy(
    *,
    source_dir: Path,
    target_conn: Any,
    progress_callback: Optional[ProgressCallback],
    tn: TableNames = DEFAULT_TABLE_NAMES,
) -> Dict[str, int]:
    _ensure_aux_tables(target_conn, tn)
    counts: dict[str, int] = {
        tn.manifest: 0,
        tn.agent_memories: 0,
        tn.agent_memory_candidates: 0,
        tn.agent_schedules: 0,
        tn.stats_cache: 0,
        tn.remote_cache: 0,
    }

    _emit_progress(
        progress_callback,
        "legacy",
        0,
        0,
        f"Importing legacy v0.7 state from {source_dir}.",
    )

    manifest_path = _resolve_v07_file(source_dir, "manifest.json")
    if manifest_path is not None:
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
            compat_upsert_rows(target_conn, tn.manifest, "source_path", rows)
        counts[tn.manifest] = len(rows)
    else:
        _warn_legacy_missing(source_dir / "manifest.json")

    memories_db_path = _resolve_v07_file(source_dir, "memories.db")
    if memories_db_path is not None:
        legacy_conn = sqlite3.connect(str(memories_db_path))
        legacy_conn.row_factory = sqlite3.Row
        try:
            memory_table = _first_existing_table(
                legacy_conn,
                (tn.agent_memories, "memories"),
            )
            candidate_table = _first_existing_table(
                legacy_conn,
                (tn.agent_memory_candidates, "memory_candidates"),
            )

            memory_rows: list[dict[str, Any]] = []
            if memory_table is not None:
                safe_memory_table = safe_identifier(memory_table)
                for row in compat_fetch_rows_as_dicts(
                    legacy_conn, f"SELECT * FROM {safe_memory_table}"
                ):
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
                    compat_upsert_rows(target_conn, tn.agent_memories, "id", memory_rows)
            counts[tn.agent_memories] = len(memory_rows)

            candidate_rows: list[dict[str, Any]] = []
            if candidate_table is not None:
                safe_candidate_table = safe_identifier(candidate_table)
                for row in compat_fetch_rows_as_dicts(
                    legacy_conn, f"SELECT * FROM {safe_candidate_table}"
                ):
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
                    compat_upsert_rows(
                        target_conn,
                        tn.agent_memory_candidates,
                        "id",
                        candidate_rows,
                    )
            counts[tn.agent_memory_candidates] = len(candidate_rows)
        finally:
            legacy_conn.close()
    else:
        _warn_legacy_missing(source_dir / "memories.db")

    schedules_path = _resolve_v07_file(source_dir, "schedules.json")
    if schedules_path is not None:
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
            compat_upsert_rows(target_conn, tn.agent_schedules, "schedule_id", schedule_rows)
        counts[tn.agent_schedules] = len(schedule_rows)
    else:
        _warn_legacy_missing(source_dir / "schedules.json")

    stats_cache_path = _resolve_v07_file(source_dir, "stats_cache.json")
    if stats_cache_path is not None:
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
            compat_upsert_rows(target_conn, tn.stats_cache, "cache_key", stats_rows)
        counts[tn.stats_cache] = len(stats_rows)
    else:
        _warn_legacy_missing(source_dir / "stats_cache.json")

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
        compat_upsert_rows(target_conn, tn.remote_cache, "cache_key", remote_rows)
    counts[tn.remote_cache] = len(remote_rows)

    _emit_progress(
        progress_callback,
        "legacy",
        1,
        1,
        (
            "Legacy import complete: "
            f"manifest={counts[tn.manifest]}, memories={counts[tn.agent_memories]}, "
            f"candidates={counts[tn.agent_memory_candidates]}, schedules={counts[tn.agent_schedules]}, "
            f"stats={counts[tn.stats_cache]}, remote={counts[tn.remote_cache]}."
        ),
    )
    return counts


def _first_existing_table(conn: Any, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if compat_table_exists(conn, name):
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
