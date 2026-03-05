"""Database health check for startup diagnostics.

Detects version mismatches, missing/corrupt databases, legacy provider state,
and partially completed migrations. Provides clear guidance on resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lsm import __version__ as LSM_VERSION
from lsm.config.models import DBConfig, VectorConfig
from lsm.db.compat import (
    execute,
    fetchone,
    is_sqlite,
    table_exists,
)
from lsm.db.connection import create_sqlite_connection, resolve_db_path
from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames


# ---------------------------------------------------------------------------
# DBHealthReport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DBHealthReport:
    """Result of a database health check.

    Attributes:
        status: One of ``"ok"``, ``"missing"``, ``"mismatch"``,
            ``"corrupt"``, ``"partial_migration"``, ``"legacy_detected"``,
            ``"stale_chunks"``.
        details: Human-readable explanation of the status.
        suggested_action: Actionable CLI command or guidance.
        schema_diff: Field-level diff when ``status == "mismatch"``.
        blocking: Whether this status should block startup.
    """

    status: str
    details: str = ""
    suggested_action: str = ""
    schema_diff: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocking: bool = False


_OK = DBHealthReport(status="ok")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_db_reachable(config: Any) -> Optional[DBHealthReport]:
    """Check whether the database file/server is reachable."""
    provider = getattr(config, "db", None)
    if provider is None:
        return None

    provider_name = getattr(provider, "provider", VectorConfig.provider)

    if provider_name == "sqlite":
        db_path = resolve_db_path(getattr(provider, "path", DBConfig().path))
        if not db_path.exists():
            return DBHealthReport(
                status="missing",
                details=f"Database file not found: {db_path}",
                suggested_action=(
                    "This is normal on first run. "
                    "Run `lsm ingest build` to create the database."
                ),
                blocking=False,
            )
        return None  # reachable

    if provider_name == "postgresql":
        try:
            import psycopg2

            conn_str = getattr(provider, "connection_string", None)
            if conn_str:
                conn = psycopg2.connect(conn_str, connect_timeout=5)
            else:
                conn = psycopg2.connect(
                    host=getattr(provider, "host", None),
                    port=getattr(provider, "port", None),
                    dbname=getattr(provider, "database", None),
                    user=getattr(provider, "user", None),
                    password=getattr(provider, "password", None),
                    connect_timeout=5,
                )
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            conn.close()
            return None  # reachable
        except ImportError:
            return DBHealthReport(
                status="corrupt",
                details="psycopg2 is not installed but PostgreSQL provider is configured.",
                suggested_action="Install psycopg2: pip install psycopg2-binary",
                blocking=True,
            )
        except Exception as exc:
            return DBHealthReport(
                status="corrupt",
                details=f"Cannot connect to PostgreSQL: {exc}",
                suggested_action=(
                    "Check your db.connection_string or db host/port/database/user settings. "
                    "Verify that PostgreSQL is running and accepting connections."
                ),
                blocking=True,
            )

    return None


def _check_legacy_provider(config: Any) -> Optional[DBHealthReport]:
    """Check whether the loaded configuration references a legacy provider.

    Only flags an issue when the *config itself* selects a discontinued
    backend (``chromadb``).  Leftover ``.chroma/`` directories on disk are
    harmless after migration and should not block startup.
    """
    db_cfg = getattr(config, "db", None)
    if db_cfg is None:
        return None

    provider_name = str(getattr(db_cfg, "provider", "") or "").strip().lower()
    if provider_name in ("chromadb", "chroma"):
        return DBHealthReport(
            status="legacy_detected",
            details=(
                f"Configuration sets db.provider to '{provider_name}', "
                "which is a legacy backend no longer supported in v0.8+."
            ),
            suggested_action=(
                "Run `lsm migrate --from-db chroma` to migrate your data to the "
                "current database backend, then update db.provider in your config "
                "to 'sqlite' or 'postgresql'."
            ),
            blocking=True,
        )
    return None


def _check_schema_version(
    conn: Any,
    config: Any,
    table_names: TableNames,
) -> Optional[DBHealthReport]:
    """Check whether schema version is compatible with current config."""
    from lsm.db.schema_version import check_schema_compatibility

    try:
        compatible, diff = check_schema_compatibility(conn, config, table_names=table_names)
    except Exception:
        return None  # table may not exist yet

    if not compatible:
        field_details = ", ".join(
            f"{k}: {v.get('old')!r} -> {v.get('new')!r}" for k, v in diff.items()
        )
        return DBHealthReport(
            status="mismatch",
            details=f"Schema version mismatch: {field_details}",
            suggested_action=(
                "Run `lsm migrate` to update the database schema, "
                "or `lsm ingest build --force-reingest-changed-config` to rebuild."
            ),
            schema_diff=diff,
            blocking=True,
        )
    return None


def _check_required_tables(
    conn: Any,
    table_names: TableNames,
) -> Optional[DBHealthReport]:
    """Verify all required application tables exist."""
    missing = []
    for tbl in table_names.application_tables():
        try:
            if not table_exists(conn, tbl):
                missing.append(tbl)
        except Exception:
            missing.append(tbl)

    if missing:
        return DBHealthReport(
            status="corrupt",
            details=f"Missing required tables: {', '.join(sorted(missing))}",
            suggested_action=(
                "The database may be corrupt or from an incompatible version. "
                "Run `lsm migrate` to attempt repair, or `lsm ingest build --force` "
                "to rebuild from scratch."
            ),
            blocking=True,
        )
    return None


def _check_partial_migration(
    conn: Any,
    table_names: TableNames,
) -> Optional[DBHealthReport]:
    """Check for incomplete migration state.

    This depends on the migration_progress table created in Phase 17.6.
    Returns None (ok) if the table does not exist yet.
    """
    try:
        if not table_exists(conn, table_names.migration_progress):
            return None  # table doesn't exist yet — no migration tracking

        stuck = fetchone(
            conn,
            f"SELECT COUNT(*) FROM {table_names.migration_progress} "
            f"WHERE status IN ('in_progress', 'interrupted', 'failed')",
        )
        if stuck and stuck[0] > 0:
            return DBHealthReport(
                status="partial_migration",
                details=(
                    f"Found {stuck[0]} incomplete or failed migration task(s). "
                    "A previous migration may have been interrupted."
                ),
                suggested_action="Run `lsm migrate --resume` to continue the migration.",
                blocking=True,
            )
    except Exception:
        return None  # table structure unexpected — skip silently

    return None


def _check_stale_chunks(
    conn: Any,
    config: Any | TableNames | None = None,
    table_names: Optional[TableNames] = None,
) -> Optional[DBHealthReport]:
    """Check for chunks missing enrichment fields using deterministic detection."""
    from lsm.db.enrichment import detect_stale_chunks

    # Backward compatibility for existing internal/test callers that passed
    # (conn, table_names) prior to config-aware stale checks.
    if isinstance(config, TableNames) and table_names is None:
        table_names = config
        config = None
    tn = table_names or DEFAULT_TABLE_NAMES

    try:
        if not table_exists(conn, tn.chunks):
            return None  # no chunks table yet

        query_cfg = getattr(config, "query", None)
        stale = detect_stale_chunks(
            conn,
            tn,
            cluster_enabled=bool(getattr(query_cfg, "cluster_enabled", False)),
            summaries_enabled=(
                str(getattr(query_cfg, "retrieval_profile", "") or "").strip().lower()
                == "multi_vector"
            ),
        )
        tier1 = stale.get("tier1", {})
        tier2 = stale.get("tier2", {})
        tier3 = stale.get("tier3", {})
        detail_parts: list[str] = []
        if int(tier1.get("simhash_null_count", 0) or 0) > 0:
            detail_parts.append(f"{tier1['simhash_null_count']} chunks with NULL simhash")
        if int(tier1.get("node_type_null_count", 0) or 0) > 0:
            detail_parts.append(f"{tier1['node_type_null_count']} chunks with NULL/empty node_type")
        if int(tier2.get("heading_path_null_count", 0) or 0) > 0:
            detail_parts.append(f"{tier2['heading_path_null_count']} headed chunks with NULL heading_path")
        missing_section = int(len(tier3.get("missing_section_summary_files", []) or []))
        missing_file = int(len(tier3.get("missing_file_summary_files", []) or []))
        if missing_section > 0:
            detail_parts.append(f"{missing_section} files missing section summaries")
        if missing_file > 0:
            detail_parts.append(f"{missing_file} files missing file summaries")

        if detail_parts:
            return DBHealthReport(
                status="stale_chunks",
                details=f"Stale chunks detected: {'; '.join(detail_parts)}.",
                suggested_action=(
                    "Run `lsm migrate --enrich` to enrich existing chunks, "
                    "or `lsm ingest build --force-reingest-changed-config` to rebuild."
                ),
                blocking=False,
            )
    except Exception:
        return None

    return None


def _default_schema_config(config: Any) -> dict[str, Any]:
    global_cfg = getattr(config, "global_settings", None)
    ingest_cfg = getattr(config, "ingest", None)
    embedding_dim = getattr(global_cfg, "embedding_dimension", None)
    return {
        "lsm_version": LSM_VERSION,
        "embedding_model": str(getattr(global_cfg, "embed_model", "") or ""),
        "embedding_dim": int(embedding_dim or 0),
        "chunking_strategy": str(getattr(ingest_cfg, "chunking_strategy", "") or ""),
        "chunk_size": int(getattr(ingest_cfg, "chunk_size", 0) or 0),
        "chunk_overlap": int(getattr(ingest_cfg, "chunk_overlap", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_db_health(
    config: Any,
    *,
    table_names: Optional[TableNames] = None,
    schema_config: Any = None,
) -> DBHealthReport:
    """Run all database health checks and return the first non-ok result.

    Checks are ordered by severity: reachability, legacy detection, schema
    compatibility, required tables, partial migration, stale chunks.

    Args:
        config: LSMConfig instance.
        table_names: Optional TableNames override (defaults to DEFAULT_TABLE_NAMES).
        schema_config: Optional schema config for version comparison.
            If not provided, attempts to build one from config.

    Returns:
        DBHealthReport with the most severe issue found, or status ``"ok"``.
    """
    tn = table_names or DEFAULT_TABLE_NAMES

    # 1. Database reachable?
    report = _check_db_reachable(config)
    if report is not None:
        return report

    # 2. Legacy provider detected?
    report = _check_legacy_provider(config)
    if report is not None:
        return report

    # For remaining checks, we need a database connection.
    db_cfg = getattr(config, "db", None)
    if db_cfg is None:
        return _OK

    provider_name = getattr(db_cfg, "provider", VectorConfig.provider)

    if provider_name == "sqlite":
        db_path = resolve_db_path(getattr(db_cfg, "path", DBConfig().path))
        if not db_path.exists():
            return _OK  # already handled by reachability check

        try:
            conn = create_sqlite_connection(db_path)
        except Exception as exc:
            return DBHealthReport(
                status="corrupt",
                details=f"Cannot open database file: {exc}",
                suggested_action="The database file may be corrupt. Check file permissions.",
                blocking=True,
            )
        close_conn = True
    elif provider_name == "postgresql":
        try:
            import psycopg2

            conn_str = getattr(db_cfg, "connection_string", None)
            if conn_str:
                conn = psycopg2.connect(conn_str, connect_timeout=5)
            else:
                conn = psycopg2.connect(
                    host=getattr(db_cfg, "host", None),
                    port=getattr(db_cfg, "port", None),
                    dbname=getattr(db_cfg, "database", None),
                    user=getattr(db_cfg, "user", None),
                    password=getattr(db_cfg, "password", None),
                    connect_timeout=5,
                )
            close_conn = True
        except Exception:
            # Connectivity failure already caught in _check_db_reachable
            return _OK
    else:
        return _OK

    try:
        # 3. Schema version compatible?
        effective_schema_config = schema_config if schema_config is not None else _default_schema_config(config)
        report = _check_schema_version(conn, effective_schema_config, tn)
        if report is not None:
            return report

        # 4. Required tables present?
        report = _check_required_tables(conn, tn)
        if report is not None:
            return report

        # 5. Partial migration?
        report = _check_partial_migration(conn, tn)
        if report is not None:
            return report

        # 6. Stale chunks?
        report = _check_stale_chunks(conn, config, tn)
        if report is not None:
            return report
    finally:
        if close_conn:
            conn.close()

    return _OK
