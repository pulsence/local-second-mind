"""Database health check for startup diagnostics.

Detects version mismatches, missing/corrupt databases, legacy provider state,
and partially completed migrations. Provides clear guidance on resolution.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lsm.db.connection import resolve_db_path
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

    provider_name = getattr(provider, "provider", "sqlite")

    if provider_name == "sqlite":
        db_path = resolve_db_path(getattr(provider, "path", Path(".")))
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
    """Check for leftover ChromaDB directory indicating legacy provider."""
    global_folder = getattr(getattr(config, "global_settings", None), "global_folder", None)
    if global_folder is None:
        return None

    chroma_dir = Path(global_folder) / ".chroma"
    if chroma_dir.is_dir():
        return DBHealthReport(
            status="legacy_detected",
            details=(
                f"Legacy ChromaDB directory found at {chroma_dir}. "
                "This suggests a pre-migration database from an older version."
            ),
            suggested_action=(
                "Run `lsm migrate --from-db chroma` to migrate your data to the "
                "current database backend, then remove the .chroma directory."
            ),
            blocking=True,
        )
    return None


def _check_schema_version(
    conn: sqlite3.Connection,
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
    conn: sqlite3.Connection,
    table_names: TableNames,
) -> Optional[DBHealthReport]:
    """Verify all required application tables exist."""
    missing = []
    for table_name in table_names.application_tables():
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if row is None:
                missing.append(table_name)
        except Exception:
            missing.append(table_name)

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
    conn: sqlite3.Connection,
    table_names: TableNames,
) -> Optional[DBHealthReport]:
    """Check for incomplete migration state.

    This depends on the migration_progress table created in Phase 17.6.
    Returns None (ok) if the table does not exist yet.
    """
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_names.migration_progress,),
        ).fetchone()
        if row is None:
            return None  # table doesn't exist yet — no migration tracking

        stuck = conn.execute(
            f"SELECT COUNT(*) FROM {table_names.migration_progress} "
            f"WHERE status IN ('in_progress', 'failed')"
        ).fetchone()
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
    conn: sqlite3.Connection,
    table_names: TableNames,
) -> Optional[DBHealthReport]:
    """Check for chunks missing enrichment fields.

    This is a stub that will be wired to detect_stale_chunks() from Phase 17.7.
    For now, performs basic NULL field checks.
    """
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_names.chunks,),
        ).fetchone()
        if row is None:
            return None  # no chunks table yet

        # Check for chunks with NULL simhash (basic enrichment indicator)
        counts = {}
        for col_name in ("simhash", "node_type"):
            try:
                result = conn.execute(
                    f"SELECT COUNT(*) FROM {table_names.chunks} WHERE {col_name} IS NULL"
                ).fetchone()
                if result and result[0] > 0:
                    counts[col_name] = result[0]
            except Exception:
                pass  # column may not exist in older schemas

        if counts:
            detail_parts = [f"{count} chunks with NULL {col}" for col, count in counts.items()]
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

    # For remaining checks, we need a SQLite connection
    db_cfg = getattr(config, "db", None)
    if db_cfg is None:
        return _OK

    provider_name = getattr(db_cfg, "provider", "sqlite")
    if provider_name != "sqlite":
        # PostgreSQL checks beyond connectivity are handled elsewhere
        return _OK

    db_path = resolve_db_path(getattr(db_cfg, "path", Path(".")))
    if not db_path.exists():
        return _OK  # already handled by reachability check

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        return DBHealthReport(
            status="corrupt",
            details=f"Cannot open database file: {exc}",
            suggested_action="The database file may be corrupt. Check file permissions.",
            blocking=True,
        )

    try:
        # 3. Schema version compatible?
        if schema_config is not None:
            report = _check_schema_version(conn, schema_config, tn)
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
        report = _check_stale_chunks(conn, tn)
        if report is not None:
            return report
    finally:
        conn.close()

    return _OK
