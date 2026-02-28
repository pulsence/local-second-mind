"""Schema version tracking for unified database ingest state."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Tuple

from lsm import __version__ as LSM_VERSION


SCHEMA_COMPARISON_FIELDS: tuple[str, ...] = (
    "lsm_version",
    "embedding_model",
    "embedding_dim",
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
)


class SchemaVersionMismatchError(RuntimeError):
    """Raised when ingest config is incompatible with active schema version."""

    def __init__(self, diff: Mapping[str, Mapping[str, Any]]) -> None:
        self.diff = dict(diff)
        details = ", ".join(
            f"{field}: {values.get('old')!r} -> {values.get('new')!r}"
            for field, values in self.diff.items()
        )
        super().__init__(
            "Schema version mismatch detected. "
            f"Changed fields: {details}. "
            "Run `lsm migrate` (or `lsm ingest build --force-reingest-changed-config`)."
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_schema_versions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lsm_schema_versions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
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
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_lsm_schema_versions_created_at
        ON lsm_schema_versions(created_at)
        """
    )
    conn.commit()


def _get_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_schema_config(config: Any) -> dict[str, Any]:
    return {
        "lsm_version": str(_get_value(config, "lsm_version", LSM_VERSION)),
        "embedding_model": str(_get_value(config, "embedding_model", "") or ""),
        "embedding_dim": int(_get_value(config, "embedding_dim", 0) or 0),
        "chunking_strategy": str(_get_value(config, "chunking_strategy", "") or ""),
        "chunk_size": int(_get_value(config, "chunk_size", 0) or 0),
        "chunk_overlap": int(_get_value(config, "chunk_overlap", 0) or 0),
    }


def get_active_schema_version(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
    """Return the most recent schema version row."""
    _ensure_schema_versions_table(conn)
    row = conn.execute(
        """
        SELECT
            id,
            manifest_version,
            lsm_version,
            embedding_model,
            embedding_dim,
            chunking_strategy,
            chunk_size,
            chunk_overlap,
            created_at,
            last_ingest_at
        FROM lsm_schema_versions
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    columns = [item[0] for item in (conn.execute("SELECT * FROM lsm_schema_versions LIMIT 1").description or [])]
    if columns:
        # Preserve compatibility for sqlite3.Row and tuple-style rows.
        if isinstance(row, sqlite3.Row):
            return {column: row[column] for column in columns if column in row.keys()}
        return dict(zip(columns, row))
    # Fallback for uncommon cursor metadata behavior.
    return dict(row) if isinstance(row, sqlite3.Row) else None


def record_schema_version(conn: sqlite3.Connection, config: Any) -> int:
    """Insert a new schema version row and return its primary key."""
    _ensure_schema_versions_table(conn)
    normalized = _normalize_schema_config(config)
    now = _now_iso()
    cursor = conn.execute(
        """
        INSERT INTO lsm_schema_versions (
            manifest_version,
            lsm_version,
            embedding_model,
            embedding_dim,
            chunking_strategy,
            chunk_size,
            chunk_overlap,
            created_at,
            last_ingest_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            None,
            normalized["lsm_version"],
            normalized["embedding_model"],
            normalized["embedding_dim"],
            normalized["chunking_strategy"],
            normalized["chunk_size"],
            normalized["chunk_overlap"],
            now,
            now,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def check_schema_compatibility(
    conn: sqlite3.Connection,
    config: Any,
    *,
    raise_on_mismatch: bool = False,
) -> Tuple[bool, dict[str, dict[str, Any]]]:
    """Compare current config to active schema row."""
    _ensure_schema_versions_table(conn)
    active = get_active_schema_version(conn)
    if active is None:
        return True, {}

    normalized = _normalize_schema_config(config)
    diff: dict[str, dict[str, Any]] = {}
    for field in SCHEMA_COMPARISON_FIELDS:
        old_value = active.get(field)
        new_value = normalized.get(field)
        if old_value != new_value:
            diff[field] = {"old": old_value, "new": new_value}

    compatible = len(diff) == 0
    if not compatible and raise_on_mismatch:
        raise SchemaVersionMismatchError(diff)
    return compatible, diff

