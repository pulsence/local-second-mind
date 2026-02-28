from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_manifest_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lsm_manifest (
            source_path TEXT PRIMARY KEY,
            mtime_ns INTEGER,
            file_size INTEGER,
            file_hash TEXT,
            version INTEGER,
            embedding_model TEXT,
            schema_version_id INTEGER,
            updated_at TEXT
        );
        """
    )
    try:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_lsm_manifest_updated_at ON lsm_manifest(updated_at);"
        )
    except sqlite3.OperationalError:
        # Existing legacy tables may not have updated_at; skip index creation.
        pass
    connection.commit()


def load_manifest(
    path: Optional[Path] = None,
    *,
    connection: Optional[sqlite3.Connection] = None,
) -> Dict[str, Dict[str, Any]]:
    if connection is not None:
        _ensure_manifest_table(connection)
        cursor = connection.execute(
            """
            SELECT
                source_path,
                mtime_ns,
                file_size,
                file_hash,
                version,
                embedding_model,
                schema_version_id,
                updated_at
            FROM lsm_manifest
            """
        )
        columns = [item[0] for item in (cursor.description or [])]
        manifest: Dict[str, Dict[str, Any]] = {}
        for values in cursor.fetchall():
            row = dict(zip(columns, values))
            source_path = str(row.get("source_path") or "").strip()
            if not source_path:
                continue
            entry: Dict[str, Any] = {
                "mtime_ns": row.get("mtime_ns"),
                "size": row.get("file_size"),
                "file_hash": row.get("file_hash"),
                "updated_at": row.get("updated_at"),
            }
            version = row.get("version")
            if version is not None:
                entry["version"] = int(version)
            if row.get("embedding_model") is not None:
                entry["embedding_model"] = row.get("embedding_model")
            if row.get("schema_version_id") is not None:
                entry["schema_version_id"] = row.get("schema_version_id")
            manifest[source_path] = entry
        return manifest

    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(source_path): dict(entry)
        for source_path, entry in raw.items()
        if isinstance(entry, dict)
    }


def save_manifest(
    path: Optional[Path],
    manifest: Dict[str, Dict[str, Any]],
    *,
    connection: Optional[sqlite3.Connection] = None,
) -> None:
    if connection is not None:
        _ensure_manifest_table(connection)
        connection.execute("BEGIN")
        try:
            for source_path, entry in manifest.items():
                if not isinstance(entry, dict):
                    continue
                connection.execute(
                    """
                    INSERT INTO lsm_manifest (
                        source_path,
                        mtime_ns,
                        file_size,
                        file_hash,
                        version,
                        embedding_model,
                        schema_version_id,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                        mtime_ns = excluded.mtime_ns,
                        file_size = excluded.file_size,
                        file_hash = excluded.file_hash,
                        version = excluded.version,
                        embedding_model = excluded.embedding_model,
                        schema_version_id = excluded.schema_version_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(source_path),
                        entry.get("mtime_ns"),
                        entry.get("size"),
                        entry.get("file_hash"),
                        entry.get("version"),
                        entry.get("embedding_model"),
                        entry.get("schema_version_id"),
                        entry.get("updated_at"),
                    ),
                )

            source_paths = [str(source_path) for source_path in manifest.keys()]
            if source_paths:
                placeholders = ", ".join(["?"] * len(source_paths))
                connection.execute(
                    f"DELETE FROM lsm_manifest WHERE source_path NOT IN ({placeholders})",
                    source_paths,
                )
            else:
                connection.execute("DELETE FROM lsm_manifest")

            connection.commit()
            return
        except Exception:
            connection.rollback()
            raise

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_next_version(
    manifest: Dict[str, Dict[str, Any]],
    source_path: str,
    *,
    connection: Optional[sqlite3.Connection] = None,
) -> int:
    if connection is not None:
        _ensure_manifest_table(connection)
        try:
            cursor = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM lsm_manifest WHERE source_path = ?",
                (source_path,),
            )
            row = cursor.fetchone()
            max_version = int(row[0] or 0) if row is not None else 0
            return max_version + 1
        except sqlite3.OperationalError:
            pass

    prev = manifest.get(source_path)
    if prev is None:
        return 1
    return int(prev.get("version", 0)) + 1
