"""Incremental completion mode detection for selective re-ingest."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional

from lsm.config.models.ingest import RootConfig
from lsm.db.schema_version import check_schema_compatibility
from lsm.ingest.fs import iter_files
from lsm.ingest.utils import canonical_path

CompletionMode = str


def _value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_roots(config: Any) -> list[RootConfig]:
    roots = _value(config, "roots", [])
    resolved: list[RootConfig] = []
    for root in roots or []:
        if isinstance(root, RootConfig):
            resolved.append(root)
        elif isinstance(root, Mapping):
            resolved.append(
                RootConfig(
                    path=Path(str(root.get("path", ""))),
                    tags=list(root.get("tags") or []) or None,
                    content_type=root.get("content_type"),
                )
            )
        else:
            resolved.append(RootConfig(path=Path(str(root))))
    return resolved


def _schema_config_from_runtime(config: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "embedding_model": _value(config, "embedding_model", ""),
        "embedding_dim": _value(config, "embedding_dim", 0),
        "chunking_strategy": _value(config, "chunking_strategy", ""),
        "chunk_size": _value(config, "chunk_size", 0),
        "chunk_overlap": _value(config, "chunk_overlap", 0),
    }
    lsm_version = _value(config, "lsm_version", None)
    if lsm_version is not None:
        payload["lsm_version"] = lsm_version
    return payload


def _configured_exts(config: Any) -> set[str]:
    values = _value(config, "exts", set())
    return {str(ext).strip().lower() for ext in values or [] if str(ext).strip()}


def _configured_excludes(config: Any) -> set[str]:
    values = _value(config, "exclude_dirs", set())
    return {str(name).strip() for name in values or [] if str(name).strip()}


def _manifest_source_paths(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT source_path FROM lsm_manifest").fetchall()
    return {str(row[0]) for row in rows if row and row[0]}


def _manifest_exts(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT source_path FROM lsm_manifest").fetchall()
    exts = set()
    for row in rows:
        if not row or not row[0]:
            continue
        ext = Path(str(row[0])).suffix.lower()
        if ext:
            exts.add(ext)
    return exts


def _discover_source_paths(config: Any) -> set[str]:
    roots = _resolve_roots(config)
    exts = _configured_exts(config)
    excludes = _configured_excludes(config)
    discovered: set[str] = set()
    for path, _root_cfg in iter_files(roots, exts, excludes):
        discovered.add(canonical_path(path))
    return discovered


def _detect_metadata_enrichment(conn: sqlite3.Connection, config: Any) -> bool:
    roots = _resolve_roots(config)
    if not roots:
        return False
    try:
        rows = conn.execute(
            """
            SELECT source_path, root_tags, content_type
            FROM lsm_chunks
            WHERE is_current = 1
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return False
    if not rows:
        return False

    normalized_roots = [
        (
            str(root.path.expanduser().resolve()),
            tuple(root.tags or ()),
            str(root.content_type) if root.content_type else None,
        )
        for root in roots
    ]

    for row in rows:
        source_path = str(row[0] or "")
        if not source_path:
            continue
        expected_tags: tuple[str, ...] = ()
        expected_content_type: Optional[str] = None
        for root_prefix, root_tags, content_type in normalized_roots:
            if source_path.startswith(root_prefix):
                expected_tags = root_tags
                expected_content_type = content_type
                break
        if expected_tags and not str(row[1] or "").strip():
            return True
        if expected_content_type and str(row[2] or "").strip() != expected_content_type:
            return True
    return False


def detect_completion_mode(conn: sqlite3.Connection, config: Any) -> Optional[CompletionMode]:
    """Detect completion mode required for current runtime config."""
    schema_config = _schema_config_from_runtime(config)
    compatible, diff = check_schema_compatibility(conn, schema_config, raise_on_mismatch=False)

    if not compatible:
        if "embedding_model" in diff or "embedding_dim" in diff:
            return "embedding_upgrade"
        return "chunk_boundary_update"

    configured_exts = _configured_exts(config)
    if configured_exts:
        known_exts = _manifest_exts(conn)
        discovered = _discover_source_paths(config)
        missing_exts = configured_exts - known_exts
        if missing_exts and any(Path(path).suffix.lower() in missing_exts for path in discovered):
            return "extension_completion"

    if _detect_metadata_enrichment(conn, config):
        return "metadata_enrichment"

    return None


def get_stale_files(conn: sqlite3.Connection, config: Any, mode: CompletionMode) -> list[str]:
    """Return source paths that should be re-processed for completion mode."""
    manifest_paths = _manifest_source_paths(conn)

    if mode in {"embedding_upgrade", "chunk_boundary_update"}:
        return sorted(manifest_paths)

    if mode == "extension_completion":
        known_exts = _manifest_exts(conn)
        configured_exts = _configured_exts(config)
        newly_enabled_exts = configured_exts - known_exts
        if not newly_enabled_exts:
            return []
        discovered = _discover_source_paths(config)
        return sorted(
            path
            for path in discovered
            if Path(path).suffix.lower() in newly_enabled_exts
        )

    if mode == "metadata_enrichment":
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT source_path
                FROM lsm_chunks
                WHERE is_current = 1
                  AND (root_tags IS NULL OR root_tags = '' OR content_type IS NULL OR content_type = '')
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return sorted(manifest_paths)
        stale = {str(row[0]) for row in rows if row and row[0]}
        if stale:
            return sorted(stale)
        return sorted(manifest_paths)

    return []
