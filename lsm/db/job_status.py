"""
Startup advisory checks for offline jobs.

Inspects lsm_job_status and corpus state to emit actionable messages
when configured jobs are stale or not yet run.

Supports both SQLite (sqlite3.Connection) and PostgreSQL (psycopg2)
connections via DB-API 2.0 cursor-based operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from lsm.db.compat import convert_placeholders, fetchone as compat_fetchone
from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames
from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Advisory:
    """A startup advisory message."""

    level: str  # "info" or "warn"
    message: str
    action: str  # CLI command to resolve


def check_job_advisories(
    conn: Any,
    config: Any = None,
    table_names: TableNames | None = None,
) -> List[Advisory]:
    """Check for stale or missing offline jobs."""
    tn = table_names or DEFAULT_TABLE_NAMES
    advisories: List[Advisory] = []

    try:
        advisories.extend(_check_cluster_status(conn, config, table_names=tn))
        advisories.extend(_check_graph_status(conn, config, table_names=tn))
        advisories.extend(_check_finetune_status(conn, config, table_names=tn))
    except Exception as exc:
        logger.debug("Advisory check failed: %s", exc)

    return advisories


def _get_corpus_size(conn: Any, table_names: TableNames | None = None) -> int:
    """Get current corpus size (active chunks)."""
    tn = table_names or DEFAULT_TABLE_NAMES
    try:
        row = compat_fetchone(conn, f"SELECT COUNT(*) FROM {tn.chunks} WHERE is_current = 1")
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _get_job_status(
    conn: Any,
    job_name: str,
    table_names: TableNames | None = None,
) -> Optional[dict]:
    """Get a job's status record."""
    tn = table_names or DEFAULT_TABLE_NAMES
    try:
        sql = convert_placeholders(
            f"SELECT status, completed_at, corpus_size FROM {tn.job_status} WHERE job_name = ?",
            conn,
        )
        row = compat_fetchone(conn, sql, (job_name,))
        if row is None:
            return None
        return {
            "status": row[0],
            "completed_at": row[1],
            "corpus_size": int(row[2]) if row[2] else 0,
        }
    except Exception:
        return None


def _check_cluster_status(
    conn: Any,
    config: Any,
    table_names: TableNames | None = None,
) -> List[Advisory]:
    """Check if clustering needs to be run or updated."""
    advisories: List[Advisory] = []

    cluster_enabled = False
    if config is not None:
        try:
            cluster_enabled = getattr(config.query, "cluster_enabled", False)
        except Exception:
            pass

    tn = table_names or DEFAULT_TABLE_NAMES
    if not cluster_enabled:
        return advisories

    job = _get_job_status(conn, "cluster_build", table_names=tn)
    corpus_size = _get_corpus_size(conn, table_names=tn)

    if job is None:
        if corpus_size > 0:
            advisories.append(
                Advisory(
                    level="info",
                    message=(
                        f"Clustering is enabled but has never been built "
                        f"({corpus_size} chunks in corpus)."
                    ),
                    action="lsm cluster build",
                )
            )
    elif job.get("corpus_size", 0) > 0:
        old_size = job["corpus_size"]
        if corpus_size > old_size * 1.2:
            advisories.append(
                Advisory(
                    level="info",
                    message=(
                        f"Corpus has grown {corpus_size - old_size} chunks since "
                        f"last cluster build ({old_size} → {corpus_size}). "
                        "Consider rebuilding clusters."
                    ),
                    action="lsm cluster build",
                )
            )

    return advisories


def _check_finetune_status(
    conn: Any,
    config: Any,
    table_names: TableNames | None = None,
) -> List[Advisory]:
    """Check if embedding fine-tuning should be run."""
    advisories: List[Advisory] = []

    finetune_enabled = True
    if config is not None:
        try:
            gs = getattr(config, "global_settings", None)
            if gs is not None and hasattr(gs, "finetune_enabled"):
                finetune_enabled = bool(gs.finetune_enabled)
        except Exception:
            pass

    tn = table_names or DEFAULT_TABLE_NAMES
    if not finetune_enabled:
        return advisories

    try:
        row = compat_fetchone(
            conn,
            f"SELECT COUNT(*) FROM {tn.embedding_models} WHERE is_active = 1",
        )
        has_active = int(row[0]) > 0 if row else False
    except Exception:
        return advisories

    if not has_active:
        corpus_size = _get_corpus_size(conn, table_names=tn)
        if corpus_size >= 100:
            advisories.append(
                Advisory(
                    level="info",
                    message=(
                        "No fine-tuned embedding model is active. "
                        f"With {corpus_size} chunks, fine-tuning may improve retrieval quality."
                    ),
                    action="lsm finetune train",
                )
            )

    return advisories


def _check_graph_status(
    conn: Any,
    config: Any,
    table_names: TableNames | None = None,
) -> List[Advisory]:
    """Check whether thematic links were built when graph expansion is enabled."""
    advisories: List[Advisory] = []

    graph_enabled = False
    if config is not None:
        try:
            graph_enabled = bool(getattr(config.query, "graph_expansion_enabled", False))
        except Exception:
            graph_enabled = False

    tn = table_names or DEFAULT_TABLE_NAMES
    if not graph_enabled:
        return advisories

    try:
        row = compat_fetchone(
            conn,
            f"SELECT COUNT(*) FROM {tn.graph_edges} WHERE edge_type = 'thematic'",
        )
        thematic_edges = int(row[0]) if row else 0
    except Exception:
        return advisories

    if thematic_edges <= 0:
        corpus_size = _get_corpus_size(conn, table_names=tn)
        advisories.append(
            Advisory(
                level="info",
                message=(
                    "Graph expansion is enabled but thematic links have not been built"
                    f" ({corpus_size} chunks in corpus)."
                ),
                action="lsm graph build-links",
            )
        )

    return advisories


def record_job_status(
    conn: Any,
    job_name: str,
    status: str = "completed",
    corpus_size: Optional[int] = None,
    table_names: TableNames | None = None,
) -> None:
    """Record a job's completion status."""
    from datetime import datetime, timezone

    from lsm.db.compat import commit as compat_commit
    from lsm.db.compat import execute as compat_execute

    tn = table_names or DEFAULT_TABLE_NAMES
    now = datetime.now(timezone.utc).isoformat()

    upsert_sql = convert_placeholders(
        (
            f"INSERT INTO {tn.job_status} (job_name, status, completed_at, corpus_size) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (job_name) DO UPDATE SET "
            "status = EXCLUDED.status, completed_at = EXCLUDED.completed_at, "
            "corpus_size = EXCLUDED.corpus_size"
        ),
        conn,
    )

    compat_execute(conn, upsert_sql, (job_name, status, now, corpus_size))
    compat_commit(conn)
