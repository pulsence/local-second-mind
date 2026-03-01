"""
Startup advisory checks for offline jobs.

Inspects lsm_job_status and corpus state to emit actionable messages
when configured jobs are stale or not yet run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional

from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Advisory:
    """A startup advisory message."""

    level: str  # "info" or "warn"
    message: str
    action: str  # CLI command to resolve


def check_job_advisories(
    conn: sqlite3.Connection,
    config: Any = None,
) -> List[Advisory]:
    """Check for stale or missing offline jobs.

    Inspects lsm_job_status and corpus state to produce advisory
    messages for jobs that should be run.

    Args:
        conn: SQLite connection with application tables.
        config: Optional LSMConfig for checking enabled features.

    Returns:
        List of Advisory objects (may be empty).
    """
    advisories: List[Advisory] = []

    try:
        advisories.extend(_check_cluster_status(conn, config))
        advisories.extend(_check_finetune_status(conn, config))
    except Exception as exc:
        logger.debug("Advisory check failed: %s", exc)

    return advisories


def _get_corpus_size(conn: sqlite3.Connection) -> int:
    """Get current corpus size (active chunks)."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM lsm_chunks WHERE is_current = 1"
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _get_job_status(
    conn: sqlite3.Connection, job_name: str
) -> Optional[dict]:
    """Get a job's status record."""
    try:
        row = conn.execute(
            "SELECT status, completed_at, corpus_size FROM lsm_job_status WHERE job_name = ?",
            (job_name,),
        ).fetchone()
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
    conn: sqlite3.Connection, config: Any
) -> List[Advisory]:
    """Check if clustering needs to be run or updated."""
    advisories: List[Advisory] = []

    # Check if clustering is enabled in config
    cluster_enabled = False
    if config is not None:
        try:
            cluster_enabled = getattr(config.query, "cluster_enabled", False)
        except Exception:
            pass

    if not cluster_enabled:
        return advisories

    job = _get_job_status(conn, "cluster_build")
    corpus_size = _get_corpus_size(conn)

    if job is None:
        # Never run
        if corpus_size > 0:
            advisories.append(Advisory(
                level="info",
                message=(
                    f"Clustering is enabled but has never been built "
                    f"({corpus_size} chunks in corpus)."
                ),
                action="lsm cluster build",
            ))
    elif job.get("corpus_size", 0) > 0:
        # Check staleness (>20% corpus growth)
        old_size = job["corpus_size"]
        if corpus_size > old_size * 1.2:
            advisories.append(Advisory(
                level="info",
                message=(
                    f"Corpus has grown {corpus_size - old_size} chunks since "
                    f"last cluster build ({old_size} → {corpus_size}). "
                    f"Consider rebuilding clusters."
                ),
                action="lsm cluster build",
            ))

    return advisories


def _check_finetune_status(
    conn: sqlite3.Connection, config: Any
) -> List[Advisory]:
    """Check if embedding fine-tuning should be run."""
    advisories: List[Advisory] = []

    # Check for active fine-tuned model
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM lsm_embedding_models WHERE is_active = 1"
        ).fetchone()
        has_active = int(row[0]) > 0 if row else False
    except Exception:
        # Table might not exist
        return advisories

    if not has_active:
        corpus_size = _get_corpus_size(conn)
        if corpus_size >= 100:
            advisories.append(Advisory(
                level="info",
                message=(
                    f"No fine-tuned embedding model is active. "
                    f"With {corpus_size} chunks, fine-tuning may improve retrieval quality."
                ),
                action="lsm finetune train",
            ))

    return advisories


def record_job_status(
    conn: sqlite3.Connection,
    job_name: str,
    status: str = "completed",
    corpus_size: Optional[int] = None,
) -> None:
    """Record a job's completion status.

    Args:
        conn: SQLite connection.
        job_name: Job identifier (e.g., 'cluster_build').
        status: Job status string.
        corpus_size: Current corpus size at completion time.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT OR REPLACE INTO lsm_job_status
           (job_name, status, completed_at, corpus_size)
           VALUES (?, ?, ?, ?)""",
        (job_name, status, now, corpus_size),
    )
    conn.commit()
