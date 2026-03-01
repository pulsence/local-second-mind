"""
Embedding model registry.

Tracks fine-tuned models in lsm_embedding_models table with
active model state management.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingModelEntry:
    """A registered embedding model."""

    model_id: str
    base_model: str
    path: str
    dimension: int
    created_at: str
    is_active: bool


def register_model(
    conn: sqlite3.Connection,
    model_id: str,
    base_model: str,
    path: str,
    dimension: int,
) -> EmbeddingModelEntry:
    """Register a fine-tuned model in the database.

    Args:
        conn: SQLite connection.
        model_id: Unique model identifier.
        base_model: Base model name (e.g., 'all-MiniLM-L6-v2').
        path: Filesystem path to the fine-tuned model.
        dimension: Embedding dimension.

    Returns:
        The registered model entry.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO lsm_embedding_models
           (model_id, base_model, path, dimension, created_at, is_active)
           VALUES (?, ?, ?, ?, ?, 0)""",
        (model_id, base_model, path, dimension, now),
    )
    conn.commit()
    return EmbeddingModelEntry(
        model_id=model_id,
        base_model=base_model,
        path=path,
        dimension=dimension,
        created_at=now,
        is_active=False,
    )


def set_active_model(conn: sqlite3.Connection, model_id: str) -> None:
    """Set a model as the active embedding model.

    Deactivates all other models first.

    Args:
        conn: SQLite connection.
        model_id: Model ID to activate.
    """
    conn.execute("UPDATE lsm_embedding_models SET is_active = 0")
    conn.execute(
        "UPDATE lsm_embedding_models SET is_active = 1 WHERE model_id = ?",
        (model_id,),
    )
    conn.commit()


def get_active_model(conn: sqlite3.Connection) -> Optional[EmbeddingModelEntry]:
    """Get the currently active embedding model.

    Returns:
        Active model entry, or None if no model is active.
    """
    row = conn.execute(
        "SELECT model_id, base_model, path, dimension, created_at, is_active "
        "FROM lsm_embedding_models WHERE is_active = 1"
    ).fetchone()
    if row is None:
        return None
    return EmbeddingModelEntry(
        model_id=row[0],
        base_model=row[1],
        path=row[2],
        dimension=row[3],
        created_at=row[4],
        is_active=bool(row[5]),
    )


def list_models(conn: sqlite3.Connection) -> List[EmbeddingModelEntry]:
    """List all registered embedding models.

    Returns:
        List of model entries.
    """
    rows = conn.execute(
        "SELECT model_id, base_model, path, dimension, created_at, is_active "
        "FROM lsm_embedding_models ORDER BY created_at DESC"
    ).fetchall()
    return [
        EmbeddingModelEntry(
            model_id=row[0],
            base_model=row[1],
            path=row[2],
            dimension=row[3],
            created_at=row[4],
            is_active=bool(row[5]),
        )
        for row in rows
    ]


def delete_model(conn: sqlite3.Connection, model_id: str) -> bool:
    """Delete a model from the registry.

    Args:
        conn: SQLite connection.
        model_id: Model ID to delete.

    Returns:
        True if a model was deleted.
    """
    cursor = conn.execute(
        "DELETE FROM lsm_embedding_models WHERE model_id = ?",
        (model_id,),
    )
    conn.commit()
    return cursor.rowcount > 0
