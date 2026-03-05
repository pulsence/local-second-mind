"""
Embedding model registry.

Tracks fine-tuned models in lsm_embedding_models table with
active model state management.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from lsm.db.compat import commit as compat_commit, execute, fetchall, fetchone
from lsm.db.tables import TableNames, DEFAULT_TABLE_NAMES
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
    conn: Any,
    model_id: str,
    base_model: str,
    path: str,
    dimension: int,
    table_names: TableNames = DEFAULT_TABLE_NAMES,
) -> EmbeddingModelEntry:
    """Register a fine-tuned model in the database.

    Args:
        conn: Database connection (SQLite or PostgreSQL).
        model_id: Unique model identifier.
        base_model: Base model name (e.g., 'all-MiniLM-L6-v2').
        path: Filesystem path to the fine-tuned model.
        dimension: Embedding dimension.

    Returns:
        The registered model entry.
    """
    tn = table_names
    now = datetime.now(timezone.utc).isoformat()
    execute(
        conn,
        f"""INSERT INTO {tn.embedding_models}
            (model_id, base_model, path, dimension, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(model_id) DO UPDATE SET
                base_model = excluded.base_model,
                path = excluded.path,
                dimension = excluded.dimension,
                created_at = excluded.created_at""",
        (model_id, base_model, path, dimension, now),
    )
    compat_commit(conn)
    return EmbeddingModelEntry(
        model_id=model_id,
        base_model=base_model,
        path=path,
        dimension=dimension,
        created_at=now,
        is_active=False,
    )


def set_active_model(
    conn: Any,
    model_id: str,
    table_names: TableNames = DEFAULT_TABLE_NAMES,
) -> None:
    """Set a model as the active embedding model.

    Deactivates all other models first.

    Args:
        conn: Database connection (SQLite or PostgreSQL).
        model_id: Model ID to activate.
    """
    tn = table_names
    execute(conn, f"UPDATE {tn.embedding_models} SET is_active = 0")
    execute(
        conn,
        f"UPDATE {tn.embedding_models} SET is_active = 1 WHERE model_id = ?",
        (model_id,),
    )
    compat_commit(conn)


def get_active_model(
    conn: Any,
    table_names: TableNames = DEFAULT_TABLE_NAMES,
) -> Optional[EmbeddingModelEntry]:
    """Get the currently active embedding model.

    Returns:
        Active model entry, or None if no model is active.
    """
    tn = table_names
    row = fetchone(
        conn,
        "SELECT model_id, base_model, path, dimension, created_at, is_active "
        f"FROM {tn.embedding_models} WHERE is_active = 1",
    )
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


def list_models(
    conn: Any,
    table_names: TableNames = DEFAULT_TABLE_NAMES,
) -> List[EmbeddingModelEntry]:
    """List all registered embedding models.

    Returns:
        List of model entries.
    """
    tn = table_names
    rows = fetchall(
        conn,
        "SELECT model_id, base_model, path, dimension, created_at, is_active "
        f"FROM {tn.embedding_models} ORDER BY created_at DESC",
    )
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


def delete_model(
    conn: Any,
    model_id: str,
    table_names: TableNames = DEFAULT_TABLE_NAMES,
) -> bool:
    """Delete a model from the registry.

    Args:
        conn: Database connection (SQLite or PostgreSQL).
        model_id: Model ID to delete.

    Returns:
        True if a model was deleted.
    """
    tn = table_names
    cursor = execute(
        conn,
        f"DELETE FROM {tn.embedding_models} WHERE model_id = ?",
        (model_id,),
    )
    compat_commit(conn)
    return cursor.rowcount > 0
