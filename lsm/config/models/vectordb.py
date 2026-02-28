"""
Vector database configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_COLLECTION, DEFAULT_VDB_PATH, DEFAULT_VDB_PROVIDER


@dataclass
class VectorDBConfig:
    """
    Configuration for vector database providers.
    """

    provider: str = DEFAULT_VDB_PROVIDER
    """Vector DB provider name (e.g., 'sqlite', 'postgresql')."""

    collection: str = DEFAULT_COLLECTION
    """Collection or namespace name."""

    path: Path = Path(DEFAULT_VDB_PATH)
    """Directory containing the unified ``lsm.db`` file."""

    connection_string: Optional[str] = None
    """PostgreSQL connection string."""

    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    index_type: str = "hnsw"
    """Index type for pgvector (e.g., 'hnsw', 'ivfflat')."""

    pool_size: int = 5
    """Connection pool size for providers that support pooling."""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    @property
    def persist_dir(self) -> Path:
        """Backward-compatible alias for ``path``."""
        return self.path

    @persist_dir.setter
    def persist_dir(self, value: Path | str) -> None:
        self.path = Path(value)

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("vectordb.provider must be set")

        if self.provider == "sqlite":
            if not self.path:
                raise ValueError("vectordb.path is required for sqlite")
            if not self.collection:
                raise ValueError("vectordb.collection is required for sqlite")

        if self.provider == "postgresql":
            if not (self.connection_string or (self.host and self.database and self.user)):
                raise ValueError(
                    "PostgreSQL vectordb requires connection_string or host/database/user"
                )
