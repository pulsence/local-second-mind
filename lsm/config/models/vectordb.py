"""
Database and vector configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_COLLECTION, DEFAULT_VDB_PATH, DEFAULT_VDB_PROVIDER


@dataclass
class VectorConfig:
    """Vector-specific settings (provider, collection, index, pool)."""

    provider: str = DEFAULT_VDB_PROVIDER
    """Vector DB provider name (e.g., 'sqlite', 'postgresql')."""

    collection: str = DEFAULT_COLLECTION
    """Collection or namespace name."""

    index_type: str = "hnsw"
    """Index type for pgvector (e.g., 'hnsw', 'ivfflat')."""

    pool_size: int = 5
    """Connection pool size for providers that support pooling."""

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("db.vector.provider must be set")


@dataclass
class DBConfig:
    """Database-level settings.

    Owns both connection settings and a nested ``VectorConfig`` for
    vector-specific parameters.
    """

    table_prefix: str = "lsm_"
    """Prefix for all SQL table names."""

    path: Path = Path(DEFAULT_VDB_PATH)
    """Directory containing the unified ``lsm.db`` file."""

    connection_string: Optional[str] = None
    """PostgreSQL connection string."""

    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    vector: VectorConfig = field(default_factory=VectorConfig)
    """Nested vector-specific configuration."""

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

    # ------------------------------------------------------------------
    # Delegation properties (convenience access to vector sub-fields)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return self.vector.provider

    @provider.setter
    def provider(self, value: str) -> None:
        self.vector.provider = value

    @property
    def collection(self) -> str:
        return self.vector.collection

    @collection.setter
    def collection(self, value: str) -> None:
        self.vector.collection = value

    @property
    def index_type(self) -> str:
        return self.vector.index_type

    @index_type.setter
    def index_type(self, value: str) -> None:
        self.vector.index_type = value

    @property
    def pool_size(self) -> int:
        return self.vector.pool_size

    @pool_size.setter
    def pool_size(self, value: int) -> None:
        self.vector.pool_size = value

    @property
    def persist_dir(self) -> Path:
        """Backward-compatible alias for ``path``."""
        return self.path

    @persist_dir.setter
    def persist_dir(self, value: Path | str) -> None:
        self.path = Path(value)

    def validate(self) -> None:
        import re

        if not re.match(r"^[a-zA-Z0-9_]*$", self.table_prefix):
            raise ValueError(
                f"db.table_prefix must contain only alphanumeric characters "
                f"and underscores, got {self.table_prefix!r}"
            )

        self.vector.validate()

        if self.provider == "sqlite":
            if not self.path:
                raise ValueError("db.path is required for sqlite")
            if not self.collection:
                raise ValueError("db.vector.collection is required for sqlite")

        if self.provider == "postgresql":
            if not (self.connection_string or (self.host and self.database and self.user)):
                raise ValueError(
                    "PostgreSQL db requires connection_string or host/database/user"
                )


# ---------------------------------------------------------------------------
# Convenience __init__ wrapper: accept provider/collection/index_type/pool_size
# as top-level kwargs and forward them to the nested VectorConfig.
# ---------------------------------------------------------------------------
_dc_init = DBConfig.__init__
_UNSET = object()


def _compat_init(
    self,
    *args,
    provider=_UNSET,
    collection=_UNSET,
    index_type=_UNSET,
    pool_size=_UNSET,
    **kwargs,
):
    _dc_init(self, *args, **kwargs)
    if provider is not _UNSET:
        self.vector.provider = provider
    if collection is not _UNSET:
        self.vector.collection = collection
    if index_type is not _UNSET:
        self.vector.index_type = index_type
    if pool_size is not _UNSET:
        self.vector.pool_size = pool_size


DBConfig.__init__ = _compat_init
