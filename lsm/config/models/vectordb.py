"""
Vector database configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_CHROMA_HNSW_SPACE, DEFAULT_COLLECTION, DEFAULT_VDB_PROVIDER


@dataclass
class VectorDBConfig:
    """
    Configuration for vector database providers.
    """

    provider: str = DEFAULT_VDB_PROVIDER
    """Vector DB provider name (e.g., 'chromadb', 'postgresql')."""

    collection: str = DEFAULT_COLLECTION
    """Collection or namespace name."""

    persist_dir: Path = Path(".chroma")
    """ChromaDB persistence directory (Chroma-only)."""

    chroma_hnsw_space: str = DEFAULT_CHROMA_HNSW_SPACE
    """ChromaDB HNSW space (e.g., 'cosine')."""

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
        if isinstance(self.persist_dir, str):
            self.persist_dir = Path(self.persist_dir)

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("vectordb.provider must be set")

        if self.provider == "chromadb":
            if not self.persist_dir:
                raise ValueError("vectordb.persist_dir is required for ChromaDB")
            if not self.collection:
                raise ValueError("vectordb.collection is required for ChromaDB")

        if self.provider == "postgresql":
            if not (self.connection_string or (self.host and self.database and self.user)):
                raise ValueError(
                    "PostgreSQL vectordb requires connection_string or host/database/user"
                )
