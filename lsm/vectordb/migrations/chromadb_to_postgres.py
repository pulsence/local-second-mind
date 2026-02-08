"""
Migration utility from ChromaDB to PostgreSQL + pgvector.

Uses the provider interface for both source and target — no raw ChromaDB APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from lsm.logging import get_logger
from lsm.config.models import VectorDBConfig
from lsm.vectordb.chromadb import ChromaDBProvider
from lsm.vectordb.postgresql import PostgreSQLProvider

logger = get_logger(__name__)


def migrate_chromadb_to_postgres(
    persist_dir: Path,
    collection_name: str,
    postgres_config: VectorDBConfig,
    batch_size: int = 2000,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Migrate vectors from a ChromaDB collection to PostgreSQL.

    Args:
        persist_dir: ChromaDB persist directory
        collection_name: ChromaDB collection name
        postgres_config: VectorDBConfig with provider='postgresql'
        batch_size: Batch size for reads/writes
        progress_callback: Optional callback(migrated, total) for progress

    Returns:
        Summary dict with counts
    """
    if postgres_config.provider != "postgresql":
        raise ValueError("postgres_config.provider must be 'postgresql'")

    logger.info(f"Starting migration from ChromaDB '{collection_name}' to PostgreSQL")

    chroma_config = VectorDBConfig(
        provider="chromadb",
        persist_dir=persist_dir,
        collection=collection_name,
    )
    source = ChromaDBProvider(chroma_config)
    target = PostgreSQLProvider(postgres_config)

    total = source.count()
    migrated = 0
    offset = 0

    while True:
        result = source.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = result.ids
        if not ids:
            break

        documents = result.documents or []
        metadatas = result.metadatas or []
        embeddings = result.embeddings or []

        target.add_chunks(ids, documents, metadatas, embeddings)

        migrated += len(ids)
        offset += len(ids)
        logger.info(f"Migrated {migrated}/{total} chunks...")
        if progress_callback:
            progress_callback(migrated, total)

    logger.info(f"Migration complete: {migrated}/{total} chunks migrated")

    return {
        "source": "chromadb",
        "target": "postgresql",
        "collection": collection_name,
        "total": total,
        "migrated": migrated,
    }
