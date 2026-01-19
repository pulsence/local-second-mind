"""
Migration utility from ChromaDB to PostgreSQL + pgvector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
) -> Dict[str, Any]:
    """
    Migrate vectors from a ChromaDB collection to PostgreSQL.

    Args:
        persist_dir: ChromaDB persist directory
        collection_name: ChromaDB collection name
        postgres_config: VectorDBConfig with provider='postgresql'
        batch_size: Batch size for reads/writes

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
    collection = ChromaDBProvider(chroma_config).get_collection()
    provider = PostgreSQLProvider(postgres_config)

    total = collection.count()
    migrated = 0
    offset = 0

    while True:
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = results.get("ids", [])
        if not ids:
            break

        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        embeddings = results.get("embeddings", []) or []

        provider.add_chunks(ids, documents, metadatas, embeddings)

        migrated += len(ids)
        offset += len(ids)
        logger.info(f"Migrated {migrated}/{total} chunks...")

    return {
        "source": "chromadb",
        "target": "postgresql",
        "collection": collection_name,
        "total": total,
        "migrated": migrated,
    }
