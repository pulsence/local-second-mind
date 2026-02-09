"""Live migration tests from ChromaDB to PostgreSQL + pgvector."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.chromadb import ChromaDBProvider
from lsm.vectordb.factory import create_vectordb_provider
from lsm.vectordb.migrations.chromadb_to_postgres import migrate_chromadb_to_postgres


pytestmark = [pytest.mark.live, pytest.mark.live_vectordb]


def _encode(real_embedder, texts: list[str]) -> list[list[float]]:
    vectors = real_embedder.encode(texts, convert_to_numpy=True)
    return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)


def test_live_migration_chromadb_to_postgres_roundtrip(
    tmp_path: Path,
    real_embedder,
    live_postgres_connection_string: str,
) -> None:
    source_collection = f"live_migration_src_{uuid4().hex[:8]}"
    target_collection = f"live_migration_pg_{uuid4().hex[:8]}"
    chroma_dir = tmp_path / ".chroma"

    source = ChromaDBProvider(
        VectorDBConfig(
            provider="chromadb",
            persist_dir=chroma_dir,
            collection=source_collection,
        )
    )
    assert source.is_available() is True

    docs = [
        "Citations improve trust because claims can be inspected.",
        "Metadata fields preserve provenance and retrieval auditability.",
        "Incremental ingest avoids redundant embedding work.",
        "Reranking can improve precision when candidate sets are noisy.",
        "Chunk boundaries affect semantic recall and context continuity.",
        "Local-first systems prioritize user control and privacy.",
    ]
    ids = [f"mig_{idx}" for idx in range(len(docs))]
    metadatas = [
        {
            "topic": "migration",
            "source_path": f"/migration/doc_{idx}.md",
            "chunk_index": idx,
        }
        for idx in range(len(docs))
    ]
    embeddings = _encode(real_embedder, docs)

    source.add_chunks(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    assert source.count() == len(docs)

    postgres_config = VectorDBConfig(
        provider="postgresql",
        connection_string=live_postgres_connection_string,
        collection=target_collection,
        pool_size=2,
    )
    target = create_vectordb_provider(postgres_config)
    if not target.is_available():
        pytest.skip("PostgreSQL provider is unavailable in this environment")

    progress_calls: list[tuple[int, int]] = []
    try:
        result = migrate_chromadb_to_postgres(
            persist_dir=chroma_dir,
            collection_name=source_collection,
            postgres_config=postgres_config,
            batch_size=2,
            progress_callback=lambda migrated, total: progress_calls.append((migrated, total)),
        )

        assert result["source"] == "chromadb"
        assert result["target"] == "postgresql"
        assert result["collection"] == source_collection
        assert result["total"] == len(docs)
        assert result["migrated"] == len(docs)
        assert progress_calls
        assert progress_calls[-1] == (len(docs), len(docs))

        assert target.count() == len(docs)

        migrated_rows = target.get(
            limit=len(docs),
            include=["documents", "metadatas", "embeddings"],
        )
        assert len(migrated_rows.ids) == len(docs)
        assert migrated_rows.documents is not None
        assert migrated_rows.metadatas is not None
        assert migrated_rows.embeddings is not None

        migrated_docs = set(migrated_rows.documents)
        assert set(docs).issubset(migrated_docs)
        assert all(meta.get("topic") == "migration" for meta in migrated_rows.metadatas)
        assert all(len(emb) > 0 for emb in migrated_rows.embeddings)

        query_vector = _encode(
            real_embedder,
            ["How do provenance metadata and citations help trust?"],
        )[0]
        query_result = target.query(query_vector, top_k=3)
        assert len(query_result.ids) == 3
        assert any("citation" in doc.lower() or "metadata" in doc.lower() for doc in query_result.documents)
    finally:
        try:
            target.delete_all()
        except Exception:
            pass
        pool = getattr(target, "_pool", None)
        if pool is not None:
            try:
                pool.closeall()
            except Exception:
                pass
