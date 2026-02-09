"""Live PostgreSQL + pgvector provider tests."""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.live, pytest.mark.live_vectordb]


def _encode(real_embedder, texts: list[str]) -> list[list[float]]:
    vectors = real_embedder.encode(texts, convert_to_numpy=True)
    return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)


def test_live_postgresql_crud_query_filter_update_delete(
    real_postgresql_provider,
    real_embedder,
) -> None:
    docs = [
        "Epistemology studies justified belief and evidence.",
        "Retrieval pipelines combine semantic search and reranking.",
        "Citation-aware synthesis improves answer trust.",
        "Local-first architecture keeps documents private on device.",
        "Metadata filtering narrows candidate chunks by topic.",
        "Incremental ingest avoids recomputing unchanged embeddings.",
        "Chunk overlap can improve recall across section boundaries.",
        "Grounded answers should preserve citation traceability.",
    ]
    ids = [f"pg_live_doc_{idx}" for idx in range(len(docs))]
    metadatas = [
        {
            "topic": "philosophy" if idx % 2 == 0 else "systems",
            "source_path": f"/docs/live_{idx}.md",
            "chunk_index": idx,
        }
        for idx in range(len(docs))
    ]
    embeddings = _encode(real_embedder, docs)

    real_postgresql_provider.add_chunks(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    assert real_postgresql_provider.count() == len(docs)

    query_vector = _encode(
        real_embedder,
        ["How do semantic retrieval and reranking work together?"],
    )[0]
    query_result = real_postgresql_provider.query(query_vector, top_k=3)
    assert len(query_result.ids) == 3
    assert len(query_result.documents) == 3
    assert len(query_result.metadatas) == 3
    assert all(distance is None or distance >= 0.0 for distance in query_result.distances)

    filtered = real_postgresql_provider.get(
        filters={"topic": {"$eq": "philosophy"}},
        include=["documents", "metadatas"],
        limit=10,
    )
    assert filtered.ids
    assert filtered.metadatas is not None
    assert all(meta["topic"] == "philosophy" for meta in filtered.metadatas)

    paged = real_postgresql_provider.get(limit=4, offset=2, include=["metadatas"])
    assert len(paged.ids) == 4
    assert paged.metadatas is not None
    assert len(paged.metadatas) == 4

    target_id = ids[0]
    real_postgresql_provider.update_metadatas(
        ids=[target_id],
        metadatas=[{"topic": "updated", "source_path": "/docs/updated.md", "chunk_index": 999}],
    )
    updated = real_postgresql_provider.get(ids=[target_id], include=["metadatas"])
    assert updated.metadatas is not None
    assert updated.metadatas[0]["topic"] == "updated"

    real_postgresql_provider.delete_by_id([target_id])
    assert real_postgresql_provider.get(ids=[target_id], include=["metadatas"]).ids == []

    before_delete = real_postgresql_provider.count()
    real_postgresql_provider.delete_by_filter({"topic": "systems"})
    after_delete = real_postgresql_provider.count()
    assert after_delete < before_delete

    health = real_postgresql_provider.health_check()
    assert health["status"] == "ok"
    assert health["provider"] == "postgresql"


def test_live_postgresql_delete_all_and_stats(
    real_postgresql_provider,
    real_embedder,
) -> None:
    docs = [
        "Migration should preserve vectors and metadata.",
        "PostgreSQL pgvector supports cosine distance search.",
        "Chunk provenance should remain auditable after migration.",
    ]
    ids = [f"pg_stats_{idx}" for idx in range(len(docs))]
    metadatas = [
        {"source_path": f"/stats/doc_{idx}.md", "chunk_index": idx}
        for idx in range(len(docs))
    ]
    embeddings = _encode(real_embedder, docs)

    real_postgresql_provider.add_chunks(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    assert real_postgresql_provider.count() == len(docs)

    stats = real_postgresql_provider.get_stats()
    assert stats["provider"] == "postgresql"
    assert stats["collection"] == real_postgresql_provider.config.collection
    assert stats["count"] == len(docs)

    deleted = real_postgresql_provider.delete_all()
    assert deleted == len(docs)
    assert real_postgresql_provider.count() == 0
