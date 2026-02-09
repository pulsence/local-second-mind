"""Integration tests for real ChromaDB provider operations."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _encode(real_embedder, texts: list[str]) -> list[list[float]]:
    vectors = real_embedder.encode(texts, convert_to_numpy=True)
    return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)


def test_real_chromadb_query_filter_update_delete(
    real_chromadb_provider,
    real_embedder,
) -> None:
    docs = [
        "Epistemology studies justified belief and evidence.",
        "Retrieval pipelines combine semantic search and reranking.",
        "Citation-aware synthesis improves answer trust.",
        "Local-first architecture keeps documents private on device.",
        "Metadata filtering narrows candidate chunks by topic.",
        "Incremental ingest avoids recomputing unchanged embeddings.",
    ]
    ids = [f"real_doc_{idx}" for idx in range(len(docs))]
    metadatas = [
        {"topic": "philosophy", "source_path": f"/docs/doc_{idx}.md", "chunk_index": idx}
        if idx % 2 == 0
        else {"topic": "systems", "source_path": f"/docs/doc_{idx}.md", "chunk_index": idx}
        for idx in range(len(docs))
    ]
    embeddings = _encode(real_embedder, docs)

    real_chromadb_provider.add_chunks(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    assert real_chromadb_provider.count() == len(docs)

    query_vector = _encode(
        real_embedder,
        ["How do semantic retrieval and reranking work together?"],
    )[0]
    query_result = real_chromadb_provider.query(query_vector, top_k=3)
    assert len(query_result.ids) == 3
    assert len(query_result.documents) == 3
    assert any("retrieval" in doc.lower() for doc in query_result.documents)

    filtered = real_chromadb_provider.get(
        filters={"topic": "philosophy"},
        limit=10,
        include=["documents", "metadatas"],
    )
    assert filtered.ids
    assert filtered.metadatas is not None
    assert all(meta["topic"] == "philosophy" for meta in filtered.metadatas)

    target_id = ids[0]
    real_chromadb_provider.update_metadatas(
        ids=[target_id],
        metadatas=[{"topic": "updated", "source_path": "/docs/updated.md"}],
    )
    updated = real_chromadb_provider.get(ids=[target_id], include=["metadatas"])
    assert updated.metadatas is not None
    assert updated.metadatas[0]["topic"] == "updated"

    before_delete = real_chromadb_provider.count()
    real_chromadb_provider.delete_by_filter({"topic": "systems"})
    after_delete = real_chromadb_provider.count()
    assert after_delete < before_delete

    health = real_chromadb_provider.health_check()
    assert health["status"] == "ok"


def test_real_chromadb_pagination_and_bulk_delete(
    real_chromadb_provider,
    real_embedder,
) -> None:
    total_docs = 120
    docs = [
        (
            f"Chunk {idx}: retrieval stability benchmark text with topic "
            f"{'epistemology' if idx % 3 == 0 else 'infrastructure'}."
        )
        for idx in range(total_docs)
    ]
    ids = [f"bulk_doc_{idx}" for idx in range(total_docs)]
    metadatas = [
        {
            "topic": "epistemology" if idx % 3 == 0 else "infrastructure",
            "source_path": f"/bulk/source_{idx // 10}.md",
            "chunk_index": idx,
        }
        for idx in range(total_docs)
    ]
    embeddings = _encode(real_embedder, docs)

    real_chromadb_provider.add_chunks(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    assert real_chromadb_provider.count() == total_docs

    page = real_chromadb_provider.get(limit=20, offset=40, include=["metadatas"])
    assert len(page.ids) == 20
    assert page.metadatas is not None
    assert len(page.metadatas) == 20

    deleted_count = real_chromadb_provider.delete_all()
    assert deleted_count == total_docs
    assert real_chromadb_provider.count() == 0
