"""Integration tests for real embedding model behavior."""

from __future__ import annotations

import math

import pytest


pytestmark = pytest.mark.integration


def _as_vector_list(vectors) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        converted = vectors.tolist()
    else:
        converted = vectors
    return [list(row) for row in converted]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


def test_real_embedder_dimension_and_shape(real_embedder) -> None:
    texts = [
        "Knowledge graphs improve retrieval explainability.",
        "Citation grounding improves trust in synthesized answers.",
        "Incremental ingest reduces redundant embedding work.",
    ]
    vectors = _as_vector_list(real_embedder.encode(texts, convert_to_numpy=True))

    assert len(vectors) == len(texts)
    assert all(len(vec) > 0 for vec in vectors)

    dim = len(vectors[0])
    assert all(len(vec) == dim for vec in vectors)

    model_name = str(getattr(real_embedder, "model_name_or_path", ""))
    if model_name.endswith("all-MiniLM-L6-v2"):
        assert dim == 384


def test_semantically_similar_texts_have_lower_cosine_distance(real_embedder) -> None:
    similar_a = (
        "Epistemology examines how beliefs are justified with evidence and revision."
    )
    similar_b = (
        "Justified belief in epistemology depends on evidence and ongoing revision."
    )
    unrelated = "This recipe uses cumin, onion, and olive oil for lentil soup."

    vectors = _as_vector_list(
        real_embedder.encode([similar_a, similar_b, unrelated], convert_to_numpy=True)
    )
    dist_similar = _cosine_distance(vectors[0], vectors[1])
    dist_unrelated = _cosine_distance(vectors[0], vectors[2])

    assert dist_similar < dist_unrelated


def test_batch_embedding_consistency(real_embedder) -> None:
    texts = [
        "Local-first systems keep private data on device.",
        "Reranking can improve retrieval precision.",
        "Citations should point back to source passages.",
    ]

    batch_vectors = _as_vector_list(real_embedder.encode(texts, convert_to_numpy=True))
    single_vectors = []
    for text in texts:
        one = real_embedder.encode([text], convert_to_numpy=True)
        single_vectors.append(_as_vector_list(one)[0])

    assert len(batch_vectors) == len(single_vectors)

    for batch_vec, single_vec in zip(batch_vectors, single_vectors):
        # Same model, same text: vectors should match closely.
        max_delta = max(abs(a - b) for a, b in zip(batch_vec, single_vec))
        assert max_delta < 1e-5
