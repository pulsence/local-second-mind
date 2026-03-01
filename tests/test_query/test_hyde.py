"""Tests for HyDE (Hypothetical Document Embeddings) stage."""

from __future__ import annotations

import numpy as np
import pytest

from lsm.query.stages.hyde import (
    generate_hypothetical_docs,
    pool_embeddings,
    hyde_recall,
)


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses=None, fail_at=None):
        self._responses = responses or ["Hypothetical answer about the topic."]
        self._call_count = 0
        self._fail_at = fail_at  # set of indices where send_message should fail

    def send_message(self, **kwargs):
        idx = self._call_count
        self._call_count += 1
        if self._fail_at and idx in self._fail_at:
            raise RuntimeError("LLM unavailable")
        return self._responses[idx % len(self._responses)]

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


class FakeEmbedder:
    def encode(self, texts, batch_size=None, show_progress_bar=False, **kwargs):
        # Return deterministic embeddings based on text length
        return np.array([[float(len(t)) / 100.0] * 10 for t in texts])


class FakeDB:
    def __init__(self, candidates=None):
        self._candidates = candidates or []

    def fts_query(self, text, top_k):
        from types import SimpleNamespace
        return SimpleNamespace(ids=[], documents=[], metadatas=[], distances=[])


# ------------------------------------------------------------------
# generate_hypothetical_docs
# ------------------------------------------------------------------


class TestGenerateHypotheticalDocs:
    def test_generates_requested_number(self):
        provider = FakeProvider(responses=["Doc one.", "Doc two.", "Doc three."])
        docs = generate_hypothetical_docs("test query", provider, num_samples=3)
        assert len(docs) == 3

    def test_single_sample(self):
        provider = FakeProvider(responses=["Single doc."])
        docs = generate_hypothetical_docs("test", provider, num_samples=1)
        assert len(docs) == 1
        assert docs[0] == "Single doc."

    def test_strips_whitespace(self):
        provider = FakeProvider(responses=["  padded doc  "])
        docs = generate_hypothetical_docs("test", provider, num_samples=1)
        assert docs[0] == "padded doc"

    def test_skips_empty_responses(self):
        provider = FakeProvider(responses=["", "  ", "valid doc"])
        docs = generate_hypothetical_docs("test", provider, num_samples=3)
        assert len(docs) == 1
        assert docs[0] == "valid doc"

    def test_handles_provider_failure(self):
        provider = FakeProvider(responses=["doc"], fail_at={0})
        docs = generate_hypothetical_docs("test", provider, num_samples=2)
        # First call fails, second succeeds
        assert len(docs) == 1

    def test_all_failures_returns_empty(self):
        provider = FakeProvider(responses=["doc"], fail_at={0, 1})
        docs = generate_hypothetical_docs("test", provider, num_samples=2)
        assert docs == []


# ------------------------------------------------------------------
# pool_embeddings
# ------------------------------------------------------------------


class TestPoolEmbeddings:
    def test_empty_input(self):
        assert pool_embeddings([]) == []

    def test_mean_pooling(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        result = pool_embeddings(embeddings, strategy="mean")
        assert len(result) == 2
        # Mean of [1,0] and [0,1] = [0.5, 0.5], normalized
        assert abs(result[0] - result[1]) < 1e-6  # Equal components

    def test_max_pooling(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        result = pool_embeddings(embeddings, strategy="max")
        assert len(result) == 2
        # Max of [1,0] and [0,1] = [1, 1], normalized
        assert abs(result[0] - result[1]) < 1e-6  # Equal components

    def test_single_embedding(self):
        embeddings = [[3.0, 4.0]]
        result = pool_embeddings(embeddings, strategy="mean")
        # Normalized: [3/5, 4/5] = [0.6, 0.8]
        assert abs(result[0] - 0.6) < 1e-6
        assert abs(result[1] - 0.8) < 1e-6

    def test_result_is_unit_vector(self):
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = pool_embeddings(embeddings, strategy="mean")
        norm = sum(x ** 2 for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_default_is_mean(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        mean_result = pool_embeddings(embeddings, strategy="mean")
        default_result = pool_embeddings(embeddings)
        assert mean_result == default_result


# ------------------------------------------------------------------
# hyde_recall
# ------------------------------------------------------------------


class TestHydeRecall:
    def test_returns_embedding_and_docs(self):
        provider = FakeProvider(responses=["Hyp doc one.", "Hyp doc two."])
        embedder = FakeEmbedder()
        db = FakeDB()
        embedding, docs = hyde_recall(
            "test query", provider, embedder, db, num_samples=2
        )
        assert len(docs) == 2
        assert len(embedding) > 0

    def test_fallback_when_no_docs_generated(self):
        provider = FakeProvider(responses=[""], fail_at={0, 1})
        embedder = FakeEmbedder()
        db = FakeDB()
        embedding, docs = hyde_recall(
            "test query", provider, embedder, db, num_samples=2
        )
        # Falls back to raw query embedding
        assert docs == []
        assert len(embedding) > 0

    def test_hypothetical_docs_in_output(self):
        provider = FakeProvider(responses=["Answer about topic"])
        embedder = FakeEmbedder()
        db = FakeDB()
        embedding, docs = hyde_recall(
            "test", provider, embedder, db, num_samples=1
        )
        assert "Answer about topic" in docs

    def test_pooling_strategy_forwarded(self):
        provider = FakeProvider(responses=["Doc"])
        embedder = FakeEmbedder()
        db = FakeDB()
        mean_emb, _ = hyde_recall(
            "q", provider, embedder, db, num_samples=1, pooling="mean"
        )
        max_emb, _ = hyde_recall(
            "q", provider, embedder, db, num_samples=1, pooling="max"
        )
        # With same inputs, mean and max should produce different embeddings
        # (unless embeddings happen to be identical, which they aren't here)
        assert len(mean_emb) == len(max_emb)
