"""Tests for retrieval profile routing in RetrievalPipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from lsm.query.pipeline import RetrievalPipeline
from lsm.query.pipeline_types import QueryRequest, ScoreBreakdown
from lsm.query.session import Candidate


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


class FakeVectorDB:
    """Minimal fake vector DB provider."""

    def __init__(self, candidates=None, fts_candidates=None, fts_available=True):
        self._candidates = candidates or []
        self._fts_candidates = fts_candidates or []
        self._fts_available = fts_available
        self.query_calls = []
        self.fts_calls = []

    def query(self, vector, top_k, filters=None):
        self.query_calls.append({"vector": vector, "top_k": top_k, "filters": filters})
        ids = [c["id"] for c in self._candidates[:top_k]]
        docs = [c.get("text", "") for c in self._candidates[:top_k]]
        metas = [c.get("meta", {}) for c in self._candidates[:top_k]]
        dists = [c.get("distance", 0.2) for c in self._candidates[:top_k]]
        return SimpleNamespace(ids=ids, documents=docs, metadatas=metas, distances=dists)

    def fts_query(self, text, top_k):
        self.fts_calls.append({"text": text, "top_k": top_k})
        if not self._fts_available:
            return SimpleNamespace(ids=[], documents=[], metadatas=[], distances=[])
        ids = [c["id"] for c in self._fts_candidates[:top_k]]
        docs = [c.get("text", "") for c in self._fts_candidates[:top_k]]
        metas = [c.get("meta", {}) for c in self._fts_candidates[:top_k]]
        dists = [c.get("distance", -5.0) for c in self._fts_candidates[:top_k]]
        return SimpleNamespace(ids=ids, documents=docs, metadatas=metas, distances=dists)

    def get(self, ids=None, limit=None, include=None, filters=None):
        return SimpleNamespace(ids=[], documents=[], metadatas=[])


class FakeEmbedder:
    """Fake embedder returning fixed vector."""

    def encode(self, texts, batch_size=None, show_progress_bar=False, **kwargs):
        import numpy as np
        return np.array([[0.1] * 384 for _ in texts])


class FakeLLMProvider:
    """Fake LLM provider."""

    name = "fake"
    model = "fake-model"
    last_response_id = None

    def send_message(self, **kwargs):
        return "Answer [S1] text"

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


def _make_config(
    profile="dense_only",
    k=5,
    k_dense=100,
    k_sparse=100,
    rrf_dense_weight=0.7,
    rrf_sparse_weight=0.3,
    local_enabled=True,
    remote_enabled=False,
    mode="grounded",
):
    """Build a minimal fake LSMConfig for pipeline tests."""
    local_policy = SimpleNamespace(
        enabled=local_enabled,
        k=k,
        min_relevance=0.1,
    )
    remote_policy = SimpleNamespace(enabled=remote_enabled)
    model_knowledge_policy = SimpleNamespace(enabled=False)
    mode_config = SimpleNamespace(
        local_policy=local_policy,
        remote_policy=remote_policy,
        model_knowledge_policy=model_knowledge_policy,
        retrieval_profile=profile,
        synthesis_instructions="Answer the question.",
        source_policy=SimpleNamespace(local=local_policy, remote=remote_policy),
    )
    modes = {mode: mode_config}
    query = SimpleNamespace(
        mode=mode,
        k=k,
        retrieve_k=None,
        min_relevance=0.1,
        retrieval_profile=profile,
        k_dense=k_dense,
        k_sparse=k_sparse,
        rrf_dense_weight=rrf_dense_weight,
        rrf_sparse_weight=rrf_sparse_weight,
        enable_llm_server_cache=False,
        path_contains=None,
        ext_allow=None,
        ext_deny=None,
        chat_mode="single",
        enable_query_cache=False,
        query_cache_ttl=3600,
        query_cache_size=100,
        cluster_enabled=False,
        cluster_algorithm="kmeans",
        cluster_k=50,
        cluster_top_n=5,
        graph_expansion_enabled=False,
        graph_expansion_hops=2,
    )
    llm_service = SimpleNamespace(
        temperature=0.7,
        max_tokens=1000,
        model="fake-model",
    )
    llm = SimpleNamespace(
        resolve_service=lambda name: llm_service,
    )
    config = SimpleNamespace(
        query=query,
        modes=modes,
        llm=llm,
        batch_size=32,
    )
    config.get_mode_config = lambda name=None: modes.get(name or mode, mode_config)
    return config


def _sample_candidates(n=3, prefix="doc"):
    return [
        {
            "id": f"{prefix}_{i}",
            "text": f"Text of {prefix} {i}",
            "meta": {"source_path": f"/path/{prefix}_{i}.txt", "is_current": True},
            "distance": 0.1 + i * 0.05,
        }
        for i in range(n)
    ]


# ------------------------------------------------------------------
# Dense-only profile
# ------------------------------------------------------------------


class TestDenseOnlyProfile:
    def test_routes_to_dense_only(self):
        db = FakeVectorDB(candidates=_sample_candidates(3))
        config = _make_config(profile="dense_only")
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert "dense_recall" in package.retrieval_trace.stages_executed
        assert "sparse_recall" not in package.retrieval_trace.stages_executed
        assert "rrf_fusion" not in package.retrieval_trace.stages_executed

    def test_returns_candidates(self):
        db = FakeVectorDB(candidates=_sample_candidates(3))
        config = _make_config(profile="dense_only", k=2)
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert len(package.candidates) <= 2

    def test_unknown_profile_defaults_to_dense(self):
        db = FakeVectorDB(candidates=_sample_candidates(3))
        config = _make_config(profile="dense_only")
        # Simulate an unrecognized profile by patching after config creation
        config.get_mode_config().retrieval_profile = "some_future_profile"
        config.query.retrieval_profile = "some_future_profile"
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert "dense_recall" in package.retrieval_trace.stages_executed


# ------------------------------------------------------------------
# Hybrid RRF profile
# ------------------------------------------------------------------


class TestHybridRRFProfile:
    def test_routes_to_hybrid_rrf(self):
        db = FakeVectorDB(
            candidates=_sample_candidates(3, "dense"),
            fts_candidates=_sample_candidates(3, "sparse"),
        )
        config = _make_config(profile="hybrid_rrf")
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert "dense_recall" in package.retrieval_trace.stages_executed
        assert "sparse_recall" in package.retrieval_trace.stages_executed
        assert "rrf_fusion" in package.retrieval_trace.stages_executed

    def test_rrf_trace_counts(self):
        db = FakeVectorDB(
            candidates=_sample_candidates(5, "dense"),
            fts_candidates=_sample_candidates(4, "sparse"),
        )
        config = _make_config(profile="hybrid_rrf")
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert package.retrieval_trace.dense_candidates_count > 0
        assert package.retrieval_trace.sparse_candidates_count > 0

    def test_rrf_respects_k_limit(self):
        db = FakeVectorDB(
            candidates=_sample_candidates(10, "dense"),
            fts_candidates=_sample_candidates(10, "sparse"),
        )
        config = _make_config(profile="hybrid_rrf", k=3)
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert len(package.candidates) <= 3

    def test_rrf_falls_back_when_fts_unavailable(self):
        db = FakeVectorDB(
            candidates=_sample_candidates(3),
            fts_available=False,
        )
        db._extension_loaded = False
        config = _make_config(profile="hybrid_rrf")
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        # Falls back to dense_only
        assert "dense_recall" in package.retrieval_trace.stages_executed
        assert "rrf_fusion" not in package.retrieval_trace.stages_executed

    def test_rrf_overlapping_candidates_deduplicated(self):
        shared = _sample_candidates(2, "shared")
        db = FakeVectorDB(candidates=shared, fts_candidates=shared)
        config = _make_config(profile="hybrid_rrf", k=5)
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        cids = [c.cid for c in package.candidates]
        assert len(cids) == len(set(cids)), "Duplicate candidates in results"


# ------------------------------------------------------------------
# Profile in trace
# ------------------------------------------------------------------


class TestProfileTrace:
    def test_retrieval_profile_recorded_in_trace(self):
        db = FakeVectorDB(candidates=_sample_candidates(3))
        for profile in ["dense_only", "hybrid_rrf"]:
            fts_data = _sample_candidates(3, "fts") if profile == "hybrid_rrf" else []
            db = FakeVectorDB(candidates=_sample_candidates(3), fts_candidates=fts_data)
            config = _make_config(profile=profile)
            pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
            request = QueryRequest(question="test query")
            package = pipeline.build_sources(request)
            assert package.retrieval_trace.retrieval_profile == profile

    def test_timings_recorded(self):
        db = FakeVectorDB(candidates=_sample_candidates(3))
        config = _make_config(profile="dense_only")
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        timing_stages = [t.stage for t in package.retrieval_trace.timings]
        assert "local_retrieval" in timing_stages


# ------------------------------------------------------------------
# Local disabled
# ------------------------------------------------------------------


class TestLocalDisabled:
    def test_no_candidates_when_local_disabled(self):
        db = FakeVectorDB(candidates=_sample_candidates(3))
        config = _make_config(profile="dense_only", local_enabled=False)
        pipeline = RetrievalPipeline(db, FakeEmbedder(), config, FakeLLMProvider())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert len(package.candidates) == 0
        assert "dense_recall" not in package.retrieval_trace.stages_executed
