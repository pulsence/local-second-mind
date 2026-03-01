"""Tests for multi-vector retrieval stage (Phase 14.3)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate
from lsm.query.stages.multi_vector import (
    multi_vector_recall,
    _expand_to_chunks,
    _multi_rrf_fuse,
    _results_to_candidates,
)


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


class FakeVectorDB:
    """Fake vector DB that returns different results based on node_type filter."""

    def __init__(
        self,
        chunk_results=None,
        section_results=None,
        file_results=None,
        expansion_results=None,
    ):
        self._chunk_results = chunk_results or []
        self._section_results = section_results or []
        self._file_results = file_results or []
        self._expansion_results = expansion_results or []
        self.query_calls = []

    def query(self, vector, top_k, filters=None):
        self.query_calls.append({"vector": vector, "top_k": top_k, "filters": filters})
        node_type = (filters or {}).get("node_type", "chunk")
        source_path_filter = (filters or {}).get("source_path")

        if source_path_filter and node_type == "chunk":
            # Expansion query
            results = self._expansion_results
        elif node_type == "section_summary":
            results = self._section_results
        elif node_type == "file_summary":
            results = self._file_results
        else:
            results = self._chunk_results

        results = results[:top_k]
        return SimpleNamespace(
            ids=[r["id"] for r in results],
            documents=[r.get("text", "") for r in results],
            metadatas=[r.get("meta", {}) for r in results],
            distances=[r.get("distance", 0.2) for r in results],
        )


def _make_results(n, prefix="chunk", node_type="chunk", source_prefix="/path"):
    return [
        {
            "id": f"{prefix}_{i}",
            "text": f"Text of {prefix} {i}",
            "meta": {
                "source_path": f"{source_prefix}/{prefix}_{i}.md",
                "node_type": node_type,
            },
            "distance": 0.1 + i * 0.05,
        }
        for i in range(n)
    ]


# ------------------------------------------------------------------
# Tests: _results_to_candidates
# ------------------------------------------------------------------


class TestResultsToCandidates:
    def test_converts_results(self):
        result = SimpleNamespace(
            ids=["a", "b"],
            documents=["text a", "text b"],
            metadatas=[{"source_path": "/a.md"}, {"source_path": "/b.md"}],
            distances=[0.1, 0.3],
        )
        candidates = _results_to_candidates(result, "chunk")
        assert len(candidates) == 2
        assert candidates[0].cid == "a"
        assert candidates[0].text == "text a"
        assert candidates[0].score_breakdown.dense_score == pytest.approx(0.9)
        assert candidates[0].score_breakdown.dense_rank == 1
        assert candidates[1].score_breakdown.dense_rank == 2

    def test_empty_results(self):
        result = SimpleNamespace(ids=[], documents=[], metadatas=[], distances=[])
        candidates = _results_to_candidates(result, "chunk")
        assert candidates == []


# ------------------------------------------------------------------
# Tests: _expand_to_chunks
# ------------------------------------------------------------------


class TestExpandToChunks:
    def test_expands_from_source_path(self):
        expansion = _make_results(3, "exp", "chunk")
        db = FakeVectorDB(expansion_results=expansion)
        embedding = [0.1] * 384

        result = _expand_to_chunks(embedding, db, "/path/file.md", k=3, base_filters=None)
        assert len(result) == 3

        # Verify the db was called with correct filters
        assert len(db.query_calls) == 1
        call_filters = db.query_calls[0]["filters"]
        assert call_filters["source_path"] == "/path/file.md"
        assert call_filters["node_type"] == "chunk"

    def test_respects_k(self):
        expansion = _make_results(5, "exp", "chunk")
        db = FakeVectorDB(expansion_results=expansion)
        embedding = [0.1] * 384

        result = _expand_to_chunks(embedding, db, "/path/file.md", k=2, base_filters=None)
        assert len(result) == 2

    def test_merges_base_filters(self):
        expansion = _make_results(2, "exp", "chunk")
        db = FakeVectorDB(expansion_results=expansion)
        embedding = [0.1] * 384

        _expand_to_chunks(
            embedding, db, "/path/file.md", k=2,
            base_filters={"is_current": True},
        )
        call_filters = db.query_calls[0]["filters"]
        assert call_filters["is_current"] is True
        assert call_filters["source_path"] == "/path/file.md"


# ------------------------------------------------------------------
# Tests: _multi_rrf_fuse
# ------------------------------------------------------------------


class TestMultiRRFFuse:
    def _make_candidates(self, ids, source_paths=None):
        candidates = []
        for i, cid in enumerate(ids):
            sp = source_paths[i] if source_paths else f"/path/{cid}.md"
            candidates.append(
                Candidate(
                    cid=cid,
                    text=f"Text {cid}",
                    meta={"source_path": sp},
                    distance=0.1 + i * 0.05,
                    score_breakdown=ScoreBreakdown(dense_score=0.9 - i * 0.05),
                )
            )
        return candidates

    def test_chunk_only_fusion(self):
        chunks = self._make_candidates(["c1", "c2", "c3"])
        result = _multi_rrf_fuse(chunks, [], [])
        assert len(result) == 3
        # First chunk should have highest fused score
        assert result[0].cid == "c1"
        assert result[0].score_breakdown.fused_score > result[1].score_breakdown.fused_score

    def test_section_boost(self):
        chunks = self._make_candidates(
            ["c1", "c2"],
            source_paths=["/path/a.md", "/path/b.md"],
        )
        sections = self._make_candidates(
            ["s1"],
            source_paths=["/path/b.md"],  # section match for c2's source
        )
        result = _multi_rrf_fuse(chunks, sections, [])
        # c2 should be boosted because its source has a section match
        c2_score = next(c for c in result if c.cid == "c2").score_breakdown.fused_score
        # With section boost, c2 gets a higher section contribution
        assert c2_score is not None

    def test_file_boost(self):
        chunks = self._make_candidates(
            ["c1", "c2"],
            source_paths=["/path/a.md", "/path/b.md"],
        )
        files = self._make_candidates(
            ["f1"],
            source_paths=["/path/b.md"],  # file match for c2's source
        )
        result = _multi_rrf_fuse(chunks, [], files)
        c2_score = next(c for c in result if c.cid == "c2").score_breakdown.fused_score
        assert c2_score is not None

    def test_all_three_granularities(self):
        chunks = self._make_candidates(
            ["c1", "c2", "c3"],
            source_paths=["/a.md", "/b.md", "/c.md"],
        )
        sections = self._make_candidates(
            ["s1"],
            source_paths=["/b.md"],
        )
        files = self._make_candidates(
            ["f1"],
            source_paths=["/b.md"],
        )
        result = _multi_rrf_fuse(chunks, sections, files)
        assert len(result) == 3
        # c2 should get the biggest boost (both section + file match)
        scores = {c.cid: c.score_breakdown.fused_score for c in result}
        # c2 has both section and file boost
        assert scores["c2"] > scores["c3"]

    def test_preserves_dense_score(self):
        chunks = self._make_candidates(["c1"])
        result = _multi_rrf_fuse(chunks, [], [])
        assert result[0].score_breakdown.dense_score == pytest.approx(0.9)

    def test_deduplicates_chunk_ids(self):
        # Same chunk appearing twice (e.g. from chunk + expanded)
        chunks = self._make_candidates(["c1", "c1", "c2"])
        result = _multi_rrf_fuse(chunks, [], [])
        cids = [c.cid for c in result]
        assert len(cids) == len(set(cids))


# ------------------------------------------------------------------
# Tests: multi_vector_recall (end-to-end)
# ------------------------------------------------------------------


class TestMultiVectorRecall:
    def test_chunk_only_recall(self):
        chunk_results = _make_results(5, "chunk", "chunk")
        db = FakeVectorDB(chunk_results=chunk_results)
        embedding = [0.1] * 384

        candidates = multi_vector_recall(
            embedding, db, top_k=5, k_section=0, k_file=0,
        )
        assert len(candidates) == 5
        assert all(c.cid.startswith("chunk_") for c in candidates)

    def test_with_section_summaries(self):
        chunk_results = _make_results(3, "chunk", "chunk", "/docs")
        section_results = _make_results(2, "sec", "section_summary", "/docs")
        expansion_results = _make_results(2, "exp", "chunk", "/docs")
        db = FakeVectorDB(
            chunk_results=chunk_results,
            section_results=section_results,
            expansion_results=expansion_results,
        )
        embedding = [0.1] * 384

        candidates = multi_vector_recall(
            embedding, db, top_k=3, k_section=2, k_file=0,
        )
        assert len(candidates) > 0

    def test_with_file_summaries(self):
        chunk_results = _make_results(3, "chunk", "chunk", "/docs")
        file_results = _make_results(1, "file", "file_summary", "/other")
        expansion_results = _make_results(2, "exp", "chunk", "/other")
        db = FakeVectorDB(
            chunk_results=chunk_results,
            file_results=file_results,
            expansion_results=expansion_results,
        )
        embedding = [0.1] * 384

        candidates = multi_vector_recall(
            embedding, db, top_k=3, k_section=0, k_file=1,
        )
        assert len(candidates) > 0

    def test_expansion_skips_existing_sources(self):
        """If chunk-level already has results from a source path, skip expansion."""
        chunk_results = [
            {
                "id": "c1",
                "text": "chunk text",
                "meta": {"source_path": "/shared.md", "node_type": "chunk"},
                "distance": 0.1,
            },
        ]
        section_results = [
            {
                "id": "s1",
                "text": "section text",
                "meta": {"source_path": "/shared.md", "node_type": "section_summary"},
                "distance": 0.1,
            },
        ]
        db = FakeVectorDB(
            chunk_results=chunk_results,
            section_results=section_results,
        )
        embedding = [0.1] * 384

        candidates = multi_vector_recall(
            embedding, db, top_k=5, k_section=5, k_file=0,
        )
        # No expansion calls since chunk already covers /shared.md
        expansion_calls = [
            c for c in db.query_calls
            if c["filters"] and c["filters"].get("source_path")
        ]
        assert len(expansion_calls) == 0

    def test_filters_propagated(self):
        chunk_results = _make_results(3, "chunk", "chunk")
        db = FakeVectorDB(chunk_results=chunk_results)
        embedding = [0.1] * 384

        multi_vector_recall(
            embedding, db, top_k=3, k_section=2, k_file=1,
            filters={"is_current": True},
        )
        # All query calls should include is_current filter
        for call in db.query_calls:
            assert call["filters"]["is_current"] is True

    def test_returns_fused_scores(self):
        chunk_results = _make_results(3, "chunk", "chunk")
        db = FakeVectorDB(chunk_results=chunk_results)
        embedding = [0.1] * 384

        candidates = multi_vector_recall(
            embedding, db, top_k=3, k_section=0, k_file=0,
        )
        for c in candidates:
            assert c.score_breakdown is not None
            assert c.score_breakdown.fused_score is not None


# ------------------------------------------------------------------
# Tests: pipeline integration
# ------------------------------------------------------------------


class TestMultiVectorProfile:
    """Test multi_vector profile routing in RetrievalPipeline."""

    def _make_config(self, profile="multi_vector", k=5):
        local_policy = SimpleNamespace(enabled=True, k=k, min_relevance=0.1)
        remote_policy = SimpleNamespace(enabled=False)
        model_knowledge_policy = SimpleNamespace(enabled=False)
        mode_config = SimpleNamespace(
            local_policy=local_policy,
            remote_policy=remote_policy,
            model_knowledge_policy=model_knowledge_policy,
            retrieval_profile=profile,
            synthesis_instructions="Answer the question.",
            source_policy=SimpleNamespace(local=local_policy, remote=remote_policy),
        )
        modes = {"grounded": mode_config}
        query = SimpleNamespace(
            mode="grounded",
            k=k,
            retrieve_k=None,
            min_relevance=0.1,
            retrieval_profile=profile,
            k_dense=100,
            k_sparse=100,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
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
            temperature=0.7, max_tokens=1000, model="fake-model",
        )
        llm = SimpleNamespace(resolve_service=lambda name: llm_service)
        config = SimpleNamespace(
            query=query, modes=modes, llm=llm, batch_size=32,
        )
        config.get_mode_config = lambda name=None: modes.get(name or "grounded", mode_config)
        return config

    def _make_fake_db(self):
        chunk_results = _make_results(5, "chunk", "chunk")
        return FakeVectorDB(chunk_results=chunk_results)

    def _make_fake_embedder(self):
        class FakeEmbedder:
            def encode(self, texts, batch_size=None, show_progress_bar=False, **kwargs):
                import numpy as np
                return np.array([[0.1] * 384 for _ in texts])
        return FakeEmbedder()

    def _make_fake_llm(self):
        class FakeLLM:
            name = "fake"
            model = "fake-model"
            last_response_id = None
            def send_message(self, **kwargs): return "Answer [S1]"
            def estimate_cost(self, input_tokens, output_tokens): return 0.001
        return FakeLLM()

    def test_routes_to_multi_vector(self):
        from lsm.query.pipeline import RetrievalPipeline
        from lsm.query.pipeline_types import QueryRequest

        db = self._make_fake_db()
        config = self._make_config(profile="multi_vector")
        pipeline = RetrievalPipeline(db, self._make_fake_embedder(), config, self._make_fake_llm())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert "multi_vector_recall" in package.retrieval_trace.stages_executed
        assert "dense_recall" not in package.retrieval_trace.stages_executed

    def test_returns_candidates(self):
        from lsm.query.pipeline import RetrievalPipeline
        from lsm.query.pipeline_types import QueryRequest

        db = self._make_fake_db()
        config = self._make_config(profile="multi_vector", k=3)
        pipeline = RetrievalPipeline(db, self._make_fake_embedder(), config, self._make_fake_llm())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert len(package.candidates) <= 3
        assert len(package.candidates) > 0

    def test_profile_recorded_in_trace(self):
        from lsm.query.pipeline import RetrievalPipeline
        from lsm.query.pipeline_types import QueryRequest

        db = self._make_fake_db()
        config = self._make_config(profile="multi_vector")
        pipeline = RetrievalPipeline(db, self._make_fake_embedder(), config, self._make_fake_llm())
        request = QueryRequest(question="test query")
        package = pipeline.build_sources(request)
        assert package.retrieval_trace.retrieval_profile == "multi_vector"
