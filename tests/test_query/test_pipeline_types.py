"""Tests for pipeline data types."""

from lsm.query.pipeline_types import (
    CostEntry,
    Citation,
    ContextPackage,
    FilterSet,
    QueryRequest,
    QueryResponse,
    RemoteSource,
    RetrievalTrace,
    ScoreBreakdown,
    StageTimings,
)
from lsm.query.session import Candidate


class TestFilterSet:
    def test_defaults(self):
        f = FilterSet()
        assert f.path_contains is None
        assert f.ext_allow is None
        assert f.ext_deny is None
        assert not f.is_active()

    def test_active(self):
        f = FilterSet(path_contains=["notes"])
        assert f.is_active()

    def test_to_dict(self):
        f = FilterSet(ext_allow=[".md"])
        d = f.to_dict()
        assert d["ext_allow"] == [".md"]
        assert d["path_contains"] is None


class TestScoreBreakdown:
    def test_defaults_all_none(self):
        s = ScoreBreakdown()
        assert s.dense_score is None
        assert s.sparse_rank is None
        assert s.to_dict() == {}

    def test_partial_fields(self):
        s = ScoreBreakdown(dense_score=0.85, dense_rank=1)
        d = s.to_dict()
        assert d == {"dense_score": 0.85, "dense_rank": 1}

    def test_all_fields(self):
        s = ScoreBreakdown(
            dense_score=0.9,
            dense_rank=1,
            sparse_score=12.5,
            sparse_rank=3,
            fused_score=0.75,
            rerank_score=0.95,
            temporal_boost=1.1,
        )
        d = s.to_dict()
        assert len(d) == 7


class TestCitation:
    def test_minimal(self):
        c = Citation(chunk_id="c1", source_path="/a.md")
        assert c.to_dict() == {"chunk_id": "c1", "source_path": "/a.md"}

    def test_full(self):
        c = Citation(
            chunk_id="c1",
            source_path="/a.md",
            heading="Intro",
            page_number=5,
            url_or_doi="https://example.com",
            snippet="some text",
        )
        d = c.to_dict()
        assert d["heading"] == "Intro"
        assert d["page_number"] == 5


class TestRetrievalTrace:
    def test_defaults(self):
        t = RetrievalTrace()
        assert t.stages_executed == []
        assert t.total_duration_ms() == 0.0

    def test_with_stages(self):
        t = RetrievalTrace(
            stages_executed=["dense_recall", "rrf_fusion"],
            timings=[
                StageTimings(stage="dense_recall", duration_ms=100.5),
                StageTimings(stage="rrf_fusion", duration_ms=25.3),
            ],
            dense_candidates_count=50,
            fused_candidates_count=20,
            retrieval_profile="hybrid_rrf",
        )
        assert t.total_duration_ms() == 125.8
        d = t.to_dict()
        assert d["retrieval_profile"] == "hybrid_rrf"
        assert d["dense_candidates_count"] == 50
        assert len(d["timings"]) == 2


class TestCostEntry:
    def test_defaults(self):
        c = CostEntry(provider="openai", model="gpt-4")
        assert c.cost == 0.0
        assert c.kind == "synthesis"

    def test_to_dict(self):
        c = CostEntry(
            provider="openai",
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
            kind="rerank",
        )
        d = c.to_dict()
        assert d["kind"] == "rerank"
        assert d["cost"] == 0.05


class TestRemoteSource:
    def test_weighted_score(self):
        r = RemoteSource(provider="brave", score=0.8, weight=0.5)
        assert r.weighted_score() == 0.4

    def test_to_dict(self):
        r = RemoteSource(provider="arxiv", title="Paper", url="http://x", score=0.9)
        d = r.to_dict()
        assert d["provider"] == "arxiv"
        assert d["weighted_score"] == 0.9


class TestQueryRequest:
    def test_mode_resolution_none_defaults(self):
        r = QueryRequest(question="test")
        assert r.resolved_mode == "grounded"

    def test_mode_resolution_explicit(self):
        r = QueryRequest(question="test", mode="insight")
        assert r.resolved_mode == "insight"

    def test_to_dict_minimal(self):
        r = QueryRequest(question="What is X?")
        d = r.to_dict()
        assert d["question"] == "What is X?"
        assert d["mode"] == "grounded"
        assert "filters" not in d

    def test_to_dict_with_conversation(self):
        r = QueryRequest(
            question="Follow up",
            conversation_id="conv-1",
            prior_response_id="resp-1",
        )
        d = r.to_dict()
        assert d["conversation_id"] == "conv-1"
        assert d["prior_response_id"] == "resp-1"

    def test_to_dict_with_filters(self):
        r = QueryRequest(
            question="test",
            filters=FilterSet(ext_allow=[".md"]),
        )
        d = r.to_dict()
        assert "filters" in d
        assert d["filters"]["ext_allow"] == [".md"]


class TestContextPackage:
    def test_total_cost(self):
        pkg = ContextPackage(
            request=QueryRequest(question="test"),
            costs=[
                CostEntry(provider="a", model="m1", cost=0.01),
                CostEntry(provider="b", model="m2", cost=0.02),
            ],
        )
        assert abs(pkg.total_cost() - 0.03) < 1e-9


class TestQueryResponse:
    def test_total_cost(self):
        pkg = ContextPackage(
            request=QueryRequest(question="test"),
            costs=[CostEntry(provider="a", model="m", cost=0.01)],
        )
        resp = QueryResponse(
            answer="The answer",
            package=pkg,
            costs=[CostEntry(provider="b", model="m", cost=0.05)],
        )
        assert abs(resp.total_cost() - 0.06) < 1e-9

    def test_conversation_fields_serialize(self):
        pkg = ContextPackage(request=QueryRequest(question="test"))
        resp = QueryResponse(
            answer="The answer",
            package=pkg,
            conversation_id="conv-123",
            response_id="resp-456",
        )
        d = resp.to_dict()
        assert d["conversation_id"] == "conv-123"
        assert d["response_id"] == "resp-456"

    def test_shortcuts(self):
        cand = Candidate(cid="c1", text="hello", meta={})
        remote = RemoteSource(provider="brave")
        pkg = ContextPackage(
            request=QueryRequest(question="test"),
            candidates=[cand],
            remote_sources=[remote],
            retrieval_trace=RetrievalTrace(stages_executed=["dense"]),
        )
        resp = QueryResponse(answer="ans", package=pkg)
        assert resp.candidates == [cand]
        assert resp.remote_sources == [remote]
        assert resp.retrieval_trace.stages_executed == ["dense"]

    def test_serialization_roundtrip(self):
        """Verify to_dict produces JSON-serializable output."""
        import json

        pkg = ContextPackage(
            request=QueryRequest(question="test"),
            retrieval_trace=RetrievalTrace(
                stages_executed=["dense_recall"],
                timings=[StageTimings(stage="dense_recall", duration_ms=42.5)],
            ),
        )
        resp = QueryResponse(
            answer="answer",
            package=pkg,
            citations=[Citation(chunk_id="c1", source_path="/a.md")],
            costs=[CostEntry(provider="p", model="m", cost=0.01)],
        )
        serialized = json.dumps(resp.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["answer"] == "answer"
        assert deserialized["costs"][0]["provider"] == "p"
        assert deserialized["retrieval_trace"]["stages_executed"] == ["dense_recall"]


class TestCandidateExtensions:
    """Test the new fields on Candidate."""

    def test_score_breakdown_default_none(self):
        c = Candidate(cid="x", text="text", meta={})
        assert c.score_breakdown is None

    def test_embedding_default_none(self):
        c = Candidate(cid="x", text="text", meta={})
        assert c.embedding is None

    def test_score_breakdown_set(self):
        from lsm.query.pipeline_types import ScoreBreakdown

        sb = ScoreBreakdown(dense_score=0.9, dense_rank=1)
        c = Candidate(cid="x", text="text", meta={}, score_breakdown=sb)
        assert c.score_breakdown.dense_score == 0.9

    def test_embedding_set(self):
        c = Candidate(cid="x", text="text", meta={}, embedding=[0.1, 0.2, 0.3])
        assert c.embedding == [0.1, 0.2, 0.3]
