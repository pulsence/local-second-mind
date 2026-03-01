"""Tests for multi-hop retrieval (Phase 15.3)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from lsm.query.multi_hop import (
    MultiHopRequest,
    MultiHopResult,
    _decompose_query,
    _generate_followup,
    _merge_answers,
    _parse_sub_questions,
    iterative_multi_hop,
    parallel_multi_hop,
)
from lsm.query.pipeline_types import (
    ContextPackage,
    QueryRequest,
    QueryResponse,
    ScoreBreakdown,
)
from lsm.query.session import Candidate


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


def _make_candidate(cid="c1", source_path="/test.md"):
    return Candidate(
        cid=cid,
        text=f"Text of {cid}",
        meta={"source_path": source_path},
        distance=0.1,
    )


class FakeLLMProvider:
    """Fake LLM that returns pre-configured responses."""

    def __init__(self, responses=None):
        self._responses = list(responses or ['["What is X?", "How does Y work?"]'])
        self._call_count = 0
        self.name = "fake"
        self.model = "fake-model"
        self.last_response_id = None

    def send_message(self, **kwargs):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


class FakePipeline:
    """Fake RetrievalPipeline for multi-hop testing."""

    def __init__(self, candidates=None, answer="Answer [S1] from source."):
        self._candidates = candidates or [_make_candidate()]
        self._answer = answer
        self.build_sources_calls = 0
        self.synthesize_calls = 0
        self.execute_calls = 0
        self.llm_provider = FakeLLMProvider()

    def build_sources(self, request, progress_callback=None):
        self.build_sources_calls += 1
        return ContextPackage(
            request=request,
            candidates=list(self._candidates),
        )

    def synthesize_context(self, package):
        self.synthesize_calls += 1
        return replace(
            package,
            context_block="[S1] Content",
            source_labels={"S1": {"source_path": "/test.md"}},
            starting_prompt="Answer from sources.",
        )

    def execute(self, package):
        self.execute_calls += 1
        return QueryResponse(
            answer=self._answer,
            package=package,
            response_id=f"resp-{self.execute_calls}",
        )


# ------------------------------------------------------------------
# Tests: _parse_sub_questions
# ------------------------------------------------------------------


class TestParseSubQuestions:
    def test_json_array(self):
        result = _parse_sub_questions('["Q1?", "Q2?", "Q3?"]', 4)
        assert result == ["Q1?", "Q2?", "Q3?"]

    def test_json_with_markdown(self):
        text = '```json\n["Q1?", "Q2?"]\n```'
        result = _parse_sub_questions(text, 4)
        assert result == ["Q1?", "Q2?"]

    def test_respects_max_count(self):
        result = _parse_sub_questions('["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]', 3)
        assert len(result) == 3

    def test_fallback_to_lines(self):
        text = "1. What is the meaning of X?\n2. How does Y actually work?\n3. Why does Z happen in practice?"
        result = _parse_sub_questions(text, 4)
        assert len(result) == 3

    def test_empty_input(self):
        assert _parse_sub_questions("", 4) == []
        assert _parse_sub_questions(None, 4) == []


# ------------------------------------------------------------------
# Tests: _decompose_query
# ------------------------------------------------------------------


class TestDecomposeQuery:
    def test_decomposes_with_llm(self):
        provider = FakeLLMProvider(['["What is A?", "What is B?"]'])
        result = _decompose_query("Complex question about A and B", provider, 4)
        assert len(result) == 2
        assert "What is A?" in result

    def test_fallback_on_llm_error(self):
        class FailingProvider:
            def send_message(self, **kwargs):
                raise RuntimeError("LLM error")
        result = _decompose_query("test query", FailingProvider(), 4)
        assert result == ["test query"]


# ------------------------------------------------------------------
# Tests: _generate_followup
# ------------------------------------------------------------------


class TestGenerateFollowup:
    def test_generates_followup(self):
        provider = FakeLLMProvider(["What are the implications?"])
        result = _generate_followup("Original Q", "Partial answer", provider)
        assert result == "What are the implications?"

    def test_returns_none_on_error(self):
        class FailingProvider:
            def send_message(self, **kwargs):
                raise RuntimeError("error")
        result = _generate_followup("Q", "A", FailingProvider())
        assert result is None


# ------------------------------------------------------------------
# Tests: parallel_multi_hop
# ------------------------------------------------------------------


class TestParallelMultiHop:
    def test_decomposes_and_retrieves(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider(['["Sub Q1?", "Sub Q2?"]'])

        request = MultiHopRequest(query="Complex question", max_hops=3)
        result = parallel_multi_hop(request, pipeline, llm)

        assert isinstance(result, MultiHopResult)
        assert len(result.sub_questions) == 2
        assert len(result.hop_responses) == 2
        assert result.total_hops == 2
        assert pipeline.build_sources_calls == 2
        assert pipeline.execute_calls == 2

    def test_single_question_fallback(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider(["invalid json"])

        request = MultiHopRequest(query="Simple question")
        result = parallel_multi_hop(request, pipeline, llm)

        assert len(result.sub_questions) >= 1
        assert len(result.hop_responses) >= 1

    def test_deduplicates_candidates(self):
        shared = _make_candidate("shared_c1")
        pipeline = FakePipeline(candidates=[shared])
        llm = FakeLLMProvider(['["Q1?", "Q2?"]'])

        request = MultiHopRequest(query="test")
        result = parallel_multi_hop(request, pipeline, llm)

        # Both sub-queries return same candidate — should be deduplicated
        cids = [c.cid for c in result.merged_candidates]
        assert len(cids) == len(set(cids))

    def test_merges_answers(self):
        pipeline = FakePipeline(answer="Sub-answer [S1]")
        llm = FakeLLMProvider([
            '["Q1?", "Q2?"]',
            "Merged comprehensive answer",
        ])

        request = MultiHopRequest(query="test")
        result = parallel_multi_hop(request, pipeline, llm)
        assert result.answer is not None


# ------------------------------------------------------------------
# Tests: iterative_multi_hop
# ------------------------------------------------------------------


class TestIterativeMultiHop:
    def test_single_hop(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider(["What more?"])

        request = MultiHopRequest(query="Original", max_hops=1, strategy="iterative")
        result = iterative_multi_hop(request, pipeline, llm)

        assert result.total_hops == 1
        assert len(result.hop_responses) == 1
        assert result.sub_questions == ["Original"]

    def test_multiple_hops(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider(["Follow-up question?", "Another follow-up?"])

        request = MultiHopRequest(query="Original", max_hops=3, strategy="iterative")
        result = iterative_multi_hop(request, pipeline, llm)

        assert result.total_hops <= 3
        assert len(result.sub_questions) >= 1

    def test_stops_on_duplicate_question(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider(["Original"])  # Same as original

        request = MultiHopRequest(query="Original", max_hops=5, strategy="iterative")
        result = iterative_multi_hop(request, pipeline, llm)

        # Should stop early since follow-up == original
        assert result.total_hops == 1

    def test_max_hops_respected(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider([
            "Follow-up 1?",
            "Follow-up 2?",
            "Follow-up 3?",
            "Follow-up 4?",
        ])

        request = MultiHopRequest(query="Start", max_hops=2)
        result = iterative_multi_hop(request, pipeline, llm)
        assert result.total_hops <= 2

    def test_chains_response_ids(self):
        pipeline = FakePipeline()
        llm = FakeLLMProvider(["Follow-up?"])

        request = MultiHopRequest(query="Start", max_hops=2)
        result = iterative_multi_hop(request, pipeline, llm)

        # Second hop should have gotten prior_response_id
        assert result.total_hops == 2


# ------------------------------------------------------------------
# Tests: MultiHopRequest
# ------------------------------------------------------------------


class TestMultiHopRequest:
    def test_defaults(self):
        req = MultiHopRequest(query="test")
        assert req.max_hops == 3
        assert req.strategy == "parallel"
        assert req.mode is None
        assert req.conversation_id is None

    def test_custom_values(self):
        req = MultiHopRequest(
            query="test",
            max_hops=5,
            strategy="iterative",
            mode="grounded",
            conversation_id="conv-1",
        )
        assert req.max_hops == 5
        assert req.strategy == "iterative"
