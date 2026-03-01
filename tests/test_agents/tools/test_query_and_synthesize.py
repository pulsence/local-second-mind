"""
Tests for QueryAndSynthesizeTool.
"""
from __future__ import annotations

import json

import pytest

from lsm.agents.tools.query_and_synthesize import QueryAndSynthesizeTool
from lsm.query.pipeline_types import (
    ContextPackage,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
)
from lsm.query.session import Candidate


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------


class _FakeConfig:
    def get_mode_config(self, name):
        from lsm.config.models.modes import GROUNDED_MODE
        return GROUNDED_MODE


class FakePipeline:
    """Minimal stand-in for RetrievalPipeline.run()."""

    def __init__(self, answer="The answer.", candidates=None, response_id=None, conversation_id=None):
        self._answer = answer
        self._candidates = candidates or []
        self._response_id = response_id
        self._conversation_id = conversation_id
        self.last_request = None
        self.config = _FakeConfig()

    def run(self, request, progress_callback=None):
        self.last_request = request
        package = ContextPackage(
            request=request,
            candidates=self._candidates,
            retrieval_trace=RetrievalTrace(stages_executed=["dense_recall"]),
        )
        return QueryResponse(
            answer=self._answer,
            package=package,
            conversation_id=self._conversation_id or request.conversation_id,
            response_id=self._response_id,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_complete_query_response():
    candidates = [
        Candidate(cid="c1", text="chunk text", meta={"source_path": "/doc.md"}, distance=0.1),
    ]
    pipeline = FakePipeline(
        answer="Python is great.",
        candidates=candidates,
        response_id="resp-001",
    )
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    output = json.loads(tool.execute({"query": "what is Python?"}))

    assert output["answer"] == "Python is great."
    assert output["response_id"] == "resp-001"
    assert len(output["candidates"]) == 1
    assert output["candidates"][0]["id"] == "c1"


def test_response_id_and_conversation_id_returned():
    pipeline = FakePipeline(response_id="resp-abc", conversation_id="conv-xyz")
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    output = json.loads(tool.execute({
        "query": "test",
        "conversation_id": "conv-xyz",
    }))

    assert output["response_id"] == "resp-abc"
    assert output["conversation_id"] == "conv-xyz"


def test_query_required():
    pipeline = FakePipeline()
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    with pytest.raises(ValueError, match="query is required"):
        tool.execute({"query": ""})


def test_mode_forwarded():
    pipeline = FakePipeline()
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    tool.execute({"query": "test", "mode": "hybrid"})
    assert pipeline.last_request.mode == "hybrid"


def test_filters_forwarded():
    pipeline = FakePipeline()
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    tool.execute({
        "query": "test",
        "filters": {"path_contains": ["/docs"], "ext_deny": [".pdf"]},
    })

    assert pipeline.last_request.filters is not None
    assert pipeline.last_request.filters.path_contains == ["/docs"]
    assert pipeline.last_request.filters.ext_deny == [".pdf"]


def test_k_forwarded():
    pipeline = FakePipeline()
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    tool.execute({"query": "test", "k": 3})
    assert pipeline.last_request.k == 3


def test_starting_prompt_forwarded():
    pipeline = FakePipeline()
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    tool.execute({"query": "test", "starting_prompt": "Custom prompt"})
    assert pipeline.last_request.starting_prompt == "Custom prompt"


def test_prior_response_id_forwarded():
    pipeline = FakePipeline()
    tool = QueryAndSynthesizeTool(pipeline=pipeline)

    tool.execute({"query": "test", "prior_response_id": "prev-resp-001"})
    assert pipeline.last_request.prior_response_id == "prev-resp-001"
