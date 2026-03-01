"""
Tests for QueryContextTool.
"""
from __future__ import annotations

import json

import pytest

from lsm.agents.tools.query_context import QueryContextTool
from lsm.query.pipeline_types import (
    ContextPackage,
    QueryRequest,
    RemoteSource,
    RetrievalTrace,
)
from lsm.query.session import Candidate


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------

class FakePipeline:
    """Minimal stand-in for RetrievalPipeline.build_sources()."""

    def __init__(self, candidates=None, remote_sources=None, relevance=0.85):
        self._candidates = candidates or []
        self._remote_sources = remote_sources or []
        self._relevance = relevance
        self.last_request = None

    def build_sources(self, request):
        self.last_request = request
        return ContextPackage(
            request=request,
            candidates=self._candidates,
            remote_sources=self._remote_sources,
            retrieval_trace=RetrievalTrace(stages_executed=["dense_recall"]),
            relevance=self._relevance,
            local_enabled=True,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_serialized_context_package():
    candidates = [
        Candidate(cid="c1", text="chunk text one", meta={"source_path": "/docs/a.md"}, distance=0.1),
        Candidate(cid="c2", text="chunk text two", meta={"source_path": "/docs/b.md"}, distance=0.2),
    ]
    pipeline = FakePipeline(candidates=candidates)
    tool = QueryContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({"query": "what is Python?"}))

    assert output["candidate_count"] == 2
    assert len(output["candidates"]) == 2
    assert output["candidates"][0]["id"] == "c1"
    assert output["relevance"] == 0.85
    assert output["local_enabled"] is True
    assert "dense_recall" in output["retrieval_trace"]["stages_executed"]


def test_mode_resolution():
    pipeline = FakePipeline()
    tool = QueryContextTool(pipeline=pipeline)

    tool.execute({"query": "test", "mode": "insight"})
    assert pipeline.last_request.mode == "insight"


def test_filters_round_trip():
    pipeline = FakePipeline()
    tool = QueryContextTool(pipeline=pipeline)

    tool.execute({
        "query": "test",
        "filters": {
            "path_contains": ["/docs"],
            "ext_allow": [".md"],
        },
    })
    assert pipeline.last_request.filters is not None
    assert pipeline.last_request.filters.path_contains == ["/docs"]
    assert pipeline.last_request.filters.ext_allow == [".md"]


def test_conversation_fields_round_trip():
    pipeline = FakePipeline()
    tool = QueryContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({
        "query": "test",
        "conversation_id": "conv-123",
        "prior_response_id": "resp-abc",
        "starting_prompt": "Custom prompt",
    }))

    assert pipeline.last_request.conversation_id == "conv-123"
    assert pipeline.last_request.prior_response_id == "resp-abc"
    assert pipeline.last_request.starting_prompt == "Custom prompt"
    assert output["conversation_id"] == "conv-123"


def test_k_parameter():
    pipeline = FakePipeline()
    tool = QueryContextTool(pipeline=pipeline)

    tool.execute({"query": "test", "k": 5})
    assert pipeline.last_request.k == 5


def test_empty_query_raises():
    pipeline = FakePipeline()
    tool = QueryContextTool(pipeline=pipeline)

    with pytest.raises(ValueError, match="query is required"):
        tool.execute({"query": ""})

    with pytest.raises(ValueError, match="query is required"):
        tool.execute({})


def test_remote_sources_included():
    remote = [RemoteSource(provider="wikipedia", title="Python", snippet="A language")]
    pipeline = FakePipeline(remote_sources=remote)
    tool = QueryContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({"query": "test"}))
    assert "remote_sources" in output
    assert output["remote_sources"][0]["provider"] == "wikipedia"


def test_candidate_text_truncated():
    long_text = "x" * 1000
    candidates = [Candidate(cid="c1", text=long_text, meta={}, distance=0.1)]
    pipeline = FakePipeline(candidates=candidates)
    tool = QueryContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({"query": "test"}))
    assert len(output["candidates"][0]["text"]) == 500
