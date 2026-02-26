"""
Tests for QueryKnowledgeBaseTool.

Covers:
- execute() calls query_sync() with injected config, embedder, collection
- Fresh SessionState is constructed on each call
- Result JSON contains 'answer' and 'sources_display' keys
- Missing query raises ValueError
- Blank query (after strip) raises ValueError
- Optional top_k is respected in candidate slicing
- Optional filters parameter is accepted
"""
from __future__ import annotations

import json

import pytest

from lsm.agents.tools.query_knowledge_base import QueryKnowledgeBaseTool
from lsm.query.api import QueryResult
from lsm.query.session import Candidate


def _make_candidate(cid: str, text: str, distance: float = 0.2) -> Candidate:
    return Candidate(cid=cid, text=text, meta={"source_path": f"{cid}.md"}, distance=distance)


def _make_result(candidates: list[Candidate], answer: str = "Test answer") -> QueryResult:
    return QueryResult(
        answer=answer,
        sources_display="source.md",
        candidates=candidates,
        cost=0.0,
        remote_sources=[],
        debug_info={},
    )


class FakeConfig:
    pass


def test_execute_calls_query_sync_with_injected_deps(monkeypatch) -> None:
    """execute() forwards config, embedder, and collection to query_sync()."""
    calls: list[dict] = []

    def fake_query_sync(**kwargs):
        calls.append(kwargs)
        return _make_result([_make_candidate("c1", "chunk text")])

    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        fake_query_sync,
    )

    config = FakeConfig()
    embedder = object()
    collection = object()

    tool = QueryKnowledgeBaseTool(config=config, embedder=embedder, collection=collection)
    tool.execute({"query": "test question"})

    assert len(calls) == 1
    assert calls[0]["config"] is config
    assert calls[0]["embedder"] is embedder
    assert calls[0]["collection"] is collection
    assert calls[0]["question"] == "test question"


def test_fresh_session_state_per_call(monkeypatch) -> None:
    """A new SessionState is created on each execute() call."""
    states: list = []

    def fake_query_sync(**kwargs):
        states.append(kwargs["state"])
        return _make_result([])

    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        fake_query_sync,
    )

    tool = QueryKnowledgeBaseTool(config=FakeConfig(), embedder=None, collection=None)
    tool.execute({"query": "q1"})
    tool.execute({"query": "q2"})

    assert len(states) == 2
    assert states[0] is not states[1], "SessionState must be freshly created per call"


def test_result_json_contains_answer_and_sources_display(monkeypatch) -> None:
    """Output JSON contains 'answer' and 'sources_display' keys."""
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda **kwargs: _make_result(
            [_make_candidate("c1", "text")],
            answer="Grounded answer",
        ),
    )

    tool = QueryKnowledgeBaseTool(config=FakeConfig(), embedder=None, collection=None)
    output = json.loads(tool.execute({"query": "question"}))

    assert "answer" in output
    assert "sources_display" in output
    assert "candidates" in output
    assert output["answer"] == "Grounded answer"


def test_missing_query_raises_value_error(monkeypatch) -> None:
    """execute() raises ValueError when 'query' key is missing."""
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda **kwargs: _make_result([]),
    )

    tool = QueryKnowledgeBaseTool(config=FakeConfig(), embedder=None, collection=None)
    with pytest.raises(ValueError, match="query is required"):
        tool.execute({})


def test_blank_query_raises_value_error(monkeypatch) -> None:
    """execute() raises ValueError when 'query' is blank after stripping."""
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda **kwargs: _make_result([]),
    )

    tool = QueryKnowledgeBaseTool(config=FakeConfig(), embedder=None, collection=None)
    with pytest.raises(ValueError, match="query is required"):
        tool.execute({"query": "   "})


def test_top_k_limits_candidates(monkeypatch) -> None:
    """Optional top_k caps the number of candidates returned in the output."""
    candidates = [_make_candidate(f"c{i}", f"text {i}") for i in range(10)]

    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda **kwargs: _make_result(candidates),
    )

    tool = QueryKnowledgeBaseTool(config=FakeConfig(), embedder=None, collection=None)
    output = json.loads(tool.execute({"query": "question", "top_k": 3}))

    assert len(output["candidates"]) == 3
    assert output["candidates"][0]["id"] == "c0"


def test_filters_parameter_is_accepted(monkeypatch) -> None:
    """Optional 'filters' dict is accepted without error."""
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda **kwargs: _make_result([_make_candidate("c1", "text")]),
    )

    tool = QueryKnowledgeBaseTool(config=FakeConfig(), embedder=None, collection=None)
    output = json.loads(
        tool.execute({"query": "question", "filters": {"source_type": "paper"}})
    )

    assert "answer" in output
