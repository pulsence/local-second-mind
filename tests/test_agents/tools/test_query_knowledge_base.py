"""
Tests for QueryKnowledgeBaseTool.
"""
from __future__ import annotations

import json

import pytest

from lsm.agents.tools.query_knowledge_base import QueryKnowledgeBaseTool
from lsm.query.api import QueryResult
from lsm.query.session import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_result(n_candidates: int = 2) -> QueryResult:
    candidates = [
        Candidate(cid=f"c{i}", text=f"chunk text {i}" * 10, meta={}, distance=0.1 * i)
        for i in range(n_candidates)
    ]
    return QueryResult(
        answer="The answer is 42.",
        candidates=candidates,
        sources_display="[S1] doc.md",
        cost=0.0,
        remote_sources=[],
        debug_info={},
    )


def _make_tool() -> tuple[QueryKnowledgeBaseTool, object, object, object]:
    config = object()
    embedder = object()
    collection = object()
    tool = QueryKnowledgeBaseTool(config=config, embedder=embedder, collection=collection)
    return tool, config, embedder, collection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_execute_calls_query_sync_with_injected_dependencies(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_sync(question, config, state, embedder, collection, **kwargs):
        calls.append(
            {
                "question": question,
                "config": config,
                "embedder": embedder,
                "collection": collection,
                "state": state,
            }
        )
        return _fake_result()

    monkeypatch.setattr("lsm.agents.tools.query_knowledge_base.query_sync", fake_sync)

    tool, config, embedder, collection = _make_tool()
    tool.execute({"query": "what is Python?"})

    assert len(calls) == 1
    assert calls[0]["question"] == "what is Python?"
    assert calls[0]["config"] is config
    assert calls[0]["embedder"] is embedder
    assert calls[0]["collection"] is collection


def test_execute_constructs_fresh_session_state_per_call(monkeypatch) -> None:
    states: list[object] = []

    def fake_sync(question, config, state, embedder, collection, **kwargs):
        states.append(state)
        return _fake_result()

    monkeypatch.setattr("lsm.agents.tools.query_knowledge_base.query_sync", fake_sync)

    tool, *_ = _make_tool()
    tool.execute({"query": "first call"})
    tool.execute({"query": "second call"})

    assert len(states) == 2
    assert states[0] is not states[1], "SessionState must be a new instance on each call"


def test_execute_result_json_contains_answer_and_sources_display(monkeypatch) -> None:
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda *a, **kw: _fake_result(),
    )

    tool, *_ = _make_tool()
    output = json.loads(tool.execute({"query": "hello"}))

    assert "answer" in output
    assert output["answer"] == "The answer is 42."
    assert "sources_display" in output
    assert output["sources_display"] == "[S1] doc.md"


def test_execute_raises_value_error_when_query_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda *a, **kw: _fake_result(),
    )

    tool, *_ = _make_tool()
    with pytest.raises(ValueError):
        tool.execute({})


def test_execute_raises_value_error_when_query_blank(monkeypatch) -> None:
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda *a, **kw: _fake_result(),
    )

    tool, *_ = _make_tool()
    with pytest.raises(ValueError):
        tool.execute({"query": "   "})


def test_execute_top_k_limits_candidates_in_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda *a, **kw: _fake_result(n_candidates=5),
    )

    tool, *_ = _make_tool()
    output = json.loads(tool.execute({"query": "hello", "top_k": 2}))

    assert len(output["candidates"]) == 2


def test_execute_accepts_filters_and_returns_valid_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "lsm.agents.tools.query_knowledge_base.query_sync",
        lambda *a, **kw: _fake_result(),
    )

    tool, *_ = _make_tool()
    output = json.loads(tool.execute({"query": "hello", "filters": {"path_contains": "/docs"}}))

    assert "answer" in output
    assert "sources_display" in output
