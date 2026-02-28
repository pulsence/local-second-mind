"""Tests for query LLM rerank stage helpers."""

from __future__ import annotations

import json

from lsm.query.stages.llm_rerank import (
    llm_rerank,
    parse_ranking_response,
    prepare_candidates_for_rerank,
)


class _Provider:
    name = "mock"
    model = "mock-model"

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def send_message(self, **_kwargs) -> str:
        return self._payload


def _candidates() -> list[dict]:
    return [
        {
            "text": "first",
            "metadata": {"source_path": "/tmp/one.md", "chunk_index": 0},
            "distance": 0.1,
        },
        {
            "text": "second",
            "metadata": {"source_path": "/tmp/two.md", "chunk_index": 1},
            "distance": 0.2,
        },
        {
            "text": "third",
            "metadata": {"source_path": "/tmp/three.md", "chunk_index": 2},
            "distance": 0.3,
        },
    ]


def test_prepare_candidates_truncates_long_text() -> None:
    candidates = [
        {
            "text": "x" * 1300,
            "metadata": {"source_path": "/tmp/source.md", "chunk_index": 2},
            "distance": 0.25,
        }
    ]
    items = prepare_candidates_for_rerank(candidates, max_text_length=1200)

    assert len(items) == 1
    assert items[0]["index"] == 0
    assert items[0]["source_path"] == "/tmp/source.md"
    assert items[0]["text"].endswith("\n...[truncated]...")
    assert len(items[0]["text"]) <= 1200


def test_parse_ranking_response_handles_invalid_indices() -> None:
    ranking = [
        {"index": 50, "reason": "bad"},
        {"index": 1, "reason": "good"},
        {"index": "2", "reason": "also good"},
        {"index": 1, "reason": "duplicate"},
    ]

    result = parse_ranking_response(ranking, _candidates(), k=3)
    assert [item["text"] for item in result] == ["second", "third", "first"]


def test_llm_rerank_returns_ranked_candidates() -> None:
    provider = _Provider(json.dumps({"ranking": [{"index": 2, "reason": "best"}]}))
    result = llm_rerank(_candidates(), "question", provider, k=2)
    assert [item["text"] for item in result] == ["third", "first"]


def test_llm_rerank_falls_back_on_invalid_response() -> None:
    provider = _Provider(json.dumps({"oops": []}))
    result = llm_rerank(_candidates(), "question", provider, k=2)
    assert [item["text"] for item in result] == ["first", "second"]
