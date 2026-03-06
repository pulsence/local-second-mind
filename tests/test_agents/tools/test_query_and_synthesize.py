"""
Real execution tests for QueryAndSynthesizeTool.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsm.agents.tools.query_and_synthesize import QueryAndSynthesizeTool
from lsm.config.loader import build_config_from_raw
from lsm.query.pipeline import RetrievalPipeline
from lsm.vectordb.base import VectorDBGetResult, VectorDBQueryResult


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "path": str(tmp_path / "data"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {
                "default": {"provider": "openai", "model": "gpt-5.2"},
                "query": {"provider": "openai", "model": "gpt-5.2"},
            },
        },
        "db": {
            "path": str(tmp_path / "data"),
            "vector": {
                "provider": "sqlite",
                "collection": "local_kb",
            },
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": True,
            },
        },
    }


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        _ = kwargs
        if isinstance(texts, str):
            texts = [texts]
        return [[0.4, 0.6] for _ in texts]


class FakeVectorCollection:
    def __init__(self) -> None:
        self.last_filters = None
        self.last_top_k = None

    def query(self, embedding, top_k, filters=None):
        _ = embedding
        self.last_filters = filters
        self.last_top_k = top_k
        return VectorDBQueryResult(
            ids=["chunk-1"],
            documents=["Aquinas treats sin as a voluntary act contrary to reason and divine law."],
            metadatas=[
                {
                    "source_path": "/docs/aquinas.md",
                    "source_name": "aquinas.md",
                    "heading": "Sin",
                    "heading_text": "Sin",
                    "author": "Thomas Aquinas",
                    "year": "1273",
                    "is_current": True,
                }
            ],
            distances=[0.05],
        )

    def get(self, ids=None, filters=None, limit=None, offset=0, include=None):
        _ = ids, limit, offset, include
        metadata = {
            "source_path": "/docs/aquinas.md",
            "source_name": "aquinas.md",
            "heading": "Sin",
            "heading_text": "Sin",
            "author": "Thomas Aquinas",
            "year": "1273",
            "is_current": True,
        }
        if filters and filters.get("source_path") == "/docs/aquinas.md":
            return VectorDBGetResult(
                ids=["chunk-1"],
                documents=["Aquinas treats sin as a voluntary act contrary to reason and divine law."],
                metadatas=[metadata],
                embeddings=[[0.4, 0.6]],
            )
        return VectorDBGetResult(ids=["chunk-1"], metadatas=[metadata])


class FakeProvider:
    name = "openai"
    model = "gpt-5.2"

    def __init__(self) -> None:
        self.last_response_id = "resp-001"
        self.calls: list[dict] = []

    def send_message(self, **kwargs):
        self.calls.append(kwargs)
        return "Sin is a voluntary act opposed to right reason and divine law. [S1]"

    def estimate_cost(self, input_tokens, output_tokens):
        _ = input_tokens, output_tokens
        return 0.0


def _make_tool(tmp_path: Path) -> tuple[QueryAndSynthesizeTool, FakeVectorCollection, FakeProvider]:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    collection = FakeVectorCollection()
    provider = FakeProvider()
    pipeline = RetrievalPipeline(
        db=collection,
        embedder=FakeEmbedder(),
        config=config,
        llm_provider=provider,
    )
    tool = QueryAndSynthesizeTool(
        pipeline=pipeline,
        sandbox_config=config.agents.sandbox if config.agents else None,
    )
    return tool, collection, provider


def test_returns_complete_query_response_from_real_pipeline(tmp_path: Path) -> None:
    tool, _, provider = _make_tool(tmp_path)

    output = json.loads(tool.execute({"query": "What is sin?", "k": 1}))

    assert output["answer"].startswith("Sin is a voluntary act")
    assert output["response_id"] == "resp-001"
    assert len(output["candidates"]) == 1
    assert output["candidates"][0]["id"] == "chunk-1"
    assert output["candidates"][0]["source_path"] == "/docs/aquinas.md"
    assert provider.calls


def test_query_request_fields_propagate_through_real_pipeline(tmp_path: Path) -> None:
    tool, collection, provider = _make_tool(tmp_path)

    output = json.loads(
        tool.execute(
            {
                "query": "What is sin?",
                "mode": "grounded",
                "k": 3,
                "filters": {"path_contains": ["/docs"], "ext_deny": [".pdf"]},
                "starting_prompt": "Answer briefly.",
                "conversation_id": "conv-1",
                "prior_response_id": "prev-1",
            }
        )
    )

    assert output["conversation_id"] == "conv-1"
    assert collection.last_top_k >= 3
    assert collection.last_filters is not None
    assert collection.last_filters.get("is_current") is True
    assert provider.calls[0]["previous_response_id"] == "prev-1"
    assert provider.calls[0]["instruction"] == "Answer briefly."


def test_query_required(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)

    with pytest.raises(ValueError, match="query is required"):
        tool.execute({"query": ""})


def test_tool_surfaces_pipeline_provider_failures_as_fallback_answers(tmp_path: Path) -> None:
    tool, _, provider = _make_tool(tmp_path)

    def _raise(**kwargs):
        _ = kwargs
        raise RuntimeError("transport down")

    provider.send_message = _raise

    output = json.loads(tool.execute({"query": "What is sin?"}))

    assert "most relevant excerpts" in output["answer"].lower() or "closest excerpts" in output["answer"].lower()
    assert output["candidates"]
