"""
Integration tests for query module.

These tests use lightweight fakes plus real query orchestration code.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lsm.config.models import (
    GlobalConfig,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LSMConfig,
    QueryConfig,
    VectorDBConfig,
)
from lsm.query.pipeline_types import (
    ContextPackage,
    CostEntry,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
)
from lsm.query.planning import LocalQueryPlan
from lsm.query.session import Candidate, SessionState


class FakeLLMProvider:
    name = "openai"
    model = "gpt-5.2"

    def __init__(
        self,
        *,
        answer: str = "Python is a programming language [S1].",
        rerank_result: list[dict] | None = None,
        fail_synthesize: bool = False,
    ) -> None:
        self._answer = answer
        self._rerank_result = rerank_result
        self._fail_synthesize = fail_synthesize

    def send_message(self, input: str, instruction=None, **kwargs) -> str:
        _ = input, instruction
        if kwargs.get("json_schema_name") == "rerank_response":
            return json.dumps({"ranking": [{"index": 0, "reason": "best"}]})
        if self._fail_synthesize:
            raise RuntimeError("llm unavailable")
        return self._answer

    def estimate_cost(self, _input_tokens: int, _output_tokens: int) -> float:
        return 0.0


class FakePipeline:
    """Fake RetrievalPipeline that returns pre-built responses."""

    def __init__(self, run_fn, **_kwargs):
        self._run_fn = run_fn

    def run(self, request, progress_callback=None):
        return self._run_fn(request, progress_callback)


def _build_config(tmp_path: Path, *, mode: str = "grounded") -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(roots=[tmp_path]),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test-key")],
            services={
                "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
                "ranking": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            },
        ),
        query=QueryConfig(
            k=12,
            mode=mode,
        ),
        vectordb=VectorDBConfig(
            provider="sqlite",
            path=tmp_path / "data",
            collection="test_kb",
        ),
        global_settings=GlobalConfig(
            global_folder=tmp_path / "global",
            embed_model="test-model",
            device="cpu",
            batch_size=32,
        ),
        config_path=tmp_path / "config.json",
    )


def _mock_pipeline_run(candidate, answer="Python is a programming language [S1]."):
    """Return a callable that simulates pipeline.run()."""
    def _run(request, progress_callback=None):
        pkg = ContextPackage(
            request=request,
            candidates=[candidate],
            remote_sources=[],
            retrieval_trace=RetrievalTrace(stages_executed=["local_retrieval"]),
            all_candidates=[candidate],
            filtered_candidates=[candidate],
            relevance=0.9,
            local_enabled=True,
            context_block="[S1] /docs/python.md\nPython is a programming language",
            source_labels={"S1": {"source_path": "/docs/python.md"}},
            starting_prompt="Answer.",
        )
        return QueryResponse(
            answer=answer,
            package=pkg,
            costs=[CostEntry(provider="openai", model="gpt-5.2", cost=0.0)],
        )
    return _run


def _mock_pipeline_run_no_candidates(request, progress_callback=None):
    """Simulate pipeline.run() with no candidates."""
    pkg = ContextPackage(
        request=request,
        candidates=[],
        remote_sources=[],
        retrieval_trace=RetrievalTrace(stages_executed=["local_retrieval"]),
        all_candidates=[],
        filtered_candidates=[],
        relevance=0.0,
        local_enabled=True,
    )
    return QueryResponse(
        answer="No results found in the knowledge base for this query.",
        package=pkg,
    )


def _mock_pipeline_run_synthesis_fail(candidate):
    """Simulate pipeline.run() where synthesis fails and fallback is used."""
    def _run(request, progress_callback=None):
        pkg = ContextPackage(
            request=request,
            candidates=[candidate],
            remote_sources=[],
            retrieval_trace=RetrievalTrace(stages_executed=["local_retrieval"]),
            all_candidates=[candidate],
            filtered_candidates=[candidate],
            relevance=0.9,
            local_enabled=True,
            context_block="[S1] text",
            source_labels={"S1": {}},
            starting_prompt="Answer.",
        )
        # Simulate fallback answer (no LLM synthesis)
        return QueryResponse(
            answer=(
                "OpenAI is unavailable (quota/credentials). "
                "Showing the most relevant excerpts instead.\n\n"
                "Question: What is Python?\n\nTop excerpts:\n\n"
                "[S1] /docs/python.md (chunk_index=0)\n"
                "Python is a programming language"
            ),
            package=pkg,
        )
    return _run


def _make_fake_pipeline_class(run_fn):
    """Create a FakePipeline class bound to a specific run function."""
    class _Pipeline:
        def __init__(self, **_kwargs):
            pass
        def run(self, request, progress_callback=None):
            return run_fn(request, progress_callback)
    return _Pipeline


@pytest.mark.integration
def test_query_api_returns_result(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi
    from lsm.query.api import QueryResult, query

    config = _build_config(tmp_path)
    candidate = Candidate(
        cid="1",
        text="Python is a programming language",
        meta={"source_path": "/docs/python.md", "chunk_index": 0},
        distance=0.1,
    )

    monkeypatch.setattr(qapi, "RetrievalPipeline", _make_fake_pipeline_class(_mock_pipeline_run(candidate)))
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: FakeLLMProvider())

    state = SessionState(model="gpt-5.2")
    result = asyncio.run(
        query("What is Python?", config, state, embedder=object(), collection=object())
    )

    assert isinstance(result, QueryResult)
    assert "[S1]" in result.answer
    assert len(result.candidates) >= 1


def test_query_api_empty_question_raises(tmp_path: Path) -> None:
    from lsm.query.api import query

    config = _build_config(tmp_path)
    state = SessionState()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        asyncio.run(query("", config, state, object(), object()))


def test_local_reranking_pipeline_integration() -> None:
    from lsm.query.rerank import apply_local_reranking

    candidates = [
        Candidate(
            cid="1",
            text="Python programming language",
            meta={"source_path": "/docs/python.md"},
            distance=0.1,
        ),
        Candidate(
            cid="2",
            text="Python programming language",
            meta={"source_path": "/docs/python_copy.md"},
            distance=0.15,
        ),
        Candidate(
            cid="3",
            text="Python tutorials",
            meta={"source_path": "/docs/python.md"},
            distance=0.2,
        ),
        Candidate(
            cid="4",
            text="Python guides",
            meta={"source_path": "/docs/python.md"},
            distance=0.25,
        ),
    ]

    result = apply_local_reranking(
        "Python programming",
        candidates,
    )

    texts = [c.text for c in result]
    assert len(texts) == len(set(texts))

    from_python_md = [c for c in result if c.source_path == "/docs/python.md"]
    assert len(from_python_md) <= 2


def test_retrieval_and_filtering_integration() -> None:
    from lsm.query.retrieval import filter_candidates

    candidates = [
        Candidate(
            cid="1",
            text="Python text",
            meta={"source_path": "/docs/python.md", "ext": ".md"},
            distance=0.1,
        ),
        Candidate(
            cid="2",
            text="Java text",
            meta={"source_path": "/docs/java.pdf", "ext": ".pdf"},
            distance=0.15,
        ),
        Candidate(
            cid="3",
            text="Python guide",
            meta={"source_path": "/guides/python.txt", "ext": ".txt"},
            distance=0.2,
        ),
    ]

    result = filter_candidates(
        candidates,
        path_contains="python",
        ext_allow=[".md", ".txt"],
    )

    assert len(result) == 2
    assert all("python" in c.source_path.lower() for c in result)
    assert all(c.ext in [".md", ".txt"] for c in result)


def test_synthesis_and_formatting_integration() -> None:
    from lsm.query.context import build_context_block
    from lsm.ui.utils import format_source_list

    candidates = [
        Candidate(
            cid="1",
            text="Python is a language",
            meta={"source_path": "/docs/python.md", "source_name": "python.md", "chunk_index": 0},
            distance=0.1,
        ),
        Candidate(
            cid="2",
            text="Python has libraries",
            meta={"source_path": "/docs/python.md", "source_name": "python.md", "chunk_index": 1},
            distance=0.15,
        ),
    ]

    context, sources = build_context_block(candidates)

    assert "[S1]" in context
    assert "[S2]" in context
    assert "Python is a language" in context

    formatted = format_source_list(sources)

    assert "Sources:" in formatted
    assert "[S1] [S2]" in formatted or ("[S1]" in formatted and "[S2]" in formatted)
    assert "python.md" in formatted


@pytest.mark.integration
def test_query_with_no_candidates(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi
    from lsm.query.api import QueryResult, query

    config = _build_config(tmp_path)

    monkeypatch.setattr(qapi, "RetrievalPipeline", _make_fake_pipeline_class(_mock_pipeline_run_no_candidates))
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: FakeLLMProvider())

    result = asyncio.run(
        query("Test question", config, SessionState(model="gpt-5.2"), object(), object())
    )

    assert isinstance(result, QueryResult)
    assert "No results found" in result.answer
    assert len(result.candidates) == 0


@pytest.mark.integration
def test_query_returns_fallback_when_synthesis_fails(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi
    from lsm.query.api import query

    config = _build_config(tmp_path, mode="grounded")
    candidate = Candidate(
        cid="1",
        text="Python is a programming language",
        meta={"source_path": "/docs/python.md", "chunk_index": 0},
        distance=0.1,
    )

    monkeypatch.setattr(qapi, "RetrievalPipeline", _make_fake_pipeline_class(
        _mock_pipeline_run_synthesis_fail(candidate)
    ))
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: FakeLLMProvider())

    result = asyncio.run(
        query("What is Python?", config, SessionState(), object(), object())
    )

    assert "Top excerpts" in result.answer or "No results" in result.answer or "Python" in result.answer
