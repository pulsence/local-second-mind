"""
Integration tests for query module.

These tests use lightweight fakes plus real query orchestration code.
"""

from __future__ import annotations

import asyncio
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

    def rerank(self, _question: str, candidates: list[dict], k: int) -> list[dict]:
        if self._rerank_result is not None:
            return self._rerank_result[:k]
        return candidates[:k]

    def synthesize(self, *_args, **_kwargs) -> str:
        if self._fail_synthesize:
            raise RuntimeError("llm unavailable")
        return self._answer

    def estimate_cost(self, _input_tokens: int, _output_tokens: int) -> float:
        return 0.0


def _build_config(tmp_path: Path, *, mode: str = "grounded") -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(
            roots=[tmp_path],
            manifest=tmp_path / ".ingest" / "manifest.json",
        ),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test-key")],
            services={
                "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
                "ranking": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            },
        ),
        query=QueryConfig(
            k=12,
            k_rerank=6,
            no_rerank=False,
            mode=mode,
        ),
        vectordb=VectorDBConfig(
            path=tmp_path / ".chroma",
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


def _build_plan(candidate: Candidate, *, should_llm_rerank: bool, no_rerank: bool) -> LocalQueryPlan:
    return LocalQueryPlan(
        local_enabled=True,
        candidates=[candidate],
        filtered=[candidate],
        relevance=0.9,
        filters_active=False,
        retrieve_k=12,
        rerank_strategy="llm" if should_llm_rerank else "none",
        should_llm_rerank=should_llm_rerank,
        k=12,
        k_rerank=6,
        min_relevance=0.3,
        max_per_file=2,
        local_pool=36,
        no_rerank=no_rerank,
    )


@pytest.mark.integration
def test_query_api_returns_result(tmp_path: Path, monkeypatch) -> None:
    from lsm.query.api import QueryResult, query

    config = _build_config(tmp_path)
    candidate = Candidate(
        cid="1",
        text="Python is a programming language",
        meta={"source_path": "/docs/python.md", "chunk_index": 0},
        distance=0.1,
    )
    local_plan = _build_plan(candidate, should_llm_rerank=True, no_rerank=False)

    monkeypatch.setattr(
        "lsm.query.context.prepare_local_candidates",
        lambda *_args, **_kwargs: local_plan,
    )

    reranked = [
        {
            "cid": "1",
            "text": candidate.text,
            "metadata": candidate.meta,
            "distance": candidate.distance,
        }
    ]
    provider_iter = iter(
        [
            FakeLLMProvider(rerank_result=reranked),
            FakeLLMProvider(answer="Python is a programming language [S1]."),
        ]
    )
    monkeypatch.setattr("lsm.query.api.create_provider", lambda _cfg: next(provider_iter))

    state = SessionState(model="gpt-5.2")
    result = asyncio.run(
        query(
            "What is Python?",
            config,
            state,
            embedder=object(),
            collection=object(),
        )
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
        max_per_file=2,
        local_pool=10,
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
def test_query_with_no_candidates(tmp_path: Path, monkeypatch) -> None:
    from lsm.query.api import QueryResult, query

    config = _build_config(tmp_path)
    local_plan = LocalQueryPlan(
        local_enabled=True,
        candidates=[],
        filtered=[],
        relevance=0.0,
        filters_active=False,
        retrieve_k=12,
        rerank_strategy="none",
        should_llm_rerank=False,
        k=12,
        k_rerank=6,
        min_relevance=0.3,
        max_per_file=2,
        local_pool=36,
        no_rerank=True,
    )
    monkeypatch.setattr(
        "lsm.query.context.prepare_local_candidates",
        lambda *_args, **_kwargs: local_plan,
    )

    result = asyncio.run(
        query(
            "Test question",
            config,
            SessionState(model="gpt-5.2"),
            object(),
            object(),
        )
    )

    assert isinstance(result, QueryResult)
    assert "No results found" in result.answer
    assert len(result.candidates) == 0


@pytest.mark.integration
def test_query_returns_fallback_when_synthesis_fails(tmp_path: Path, monkeypatch) -> None:
    from lsm.query.api import query

    config = _build_config(tmp_path, mode="grounded")
    candidate = Candidate(
        cid="1",
        text="Python is a programming language",
        meta={"source_path": "/docs/python.md", "chunk_index": 0},
        distance=0.1,
    )
    local_plan = _build_plan(candidate, should_llm_rerank=True, no_rerank=False)
    monkeypatch.setattr(
        "lsm.query.context.prepare_local_candidates",
        lambda *_args, **_kwargs: local_plan,
    )

    reranked = [
        {
            "cid": "1",
            "text": candidate.text,
            "metadata": candidate.meta,
            "distance": candidate.distance,
        }
    ]
    provider_iter = iter(
        [
            FakeLLMProvider(rerank_result=reranked),
            FakeLLMProvider(fail_synthesize=True),
        ]
    )
    monkeypatch.setattr("lsm.query.api.create_provider", lambda _cfg: next(provider_iter))

    result = asyncio.run(query("What is Python?", config, SessionState(), object(), object()))

    assert "Top excerpts:" in result.answer
    assert result.debug_info.get("synthesis_fallback") is True
