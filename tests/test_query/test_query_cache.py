from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

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
    QueryResponse,
    RetrievalTrace,
)
from lsm.query.session import Candidate, SessionState


def _make_config(tmp_path: Path) -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(roots=[tmp_path]),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={
                "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
                "ranking": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            },
        ),
        query=QueryConfig(
            mode="grounded",
            chat_mode="single",
        ),
        vectordb=VectorDBConfig(provider="sqlite", path=tmp_path / "data", collection="kb"),
        global_settings=GlobalConfig(global_folder=tmp_path / "global"),
        config_path=tmp_path / "config.json",
    )


def _make_candidate() -> Candidate:
    return Candidate(
        cid="c1",
        text="cached context",
        meta={"source_path": "/docs/a.md"},
        distance=0.1,
    )


def _pipeline_response(candidate: Candidate, request) -> QueryResponse:
    package = ContextPackage(
        request=request,
        candidates=[candidate],
        remote_sources=[],
        retrieval_trace=RetrievalTrace(stages_executed=["local_retrieval"]),
        all_candidates=[candidate],
        filtered_candidates=[candidate],
        relevance=0.9,
        local_enabled=True,
        context_block="[S1] cached context",
        source_labels={"S1": {}},
        starting_prompt="Answer.",
    )
    return QueryResponse(
        answer="Cached answer [S1]",
        package=package,
        costs=[CostEntry(provider="openai", model="gpt-5.2", cost=0.0)],
    )


def test_query_cache_hit_short_circuits_pipeline(tmp_path: Path) -> None:
    from lsm.query import api as qapi

    qapi.clear_query_caches()
    config = _make_config(tmp_path)
    config.query.enable_query_cache = True
    config.query.query_cache_ttl = 60
    config.query.query_cache_size = 16
    state = SessionState(model="gpt-5.2")
    candidate = _make_candidate()
    calls = {"run": 0}

    def _run(request, progress_callback=None):
        calls["run"] += 1
        return _pipeline_response(candidate, request)

    with patch("lsm.query.api.RetrievalPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.side_effect = _run

        first = asyncio.run(qapi.query("What is cache?", config, state, embedder=Mock(), collection=Mock()))
        second = asyncio.run(qapi.query("What is cache?", config, state, embedder=Mock(), collection=Mock()))

    assert first.answer == second.answer
    assert calls["run"] == 1
    assert state.last_debug.get("cache_hit") is True


def test_clear_query_caches_for_config_invalidates_entries(tmp_path: Path) -> None:
    from lsm.query import api as qapi

    qapi.clear_query_caches()
    config = _make_config(tmp_path)
    config.query.enable_query_cache = True
    config.query.query_cache_ttl = 60
    config.query.query_cache_size = 16
    state = SessionState(model="gpt-5.2")
    candidate = _make_candidate()
    calls = {"run": 0}

    def _run(request, progress_callback=None):
        calls["run"] += 1
        return _pipeline_response(candidate, request)

    with patch("lsm.query.api.RetrievalPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.side_effect = _run

        asyncio.run(qapi.query("Q1", config, state, embedder=Mock(), collection=Mock()))
        removed = qapi.clear_query_caches(config)
        asyncio.run(qapi.query("Q1", config, state, embedder=Mock(), collection=Mock()))

    assert removed >= 1
    assert calls["run"] == 2
