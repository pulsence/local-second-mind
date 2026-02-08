from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
from lsm.query.context import ContextResult
from lsm.query.planning import LocalQueryPlan
from lsm.query.session import Candidate, SessionState


def _make_config(tmp_path: Path, *, chat_mode: str = "single", cache_enabled: bool = False) -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(roots=[tmp_path], manifest=tmp_path / ".ingest" / "manifest.json"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={
                "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
                "ranking": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            },
        ),
        query=QueryConfig(
            mode="grounded",
            chat_mode=chat_mode,
            enable_query_cache=cache_enabled,
            query_cache_ttl=60,
            query_cache_size=10,
        ),
        vectordb=VectorDBConfig(persist_dir=tmp_path / ".chroma", collection="kb"),
        global_settings=GlobalConfig(global_folder=tmp_path / "global"),
        config_path=tmp_path / "config.json",
    )


def test_query_cache_short_circuits_second_call(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="single", cache_enabled=True)
    state = SessionState(model="gpt-5.2")
    candidate = Candidate(cid="c1", text="Python", meta={"source_path": "/docs/a.md"}, distance=0.1)
    plan = LocalQueryPlan(
        local_enabled=True,
        candidates=[candidate],
        filtered=[candidate],
        relevance=0.9,
        filters_active=False,
        retrieve_k=12,
        rerank_strategy="none",
        should_llm_rerank=False,
        k=12,
        k_rerank=6,
        min_relevance=0.25,
        max_per_file=2,
        local_pool=36,
        no_rerank=True,
    )

    calls = {"context": 0, "synthesize": 0}

    async def _fake_context(*args, **kwargs):
        calls["context"] += 1
        return ContextResult(
            candidates=[candidate],
            context_block="[S1] Python",
            sources=[{"label": "S1"}],
            local_candidates=[candidate],
            remote_candidates=[],
            remote_sources=[],
            plan=plan,
        )

    provider = Mock()
    provider.synthesize.side_effect = lambda *a, **k: calls.__setitem__("synthesize", calls["synthesize"] + 1) or "Answer [S1]"
    provider.estimate_cost.return_value = 0.0
    provider.name = "openai"
    provider.model = "gpt-5.2"

    monkeypatch.setattr(qapi, "build_combined_context_async", _fake_context)
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: provider)

    asyncio.run(qapi.query("What is Python?", config, state, embedder=Mock(), collection=Mock()))
    asyncio.run(qapi.query("What is Python?", config, state, embedder=Mock(), collection=Mock()))

    assert calls["context"] == 1
    assert calls["synthesize"] == 1


def test_chat_mode_appends_conversation_history(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat", cache_enabled=False)
    config.chats.auto_save = False
    state = SessionState(model="gpt-5.2")
    candidate = Candidate(cid="c1", text="Context", meta={"source_path": "/docs/a.md"}, distance=0.1)
    plan = LocalQueryPlan(
        local_enabled=True,
        candidates=[candidate],
        filtered=[candidate],
        relevance=0.9,
        filters_active=False,
        retrieve_k=12,
        rerank_strategy="none",
        should_llm_rerank=False,
        k=12,
        k_rerank=6,
        min_relevance=0.25,
        max_per_file=2,
        local_pool=36,
        no_rerank=True,
    )

    async def _fake_context(*args, **kwargs):
        return ContextResult(
            candidates=[candidate],
            context_block="[S1] Context",
            sources=[{"label": "S1"}],
            local_candidates=[candidate],
            remote_candidates=[],
            remote_sources=[],
            plan=plan,
        )

    provider = Mock()
    provider.synthesize.return_value = "Chat answer [S1]"
    provider.estimate_cost.return_value = 0.0
    provider.name = "openai"
    provider.model = "gpt-5.2"

    monkeypatch.setattr(qapi, "build_combined_context_async", _fake_context)
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: provider)

    asyncio.run(qapi.query("Hello there", config, state, embedder=Mock(), collection=Mock()))

    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["role"] == "user"
    assert state.conversation_history[1]["role"] == "assistant"
