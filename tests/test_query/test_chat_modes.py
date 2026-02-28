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
    ModeChatsConfig,
    ModeConfig,
    SourcePolicyConfig,
    QueryConfig,
    VectorDBConfig,
)
from lsm.query.context import ContextResult
from lsm.query.planning import LocalQueryPlan
from lsm.query.session import Candidate, SessionState


def _make_config(
    tmp_path: Path,
    *,
    chat_mode: str = "single",
    cache_enabled: bool = False,
    llm_server_cache_enabled: bool = False,
) -> LSMConfig:
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
            chat_mode=chat_mode,
            enable_query_cache=cache_enabled,
            query_cache_ttl=60,
            query_cache_size=10,
            enable_llm_server_cache=llm_server_cache_enabled,
        ),
        vectordb=VectorDBConfig(provider="sqlite", path=tmp_path / "data", collection="kb"),
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


def test_chat_mode_passes_previous_response_id_when_llm_server_cache_enabled(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(
        tmp_path,
        chat_mode="chat",
        cache_enabled=False,
        llm_server_cache_enabled=True,
    )
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

    captured = []
    provider = Mock()

    def _synth(*args, **kwargs):
        captured.append(kwargs.get("previous_response_id"))
        provider.last_response_id = f"resp-{len(captured)}"
        return "Chat answer [S1]"

    provider.synthesize.side_effect = _synth
    provider.estimate_cost.return_value = 0.0
    provider.name = "openai"
    provider.model = "gpt-5.2"
    provider.last_response_id = None

    monkeypatch.setattr(qapi, "build_combined_context_async", _fake_context)
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: provider)

    asyncio.run(qapi.query("first turn", config, state, embedder=Mock(), collection=Mock()))
    asyncio.run(qapi.query("second turn", config, state, embedder=Mock(), collection=Mock()))

    assert captured[0] is None
    assert captured[1] == "resp-1"


def test_chat_mode_respects_mode_auto_save_override_off(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat", cache_enabled=False)
    config.chats.auto_save = True
    config.modes = {
        "grounded": ModeConfig(
            synthesis_style="grounded",
            source_policy=SourcePolicyConfig(),
            chats=ModeChatsConfig(auto_save=False),
        )
    }
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

    saved = {"count": 0}

    def _fake_save(*args, **kwargs):
        saved["count"] += 1

    monkeypatch.setattr(qapi, "build_combined_context_async", _fake_context)
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: provider)
    monkeypatch.setattr(qapi, "save_conversation_markdown", _fake_save)

    asyncio.run(qapi.query("Hello there", config, state, embedder=Mock(), collection=Mock()))
    assert saved["count"] == 0


def test_chat_mode_respects_mode_chat_dir_override(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat", cache_enabled=False)
    config.chats.auto_save = True
    config.chats.dir = "Chats"
    config.modes = {
        "grounded": ModeConfig(
            synthesis_style="grounded",
            source_policy=SourcePolicyConfig(),
            chats=ModeChatsConfig(dir="Chats/CustomModeFolder"),
        )
    }
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

    captured = {}

    def _fake_save(state, chats_dir, mode_name):
        captured["chats_dir"] = chats_dir
        captured["mode_name"] = mode_name

    monkeypatch.setattr(qapi, "build_combined_context_async", _fake_context)
    monkeypatch.setattr(qapi, "create_provider", lambda cfg: provider)
    monkeypatch.setattr(qapi, "save_conversation_markdown", _fake_save)

    asyncio.run(qapi.query("Hello there", config, state, embedder=Mock(), collection=Mock()))
    assert captured["mode_name"] == "grounded"
    assert str(captured["chats_dir"]).endswith(str(Path("Chats") / "CustomModeFolder" / "grounded"))
