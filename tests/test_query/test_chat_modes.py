from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

from lsm.config.models import (
    GlobalConfig,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LSMConfig,
    ModeChatsConfig,
    ModeConfig,
    QueryConfig,
    VectorDBConfig,
)
from lsm.query.context import ContextResult
from lsm.query.pipeline_types import (
    ContextPackage,
    CostEntry,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
)
from lsm.query.planning import LocalQueryPlan
from lsm.query.session import Candidate, SessionState


def _make_config(
    tmp_path: Path,
    *,
    chat_mode: str = "single",
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
            enable_llm_server_cache=llm_server_cache_enabled,
        ),
        vectordb=VectorDBConfig(provider="sqlite", path=tmp_path / "data", collection="kb"),
        global_settings=GlobalConfig(global_folder=tmp_path / "global"),
        config_path=tmp_path / "config.json",
    )


def _make_candidate():
    return Candidate(
        cid="c1", text="Context", meta={"source_path": "/docs/a.md"}, distance=0.1
    )


def _mock_pipeline_run(candidate, response_id=None):
    """Return a mock pipeline.run() that returns a valid QueryResponse."""
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
            context_block="[S1] Context",
            source_labels={"S1": {"source_path": "/docs/a.md"}},
            starting_prompt="Answer.",
        )
        return QueryResponse(
            answer="Chat answer [S1]",
            package=pkg,
            costs=[CostEntry(provider="openai", model="gpt-5.2", cost=0.0)],
            conversation_id=request.conversation_id,
            response_id=response_id,
        )
    return _run


def test_chat_mode_appends_conversation_history(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat")
    config.chats.auto_save = False
    state = SessionState(model="gpt-5.2")
    candidate = _make_candidate()

    with patch("lsm.query.api.RetrievalPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.side_effect = _mock_pipeline_run(candidate)

        asyncio.run(qapi.query("Hello there", config, state, embedder=Mock(), collection=Mock()))

    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["role"] == "user"
    assert state.conversation_history[1]["role"] == "assistant"


def test_chat_mode_passes_previous_response_id_when_llm_server_cache_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    from lsm.query import api as qapi

    config = _make_config(
        tmp_path,
        chat_mode="chat",
        llm_server_cache_enabled=True,
    )
    config.chats.auto_save = False
    state = SessionState(model="gpt-5.2")
    candidate = _make_candidate()

    captured_requests = []

    def _tracking_run(request, progress_callback=None):
        captured_requests.append(request)
        resp_id = f"resp-{len(captured_requests)}"
        pkg = ContextPackage(
            request=request,
            candidates=[candidate],
            remote_sources=[],
            retrieval_trace=RetrievalTrace(stages_executed=["local_retrieval"]),
            all_candidates=[candidate],
            filtered_candidates=[candidate],
            relevance=0.9,
            local_enabled=True,
            context_block="[S1] Context",
            source_labels={"S1": {}},
            starting_prompt="Answer.",
        )
        return QueryResponse(
            answer="Chat answer [S1]",
            package=pkg,
            costs=[CostEntry(provider="openai", model="gpt-5.2", cost=0.0)],
            conversation_id=request.conversation_id,
            response_id=resp_id,
        )

    with patch("lsm.query.api.RetrievalPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.side_effect = _tracking_run

        asyncio.run(qapi.query("first turn", config, state, embedder=Mock(), collection=Mock()))
        asyncio.run(qapi.query("second turn", config, state, embedder=Mock(), collection=Mock()))

    # First turn: no prior_response_id
    assert captured_requests[0].prior_response_id is None
    # Second turn: should have the resp-1 from first turn
    assert captured_requests[1].prior_response_id == "resp-1"


def test_chat_mode_respects_mode_auto_save_override_off(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat")
    config.chats.auto_save = True
    config.modes = {
        "grounded": ModeConfig(
            synthesis_style="grounded",
            chats=ModeChatsConfig(auto_save=False),
        )
    }
    state = SessionState(model="gpt-5.2")
    candidate = _make_candidate()

    saved = {"count": 0}

    def _fake_save(*args, **kwargs):
        saved["count"] += 1

    monkeypatch.setattr(qapi, "save_conversation_markdown", _fake_save)

    with patch("lsm.query.api.RetrievalPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.side_effect = _mock_pipeline_run(candidate)

        asyncio.run(qapi.query("Hello there", config, state, embedder=Mock(), collection=Mock()))

    assert saved["count"] == 0


def test_chat_mode_respects_mode_chat_dir_override(monkeypatch, tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat")
    config.chats.auto_save = True
    config.chats.dir = "Chats"
    config.modes = {
        "grounded": ModeConfig(
            synthesis_style="grounded",
            chats=ModeChatsConfig(dir="Chats/CustomModeFolder"),
        )
    }
    state = SessionState(model="gpt-5.2")
    candidate = _make_candidate()

    captured = {}

    def _fake_save(state, chats_dir, mode_name):
        captured["chats_dir"] = chats_dir
        captured["mode_name"] = mode_name

    monkeypatch.setattr(qapi, "save_conversation_markdown", _fake_save)

    with patch("lsm.query.api.RetrievalPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.run.side_effect = _mock_pipeline_run(candidate)

        asyncio.run(qapi.query("Hello there", config, state, embedder=Mock(), collection=Mock()))

    assert captured["mode_name"] == "grounded"
    assert str(captured["chats_dir"]).endswith(
        str(Path("Chats") / "CustomModeFolder" / "grounded")
    )


def test_conversation_invalidation_on_model_provider_mode_switch(tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="chat")
    state = SessionState(model="gpt-5.2")
    state.conversation_id = "conv-1"
    state.prior_response_id = "resp-1"

    # First call initializes conversation key only.
    qapi._check_conversation_invalidation(config, state)
    assert state.conversation_id == "conv-1"
    assert state.prior_response_id == "resp-1"

    # Model switch should invalidate.
    state.model = "gpt-4.1"
    qapi._check_conversation_invalidation(config, state)
    assert state.conversation_id is None
    assert state.prior_response_id is None

    # Re-seed state then switch provider.
    state.conversation_id = "conv-2"
    state.prior_response_id = "resp-2"
    state.model = "gpt-4.1"
    config.llm.providers.append(LLMProviderConfig(provider_name="openrouter", api_key="test"))
    config.llm.services["query"] = LLMServiceConfig(provider="openrouter", model="gpt-4.1")
    qapi._check_conversation_invalidation(config, state)
    assert state.conversation_id is None
    assert state.prior_response_id is None

    # Re-seed state then switch mode.
    state.conversation_id = "conv-3"
    state.prior_response_id = "resp-3"
    config.query.mode = "insight"
    config.modes["insight"] = ModeConfig(synthesis_style="insight")
    qapi._check_conversation_invalidation(config, state)
    assert state.conversation_id is None
    assert state.prior_response_id is None


def test_conversation_invalidation_in_single_chat_mode(tmp_path: Path) -> None:
    from lsm.query import api as qapi

    config = _make_config(tmp_path, chat_mode="single")
    state = SessionState(model="gpt-5.2")
    state.conversation_id = "conv-1"
    state.prior_response_id = "resp-1"

    qapi._check_conversation_invalidation(config, state)
    assert state.conversation_id is None
    assert state.prior_response_id is None
