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
    LocalSourcePolicy,
    ModeConfig,
    ModelKnowledgePolicy,
    QueryConfig,
    RemoteProviderConfig,
    RemoteSourcePolicy,
    VectorDBConfig,
)
from lsm.query.api import query
from lsm.query.planning import LocalQueryPlan
from lsm.query.session import Candidate, SessionState
from lsm.remote.base import RemoteResult


class FakeRerankProvider:
    name = "openai"
    model = "gpt-5.2"

    def send_message(self, input: str, instruction=None, **kwargs) -> str:
        _ = input, instruction, kwargs
        return json.dumps({"ranking": [{"index": 0, "reason": "best"}]})

    def estimate_cost(self, _input_tokens: int, _output_tokens: int) -> float:
        return 0.0


class FakeSynthesisProvider:
    name = "openai"
    model = "gpt-5.2"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def send_message(self, *_args, **_kwargs) -> str:
        return self._answer

    def estimate_cost(self, _input_tokens: int, _output_tokens: int) -> float:
        return 0.0


class FakeRemoteProvider:
    def search(self, _query: str, max_results: int = 3) -> list[RemoteResult]:
        return [
            RemoteResult(
                title="Remote Source",
                url="https://example.com/remote",
                snippet="Remote source snippet for integration progress test.",
                score=0.9,
            )
        ][:max_results]


def _build_query_config(tmp_path: Path, remote_enabled: bool) -> LSMConfig:
    mode_name = "integration_mode"
    return LSMConfig(
        ingest=IngestConfig(roots=[tmp_path]),
        query=QueryConfig(mode=mode_name, rerank_strategy="llm", no_rerank=False),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test-key")],
            services={
                "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
                "ranking": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            },
        ),
        vectordb=VectorDBConfig(
            provider="sqlite",
            path=tmp_path / "data",
            collection="test_collection",
        ),
        modes={
            mode_name: ModeConfig(
                retrieval_profile="hybrid_rrf",
                synthesis_style="grounded",
                local_policy=LocalSourcePolicy(enabled=True, min_relevance=0.0, k=2),
                remote_policy=RemoteSourcePolicy(
                    enabled=remote_enabled,
                    remote_providers=["mock_remote"] if remote_enabled else None,
                    max_results=3,
                ),
                model_knowledge_policy=ModelKnowledgePolicy(enabled=False),
            )
        },
        remote_providers=(
            [RemoteProviderConfig(name="mock_remote", type="mock", max_results=3, timeout=5)]
            if remote_enabled
            else None
        ),
        global_settings=GlobalConfig(global_folder=tmp_path / "global"),
    )


def _build_local_plan() -> LocalQueryPlan:
    candidate = Candidate(
        cid="local-1",
        text="Local source content for integration testing.",
        meta={"source_path": "/docs/local.md", "source_name": "local.md", "chunk_index": 0},
        distance=0.1,
    )
    return LocalQueryPlan(
        local_enabled=True,
        candidates=[candidate],
        filtered=[candidate],
        relevance=0.95,
        filters_active=False,
        retrieve_k=4,
        rerank_strategy="llm",
        should_llm_rerank=True,
        k=4,
        k_rerank=2,
        min_relevance=0.0,
        max_per_file=2,
        local_pool=8,
        no_rerank=False,
    )


@pytest.mark.integration
class TestQueryProgressCallbacks:
    def test_progress_callback_receives_all_stages(self, tmp_path: Path, monkeypatch) -> None:
        config = _build_query_config(tmp_path, remote_enabled=False)
        progress_updates = []
        local_plan = _build_local_plan()

        monkeypatch.setattr(
            "lsm.query.context.prepare_local_candidates",
            lambda *_args, **_kwargs: local_plan,
        )

        provider_iter = iter(
            [
                FakeRerankProvider(),
                FakeSynthesisProvider("Integrated answer [S1]."),
            ]
        )
        monkeypatch.setattr("lsm.query.api.create_provider", lambda _cfg: next(provider_iter))

        result = asyncio.run(
            query(
                "What is in the local source?",
                config,
                SessionState(),
                embedder=object(),
                collection=object(),
                progress_callback=lambda update: progress_updates.append(update),
            )
        )

        stages = [update.stage for update in progress_updates]
        assert "retrieval" in stages
        assert "rerank" in stages
        assert "synthesis" in stages
        assert stages.index("retrieval") < stages.index("rerank")
        assert stages.index("rerank") < stages.index("synthesis")
        assert "[S1]" in result.answer

    def test_progress_callback_with_remote_sources(self, tmp_path: Path, monkeypatch) -> None:
        config = _build_query_config(tmp_path, remote_enabled=True)
        progress_updates = []
        local_plan = _build_local_plan()

        monkeypatch.setattr(
            "lsm.query.context.prepare_local_candidates",
            lambda *_args, **_kwargs: local_plan,
        )
        monkeypatch.setattr(
            "lsm.query.context.create_remote_provider",
            lambda _provider_type, _runtime_cfg: FakeRemoteProvider(),
        )

        provider_iter = iter(
            [
                FakeRerankProvider(),
                FakeSynthesisProvider("Integrated answer with remote [S1] [S2]."),
            ]
        )
        monkeypatch.setattr("lsm.query.api.create_provider", lambda _cfg: next(provider_iter))

        result = asyncio.run(
            query(
                "What do local and remote sources say?",
                config,
                SessionState(),
                embedder=object(),
                collection=object(),
                progress_callback=lambda update: progress_updates.append(update),
            )
        )

        stages = [update.stage for update in progress_updates]
        assert "remote" in stages
        assert result.remote_sources
        assert any(source.get("provider") == "mock_remote" for source in result.remote_sources)
