import asyncio
from pathlib import Path

import pytest

from lsm.config.models import (
    FeatureLLMConfig,
    GlobalConfig,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LSMConfig,
    LocalSourcePolicy,
    ModeConfig,
    ModelKnowledgePolicy,
    QueryConfig,
    RemoteProviderConfig,
    RemoteSourcePolicy,
    SourcePolicyConfig,
    VectorDBConfig,
)
from lsm.query.api import query
from lsm.query.planning import LocalQueryPlan
from lsm.query.session import Candidate, SessionState
from lsm.remote.base import RemoteResult


def _build_query_config(tmp_path: Path, remote_enabled: bool) -> LSMConfig:
    mode_name = "integration_mode"
    return LSMConfig(
        ingest=IngestConfig(roots=[tmp_path], manifest=tmp_path / ".ingest" / "manifest.json"),
        query=QueryConfig(mode=mode_name, rerank_strategy="llm", no_rerank=False),
        llm=LLMRegistryConfig(
            llms=[
                LLMProviderConfig(
                    provider_name="openai",
                    api_key="test-key",
                    query=FeatureLLMConfig(model="gpt-5.2"),
                    ranking=FeatureLLMConfig(model="gpt-5.2"),
                )
            ]
        ),
        vectordb=VectorDBConfig(
            provider="chromadb",
            persist_dir=tmp_path / ".chroma",
            collection="test_collection",
        ),
        modes={
            mode_name: ModeConfig(
                synthesis_style="grounded",
                source_policy=SourcePolicyConfig(
                    local=LocalSourcePolicy(enabled=True, min_relevance=0.0, k=4, k_rerank=2),
                    remote=RemoteSourcePolicy(
                        enabled=remote_enabled,
                        remote_providers=["mock_remote"] if remote_enabled else None,
                        max_results=3,
                    ),
                    model_knowledge=ModelKnowledgePolicy(enabled=False),
                ),
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
    def test_progress_callback_receives_all_stages(self, tmp_path: Path, mocker) -> None:
        config = _build_query_config(tmp_path, remote_enabled=False)
        progress_updates = []
        local_plan = _build_local_plan()

        mocker.patch("lsm.query.context.prepare_local_candidates", return_value=local_plan)

        ranking_provider = mocker.Mock()
        ranking_provider.name = "openai"
        ranking_provider.model = "gpt-5.2"
        ranking_provider.rerank.return_value = [
            {"cid": "local-1", "text": local_plan.filtered[0].text, "metadata": local_plan.filtered[0].meta, "distance": 0.1}
        ]
        ranking_provider.estimate_cost.return_value = 0.0

        synthesis_provider = mocker.Mock()
        synthesis_provider.name = "openai"
        synthesis_provider.model = "gpt-5.2"
        synthesis_provider.synthesize.return_value = "Integrated answer [S1]."
        synthesis_provider.estimate_cost.return_value = 0.0

        mocker.patch(
            "lsm.query.api.create_provider",
            side_effect=[ranking_provider, synthesis_provider],
        )

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

    def test_progress_callback_with_remote_sources(self, tmp_path: Path, mocker) -> None:
        config = _build_query_config(tmp_path, remote_enabled=True)
        progress_updates = []
        local_plan = _build_local_plan()

        mocker.patch("lsm.query.context.prepare_local_candidates", return_value=local_plan)

        mock_remote_provider = mocker.Mock()
        mock_remote_provider.search.return_value = [
            RemoteResult(
                title="Remote Source",
                url="https://example.com/remote",
                snippet="Remote source snippet for integration progress test.",
                score=0.9,
            )
        ]
        mocker.patch("lsm.query.context.create_remote_provider", return_value=mock_remote_provider)

        ranking_provider = mocker.Mock()
        ranking_provider.name = "openai"
        ranking_provider.model = "gpt-5.2"
        ranking_provider.rerank.return_value = [
            {"cid": "local-1", "text": local_plan.filtered[0].text, "metadata": local_plan.filtered[0].meta, "distance": 0.1}
        ]
        ranking_provider.estimate_cost.return_value = 0.0

        synthesis_provider = mocker.Mock()
        synthesis_provider.name = "openai"
        synthesis_provider.model = "gpt-5.2"
        synthesis_provider.synthesize.return_value = "Integrated answer with remote [S1] [S2]."
        synthesis_provider.estimate_cost.return_value = 0.0

        mocker.patch(
            "lsm.query.api.create_provider",
            side_effect=[ranking_provider, synthesis_provider],
        )

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
