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
    LocalSourcePolicy,
    ModeConfig,
    ModelKnowledgePolicy,
    QueryConfig,
    RemoteProviderConfig,
    RemoteSourcePolicy,
    DBConfig,
)
from lsm.query.api import query
from lsm.query.pipeline_types import (
    ContextPackage,
    CostEntry,
    QueryRequest,
    QueryResponse,
    RemoteSource,
    RetrievalTrace,
    StageTimings,
)
from lsm.query.session import Candidate, SessionState


class FakeLLMProvider:
    name = "openai"
    model = "gpt-5.2"

    def send_message(self, input, instruction=None, **kwargs):
        return "fake"

    def estimate_cost(self, _input_tokens, _output_tokens):
        return 0.0


def _build_query_config(tmp_path: Path, remote_enabled: bool) -> LSMConfig:
    mode_name = "integration_mode"
    return LSMConfig(
        ingest=IngestConfig(roots=[tmp_path]),
        query=QueryConfig(mode=mode_name),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test-key")],
            services={
                "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
                "ranking": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            },
        ),
        db=DBConfig(
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


def _make_candidate():
    return Candidate(
        cid="local-1",
        text="Local source content for integration testing.",
        meta={"source_path": "/docs/local.md", "source_name": "local.md", "chunk_index": 0},
        distance=0.1,
    )


def _mock_pipeline_run_with_stages(candidate, stages, answer="Integrated answer [S1]."):
    """Build a pipeline.run callable that simulates specific stages and timing."""
    def _run(request, progress_callback=None):
        # Emit progress callbacks to simulate pipeline stages
        if progress_callback:
            for i, stage in enumerate(stages):
                progress_callback(stage, 0, 1, f"{stage} starting...")
                progress_callback(stage, 1, 1, f"{stage} complete")

        pkg = ContextPackage(
            request=request,
            candidates=[candidate],
            remote_sources=[],
            retrieval_trace=RetrievalTrace(
                stages_executed=stages,
                timings=[
                    StageTimings(stage=s, duration_ms=10.0) for s in stages
                ],
            ),
            all_candidates=[candidate],
            filtered_candidates=[candidate],
            relevance=0.95,
            local_enabled=True,
            context_block="[S1] Local content",
            source_labels={"S1": {}},
            starting_prompt="Answer.",
        )
        return QueryResponse(
            answer=answer,
            package=pkg,
            costs=[CostEntry(provider="openai", model="gpt-5.2", cost=0.0)],
        )
    return _run


def _mock_pipeline_run_with_remote(candidate, remote_sources):
    """Build a pipeline.run callable that includes remote sources."""
    def _run(request, progress_callback=None):
        stages = ["retrieval", "rerank", "remote", "synthesis"]
        if progress_callback:
            for stage in stages:
                progress_callback(stage, 0, 1, f"{stage}...")
                progress_callback(stage, 1, 1, f"{stage} done")

        pkg = ContextPackage(
            request=request,
            candidates=[candidate],
            remote_sources=remote_sources,
            retrieval_trace=RetrievalTrace(
                stages_executed=stages,
                timings=[StageTimings(stage=s, duration_ms=5.0) for s in stages],
            ),
            all_candidates=[candidate],
            filtered_candidates=[candidate],
            relevance=0.95,
            local_enabled=True,
            context_block="[S1] [S2] content",
            source_labels={"S1": {}, "S2": {}},
            starting_prompt="Answer.",
        )
        return QueryResponse(
            answer="Integrated answer with remote [S1] [S2].",
            package=pkg,
            costs=[CostEntry(provider="openai", model="gpt-5.2", cost=0.0)],
        )
    return _run


def _make_fake_pipeline_class(run_fn):
    """Create a fake pipeline class bound to a specific run function."""
    class _Pipeline:
        def __init__(self, **_kwargs):
            pass
        def run(self, request, progress_callback=None):
            return run_fn(request, progress_callback)
    return _Pipeline


@pytest.mark.integration
class TestQueryProgressCallbacks:
    def test_progress_callback_receives_all_stages(self, monkeypatch, tmp_path: Path) -> None:
        from lsm.query import api as qapi

        config = _build_query_config(tmp_path, remote_enabled=False)
        progress_updates = []
        candidate = _make_candidate()

        run_fn = _mock_pipeline_run_with_stages(
            candidate,
            stages=["retrieval", "rerank", "synthesis"],
            answer="Integrated answer [S1].",
        )
        monkeypatch.setattr(qapi, "RetrievalPipeline", _make_fake_pipeline_class(run_fn))
        monkeypatch.setattr(qapi, "create_provider", lambda cfg: FakeLLMProvider())

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

    def test_progress_callback_with_remote_sources(self, monkeypatch, tmp_path: Path) -> None:
        from lsm.query import api as qapi

        config = _build_query_config(tmp_path, remote_enabled=True)
        progress_updates = []
        candidate = _make_candidate()

        remote_sources = [
            RemoteSource(
                provider="mock_remote",
                title="Remote Source",
                url="https://example.com/remote",
                snippet="Remote source snippet for integration progress test.",
                score=0.9,
            )
        ]

        run_fn = _mock_pipeline_run_with_remote(candidate, remote_sources)
        monkeypatch.setattr(qapi, "RetrievalPipeline", _make_fake_pipeline_class(run_fn))
        monkeypatch.setattr(qapi, "create_provider", lambda cfg: FakeLLMProvider())

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
