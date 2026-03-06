"""Tests for RetrievalPipeline three-stage API."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lsm.query.pipeline import RetrievalPipeline
from lsm.query.pipeline_types import (
    ContextPackage,
    CostEntry,
    FilterSet,
    QueryRequest,
    QueryResponse,
    RemoteSource,
)
from lsm.query.session import Candidate


def _make_config(
    mode="grounded",
    chat_mode="single",
    enable_llm_server_cache=True,
    k=12,
    min_relevance=0.25,
):
    """Build a minimal config-like namespace for pipeline tests."""
    query = SimpleNamespace(
        k=k,
        retrieve_k=None,
        retrieval_profile="hybrid_rrf",
        k_dense=100,
        k_sparse=100,
        rrf_dense_weight=0.7,
        rrf_sparse_weight=0.3,
        min_relevance=min_relevance,
        mode=mode,
        path_contains=None,
        ext_allow=None,
        ext_deny=None,
        chat_mode=chat_mode,
        enable_llm_server_cache=enable_llm_server_cache,
        cluster_enabled=False,
        cluster_algorithm="kmeans",
        cluster_k=50,
        cluster_top_n=5,
        graph_expansion_enabled=False,
        graph_expansion_hops=2,
    )

    from lsm.config.models.modes import GROUNDED_MODE, INSIGHT_MODE, HYBRID_MODE

    modes = {
        "grounded": GROUNDED_MODE,
        "insight": INSIGHT_MODE,
        "hybrid": HYBRID_MODE,
    }

    query_service = SimpleNamespace(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        api_key="test-key",
    )
    ranking_service = SimpleNamespace(
        provider="openai",
        model="gpt-4",
        temperature=0.2,
        max_tokens=800,
        api_key="test-key",
    )

    llm = SimpleNamespace(
        resolve_service=lambda name: ranking_service if name == "ranking" else query_service,
    )

    config = SimpleNamespace(
        query=query,
        llm=llm,
        global_folder="/tmp/lsm",
        global_settings=SimpleNamespace(
            global_folder="/tmp/lsm",
            embed_model="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=32,
        ),
        batch_size=32,
        get_mode_config=lambda name=None: modes.get(name or mode, GROUNDED_MODE),
        db=SimpleNamespace(path=".lsm", collection="test"),
        remote_providers=None,
        chats=SimpleNamespace(enabled=False, auto_save=False, dir="chats"),
        get_active_remote_providers=lambda **kw: [],
    )
    return config


def _make_candidates(n=3):
    return [
        Candidate(
            cid=f"chunk-{i}",
            text=f"Content of chunk {i} about topic.",
            meta={
                "source_path": f"/docs/file{i}.md",
                "source_name": f"file{i}.md",
                "chunk_index": i,
                "ext": ".md",
            },
            distance=0.1 + i * 0.05,
        )
        for i in range(n)
    ]


def _make_plan(candidates=None, filtered=None, k=12):
    from lsm.query.planning import LocalQueryPlan

    if candidates is None:
        candidates = _make_candidates(5)
    if filtered is None:
        filtered = candidates[:3]
    return LocalQueryPlan(
        local_enabled=True,
        candidates=candidates,
        filtered=filtered,
        relevance=0.85,
        filters_active=False,
        retrieve_k=k,
        k=k,
        min_relevance=0.25,
    )


class TestBuildSources:
    @patch("lsm.query.pipeline.prepare_local_candidates")
    def test_returns_context_package(self, mock_prepare):
        plan = _make_plan()
        mock_prepare.return_value = plan
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        request = QueryRequest(question="What is X?")
        package = pipeline.build_sources(request)

        assert isinstance(package, ContextPackage)
        assert len(package.candidates) == 3
        assert package.local_enabled is True
        assert "dense_recall" in package.retrieval_trace.stages_executed
        assert package.retrieval_trace.dense_candidates_count == 5

    @patch("lsm.query.pipeline.prepare_local_candidates")
    def test_filters_propagate(self, mock_prepare):
        plan = _make_plan()
        mock_prepare.return_value = plan
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        request = QueryRequest(
            question="test",
            filters=FilterSet(ext_allow=[".md"]),
        )
        package = pipeline.build_sources(request)

        # Verify the state was built with filters
        call_args = mock_prepare.call_args
        state = call_args[0][2]  # third positional arg = state
        assert state.ext_allow == [".md"]


class TestSynthesizeContext:
    def test_assigns_labels(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        pipeline = RetrievalPipeline(db, embedder, config, provider)

        candidates = _make_candidates(2)
        package = ContextPackage(
            request=QueryRequest(question="test"),
            candidates=candidates,
        )
        result = pipeline.synthesize_context(package)

        assert result.context_block is not None
        assert "[S1]" in result.context_block
        assert "[S2]" in result.context_block
        assert "S1" in result.source_labels
        assert "S2" in result.source_labels

    def test_starting_prompt_explicit_priority(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        pipeline = RetrievalPipeline(db, embedder, config, provider)

        package = ContextPackage(
            request=QueryRequest(
                question="test",
                starting_prompt="Custom prompt override",
            ),
            candidates=_make_candidates(1),
        )
        result = pipeline.synthesize_context(package)
        assert result.starting_prompt == "Custom prompt override"

    def test_starting_prompt_defaults_to_mode(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        pipeline = RetrievalPipeline(db, embedder, config, provider)

        package = ContextPackage(
            request=QueryRequest(question="test"),
            candidates=_make_candidates(1),
        )
        result = pipeline.synthesize_context(package)
        # Should be the grounded instructions
        assert "ONLY the provided sources" in result.starting_prompt


class TestExecute:
    def test_calls_provider(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "gpt-4"
        provider.send_message.return_value = "The answer is [S1] from the source."
        provider.estimate_cost.return_value = 0.01
        provider.last_response_id = "resp-123"

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        candidates = _make_candidates(1)
        package = ContextPackage(
            request=QueryRequest(question="What?"),
            candidates=candidates,
            context_block="[S1] /docs/file0.md\nContent",
            source_labels={"S1": {"source_path": "/docs/file0.md"}},
            starting_prompt="Answer with citations.",
        )
        response = pipeline.execute(package)

        assert isinstance(response, QueryResponse)
        assert "answer" in response.answer.lower() or "[S1]" in response.answer
        provider.send_message.assert_called_once()

    def test_forwards_prior_response_id(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "gpt-4"
        provider.send_message.return_value = "Answer [S1]"
        provider.estimate_cost.return_value = 0.0
        provider.last_response_id = None

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        package = ContextPackage(
            request=QueryRequest(question="test"),
            candidates=_make_candidates(1),
            context_block="text",
            starting_prompt="prompt",
            prior_response_id="prev-resp-id",
        )
        pipeline.execute(package)

        call_kwargs = provider.send_message.call_args
        assert call_kwargs.kwargs.get("previous_response_id") == "prev-resp-id" or \
               (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("previous_response_id") == "prev-resp-id"

    def test_returns_response_id(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "gpt-4"
        provider.send_message.return_value = "Answer [S1]"
        provider.estimate_cost.return_value = 0.0
        provider.last_response_id = "new-resp-456"

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        package = ContextPackage(
            request=QueryRequest(question="test"),
            candidates=_make_candidates(1),
            context_block="text",
            starting_prompt="prompt",
        )
        response = pipeline.execute(package)
        assert response.response_id == "new-resp-456"

    def test_synthesis_failure_returns_provider_specific_fallback(self):
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "anthropic"
        provider.model = "claude-sonnet-4-6"
        provider.send_message.side_effect = RuntimeError("boom")
        provider.estimate_cost.return_value = 0.0
        provider.last_response_id = None

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        package = ContextPackage(
            request=QueryRequest(question="What?"),
            candidates=_make_candidates(1),
            context_block="[S1] /docs/file0.md\nContent",
            source_labels={"S1": {"source_path": "/docs/file0.md"}},
            starting_prompt="Answer with citations.",
        )
        response = pipeline.execute(package)

        assert "configured query model (claude) is unavailable" in response.answer.lower()
        assert "[S1]" in response.answer


class TestRun:
    @patch("lsm.query.pipeline.prepare_local_candidates")
    def test_chains_all_stages(self, mock_prepare):
        plan = _make_plan()
        mock_prepare.return_value = plan
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "gpt-4"
        provider.send_message.return_value = "The answer is [S1] great."
        provider.estimate_cost.return_value = 0.01
        provider.last_response_id = None

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        request = QueryRequest(question="What is X?")
        response = pipeline.run(request)

        assert isinstance(response, QueryResponse)
        assert response.answer
        assert response.package.context_block is not None
        mock_prepare.assert_called_once()
        provider.send_message.assert_called_once()

    @patch("lsm.query.pipeline.prepare_local_candidates")
    def test_conversation_id_chaining(self, mock_prepare):
        plan = _make_plan()
        mock_prepare.return_value = plan
        config = _make_config(chat_mode="chat")
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "gpt-4"
        provider.send_message.return_value = "Answer [S1]."
        provider.estimate_cost.return_value = 0.0
        provider.last_response_id = "resp-turn-1"

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        request = QueryRequest(
            question="First question",
            conversation_id="conv-1",
            chat_mode="chat",
        )
        response = pipeline.run(request)
        assert response.conversation_id == "conv-1"
        assert response.response_id == "resp-turn-1"

    @patch("lsm.query.pipeline.prepare_local_candidates")
    def test_starting_prompt_resolution_priority(self, mock_prepare):
        """Explicit starting_prompt > mode default."""
        plan = _make_plan()
        mock_prepare.return_value = plan
        config = _make_config()
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "openai"
        provider.model = "gpt-4"
        provider.send_message.return_value = "Answer [S1]"
        provider.estimate_cost.return_value = 0.0
        provider.last_response_id = None

        pipeline = RetrievalPipeline(db, embedder, config, provider)

        # With explicit starting_prompt
        request = QueryRequest(
            question="test",
            starting_prompt="Use only bullet points.",
        )
        response = pipeline.run(request)

        # The instruction sent to provider should be the explicit one
        call_kwargs = provider.send_message.call_args
        assert "bullet points" in str(call_kwargs)

    @patch("lsm.query.pipeline.prepare_local_candidates")
    def test_low_relevance_uses_relevance_fallback_message(self, mock_prepare):
        plan = _make_plan()
        plan.relevance = 0.05
        plan.min_relevance = 0.25
        mock_prepare.return_value = plan
        config = _make_config(min_relevance=0.25)
        db = MagicMock()
        embedder = MagicMock()
        provider = MagicMock()
        provider.name = "anthropic"
        provider.model = "claude-sonnet-4-6"

        pipeline = RetrievalPipeline(db, embedder, config, provider)
        request = QueryRequest(question="What is X?")
        response = pipeline.run(request)

        assert "relevance threshold" in response.answer.lower()
        assert "openai is unavailable" not in response.answer.lower()
        provider.send_message.assert_not_called()
