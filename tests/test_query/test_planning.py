from __future__ import annotations

from types import SimpleNamespace

import pytest

import lsm.query.planning as planning
from lsm.query.session import Candidate
from lsm.vectordb.base import VectorDBGetResult


def _config(
    *,
    local_enabled: bool = True,
    rerank_strategy: str = "hybrid",
    retrieve_k=None,
    no_rerank: bool = False,
):
    local = SimpleNamespace(enabled=local_enabled, k=4, k_rerank=2, min_relevance=0.25)
    mode = SimpleNamespace(source_policy=SimpleNamespace(local=local))
    query = SimpleNamespace(
        no_rerank=no_rerank,
        max_per_file=2,
        local_pool=8,
        rerank_strategy=rerank_strategy,
        retrieve_k=retrieve_k,
    )
    ingest = SimpleNamespace(enable_versioning=False)
    llm = SimpleNamespace(resolve_service=lambda name: None)
    return SimpleNamespace(get_mode_config=lambda: mode, query=query, batch_size=32, ingest=ingest, llm=llm)


def _state(*, path_contains=None, ext_allow=None, ext_deny=None, pinned_chunks=None):
    return SimpleNamespace(
        path_contains=path_contains,
        ext_allow=ext_allow,
        ext_deny=ext_deny,
        pinned_chunks=pinned_chunks or [],
        context_documents=[],
        context_chunks=[],
    )


def test_prepare_local_candidates_local_disabled() -> None:
    cfg = _config(local_enabled=False)
    state = _state()
    plan = planning.prepare_local_candidates("q", cfg, state, embedder=object(), collection=object())

    assert plan.local_enabled is False
    assert plan.candidates == []
    assert plan.filtered == []
    assert plan.retrieve_k == 0


def test_prepare_local_candidates_hybrid_rerank_with_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(local_enabled=True, rerank_strategy="hybrid", retrieve_k=None, no_rerank=False)
    state = _state(path_contains="docs")

    c1 = Candidate(cid="1", text="a", meta={"source_path": "/docs/a.md"}, distance=0.2)
    c2 = Candidate(cid="2", text="b", meta={"source_path": "/docs/b.md"}, distance=0.3)

    monkeypatch.setattr(planning, "embed_text", lambda embedder, question, batch_size: [0.1, 0.2])
    monkeypatch.setattr(planning, "retrieve_candidates", lambda collection, query_vector, retrieve_k, **kw: [c1, c2])
    monkeypatch.setattr(planning, "filter_candidates", lambda *args, **kwargs: [c1, c2])
    monkeypatch.setattr(planning, "apply_local_reranking", lambda *args, **kwargs: [c2, c1])
    monkeypatch.setattr(planning, "compute_relevance", lambda filtered: 0.77)

    plan = planning.prepare_local_candidates("question", cfg, state, embedder=object(), collection=object())

    assert plan.local_enabled is True
    assert plan.filters_active is True
    assert plan.retrieve_k == 12  # max(k, k*3) with filters and k=4
    assert [c.cid for c in plan.filtered] == ["2", "1"][: plan.k]
    assert plan.relevance == 0.77
    assert plan.should_llm_rerank is True


def test_prepare_local_candidates_rerank_none_uses_enforce_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(local_enabled=True, rerank_strategy="none", retrieve_k=5, no_rerank=False)
    state = _state()

    c1 = Candidate(cid="1", text="a", meta={}, distance=0.2)
    c2 = Candidate(cid="2", text="b", meta={}, distance=0.3)
    c3 = Candidate(cid="3", text="c", meta={}, distance=0.4)

    monkeypatch.setattr(planning, "embed_text", lambda *args, **kwargs: [0.1])
    monkeypatch.setattr(planning, "retrieve_candidates", lambda *a, **kw: [c1, c2, c3])
    monkeypatch.setattr(planning, "filter_candidates", lambda *a, **kw: [c1, c2, c3])
    monkeypatch.setattr(planning, "compute_relevance", lambda filtered: 0.5)
    monkeypatch.setattr("lsm.query.rerank.enforce_diversity", lambda candidates, max_per_file: [c3, c2, c1])

    plan = planning.prepare_local_candidates("q", cfg, state, embedder=object(), collection=object())

    assert plan.retrieve_k == 5
    assert [c.cid for c in plan.filtered] == ["3", "2", "1"][: plan.k]
    assert plan.should_llm_rerank is False


def test_prepare_local_candidates_pinned_chunks_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(local_enabled=True, rerank_strategy="lexical", no_rerank=False)
    state = _state(pinned_chunks=["p1"])

    base = Candidate(cid="1", text="a", meta={}, distance=0.2)
    monkeypatch.setattr(planning, "embed_text", lambda *args, **kwargs: [0.1])
    monkeypatch.setattr(planning, "retrieve_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "filter_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "apply_local_reranking", lambda question, candidates, **kwargs: candidates)
    monkeypatch.setattr(planning, "compute_relevance", lambda filtered: 0.2)

    # The collection must pass isinstance(collection, BaseVectorDBProvider)
    from unittest.mock import Mock
    from lsm.vectordb.base import BaseVectorDBProvider
    mock_collection = Mock(spec=BaseVectorDBProvider)
    mock_collection.get.return_value = VectorDBGetResult(
        ids=["p1"],
        documents=["pinned text"],
        metadatas=[{"source_path": "/docs/pinned.md"}],
    )

    plan = planning.prepare_local_candidates("q", cfg, state, embedder=object(), collection=mock_collection)
    assert plan.filtered[0].cid == "p1"


def test_prepare_local_candidates_prefilter_uses_metadata_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(local_enabled=True, rerank_strategy="none", retrieve_k=4, no_rerank=False)
    state = _state()

    base = Candidate(cid="1", text="a", meta={"source_path": "/docs/a.md"}, distance=0.2)
    monkeypatch.setattr(planning, "embed_text", lambda *args, **kwargs: [0.1])
    monkeypatch.setattr(planning, "retrieve_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "filter_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "compute_relevance", lambda filtered: 0.8)
    monkeypatch.setattr("lsm.query.rerank.enforce_diversity", lambda candidates, max_per_file: candidates)

    captured = {}

    def _fake_prefilter(question, available_metadata, llm_config=None):
        captured["metadata"] = available_metadata
        return {"content_type": "theology"}

    monkeypatch.setattr(planning, "prefilter_by_metadata", _fake_prefilter)

    from unittest.mock import Mock
    from lsm.vectordb.base import BaseVectorDBProvider

    mock_collection = Mock(spec=BaseVectorDBProvider)
    mock_collection.get.return_value = VectorDBGetResult(
        ids=["m1"],
        metadatas=[
            {
                "content_type": "theology",
                "ai_tags": '["christology"]',
                "user_tags": '["doctrine"]',
                "root_tags": '["theology"]',
                "folder_tags": '["research"]',
            }
        ],
    )

    plan = planning.prepare_local_candidates("q", cfg, state, embedder=object(), collection=mock_collection)

    assert plan.metadata_filter == {"content_type": "theology"}
    assert captured["metadata"]["content_type"] == ["theology"]
    assert captured["metadata"]["ai_tags"] == ["christology"]
    assert captured["metadata"]["user_tags"] == ["doctrine"]
    assert captured["metadata"]["root_tags"] == ["theology"]
    assert captured["metadata"]["folder_tags"] == ["research"]


def test_prepare_local_candidates_uses_decomposition_service_for_prefilter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(local_enabled=True, rerank_strategy="none", retrieve_k=4, no_rerank=False)
    state = _state()

    base = Candidate(cid="1", text="a", meta={"source_path": "/docs/a.md"}, distance=0.2)
    monkeypatch.setattr(planning, "embed_text", lambda *args, **kwargs: [0.1])
    monkeypatch.setattr(planning, "retrieve_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "filter_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "compute_relevance", lambda filtered: 0.8)
    monkeypatch.setattr("lsm.query.rerank.enforce_diversity", lambda candidates, max_per_file: candidates)

    sentinel_llm = object()
    captured = {"service_name": None, "llm_config": None}

    def _resolve_service(name):
        captured["service_name"] = name
        return sentinel_llm

    cfg.llm = SimpleNamespace(resolve_service=_resolve_service)

    def _fake_prefilter(question, available_metadata, llm_config=None):
        captured["llm_config"] = llm_config
        return {}

    monkeypatch.setattr(planning, "prefilter_by_metadata", _fake_prefilter)

    from unittest.mock import Mock
    from lsm.vectordb.base import BaseVectorDBProvider

    mock_collection = Mock(spec=BaseVectorDBProvider)
    mock_collection.get.return_value = VectorDBGetResult(ids=[], metadatas=[])

    planning.prepare_local_candidates("q", cfg, state, embedder=object(), collection=mock_collection)
    assert captured["service_name"] == "decomposition"
    assert captured["llm_config"] is sentinel_llm


def test_prepare_local_candidates_anchor_chunks_stay_prioritized_after_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(local_enabled=True, rerank_strategy="lexical", retrieve_k=4, no_rerank=False)
    state = _state()
    state.context_chunks = ["anchor-1"]

    base = Candidate(cid="1", text="a", meta={}, distance=0.2)
    monkeypatch.setattr(planning, "embed_text", lambda *args, **kwargs: [0.1])
    monkeypatch.setattr(planning, "retrieve_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "filter_candidates", lambda *args, **kwargs: [base])
    monkeypatch.setattr(planning, "compute_relevance", lambda filtered: 0.8)
    monkeypatch.setattr(planning, "apply_local_reranking", lambda question, candidates, **kwargs: [base])

    from unittest.mock import Mock
    from lsm.vectordb.base import BaseVectorDBProvider

    mock_collection = Mock(spec=BaseVectorDBProvider)
    mock_collection.get.side_effect = [
        VectorDBGetResult(ids=[], metadatas=[]),  # metadata inventory
        VectorDBGetResult(
            ids=["anchor-1"],
            documents=["anchored text"],
            metadatas=[{"source_path": "/docs/anchor.md"}],
        ),  # context chunk anchor fetch
    ]

    plan = planning.prepare_local_candidates("q", cfg, state, embedder=object(), collection=mock_collection)
    assert plan.filtered[0].cid == "anchor-1"
