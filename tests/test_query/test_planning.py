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
    return SimpleNamespace(get_mode_config=lambda: mode, query=query, batch_size=32, ingest=ingest)


def _state(*, path_contains=None, ext_allow=None, ext_deny=None, pinned_chunks=None):
    return SimpleNamespace(
        path_contains=path_contains,
        ext_allow=ext_allow,
        ext_deny=ext_deny,
        pinned_chunks=pinned_chunks or [],
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
