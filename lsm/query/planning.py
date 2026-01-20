"""
Shared query planning utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lsm.config.models import LSMConfig
from lsm.query.session import SessionState, Candidate
from lsm.query.retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from lsm.query.rerank import apply_local_reranking
from lsm.vectordb.utils import require_chroma_collection
from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LocalQueryPlan:
    local_enabled: bool
    candidates: List[Candidate]
    filtered: List[Candidate]
    relevance: float
    filters_active: bool
    retrieve_k: int
    rerank_strategy: str
    should_llm_rerank: bool
    k: int
    k_rerank: int
    min_relevance: float
    max_per_file: int
    local_pool: int
    no_rerank: bool


def prepare_local_candidates(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> LocalQueryPlan:
    """
    Prepare local candidates for query execution or cost estimation.
    """
    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local

    k = local_policy.k
    k_rerank = local_policy.k_rerank
    min_relevance = local_policy.min_relevance
    no_rerank = config.query.no_rerank
    max_per_file = config.query.max_per_file
    local_pool = config.query.local_pool or max(k * 3, k_rerank * 4)
    rerank_strategy = config.query.rerank_strategy.lower()
    local_enabled = local_policy.enabled

    if not local_enabled:
        return LocalQueryPlan(
            local_enabled=False,
            candidates=[],
            filtered=[],
            relevance=0.0,
            filters_active=False,
            retrieve_k=0,
            rerank_strategy=rerank_strategy,
            should_llm_rerank=False,
            k=k,
            k_rerank=k_rerank,
            min_relevance=min_relevance,
            max_per_file=max_per_file,
            local_pool=local_pool,
            no_rerank=no_rerank,
        )

    batch_size = config.batch_size
    query_vector = embed_text(embedder, question, batch_size=batch_size)

    path_contains = state.path_contains
    ext_allow = state.ext_allow
    ext_deny = state.ext_deny

    filters_active = bool(path_contains) or bool(ext_allow) or bool(ext_deny)
    retrieve_k = config.query.retrieve_k or (max(k, k * 3) if filters_active else k)

    candidates = retrieve_candidates(collection, query_vector, retrieve_k)

    filtered = filter_candidates(
        candidates,
        path_contains=path_contains,
        ext_allow=ext_allow,
        ext_deny=ext_deny,
    )

    if state.pinned_chunks:
        try:
            chroma = require_chroma_collection(collection, "pinned chunk retrieval")
            pinned_results = chroma.get(
                ids=state.pinned_chunks,
                include=["documents", "metadatas", "distances"],
            )
            if pinned_results and pinned_results.get("ids"):
                for i, chunk_id in enumerate(pinned_results["ids"]):
                    pinned_candidate = Candidate(
                        cid=chunk_id,
                        text=pinned_results["documents"][i],
                        meta=pinned_results["metadatas"][i],
                        distance=0.0,
                    )
                    if not any(c.cid == chunk_id for c in filtered):
                        filtered.insert(0, pinned_candidate)
        except Exception as exc:
            logger.error(f"Failed to load pinned chunks: {exc}")

    if filtered:
        if rerank_strategy in ("lexical", "hybrid"):
            filtered = apply_local_reranking(
                question,
                filtered,
                max_per_file=max_per_file,
                local_pool=local_pool,
            )
            filtered = filtered[: min(k, len(filtered))]
        elif rerank_strategy == "none":
            from lsm.query.rerank import enforce_diversity
            filtered = enforce_diversity(filtered, max_per_file=max_per_file)
            filtered = filtered[: min(k, len(filtered))]

    relevance = compute_relevance(filtered) if filtered else 0.0
    should_llm_rerank = rerank_strategy in ("llm", "hybrid") and not no_rerank

    return LocalQueryPlan(
        local_enabled=True,
        candidates=candidates,
        filtered=filtered,
        relevance=relevance,
        filters_active=filters_active,
        retrieve_k=retrieve_k,
        rerank_strategy=rerank_strategy,
        should_llm_rerank=should_llm_rerank,
        k=k,
        k_rerank=k_rerank,
        min_relevance=min_relevance,
        max_per_file=max_per_file,
        local_pool=local_pool,
        no_rerank=no_rerank,
    )
