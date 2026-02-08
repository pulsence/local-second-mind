"""
Shared query planning utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lsm.config.models import LSMConfig
from lsm.query.session import SessionState, Candidate
from lsm.query.retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from lsm.query.prefilter import prefilter_by_metadata
from lsm.query.rerank import apply_local_reranking
from lsm.vectordb.base import BaseVectorDBProvider
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
    metadata_filter: dict | None = None


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
            metadata_filter=None,
        )

    batch_size = config.batch_size
    query_vector = embed_text(embedder, question, batch_size=batch_size)

    path_contains = state.path_contains
    ext_allow = state.ext_allow
    ext_deny = state.ext_deny

    filters_active = bool(path_contains) or bool(ext_allow) or bool(ext_deny)
    retrieve_k = config.query.retrieve_k or (max(k, k * 3) if filters_active else k)

    where_filter = prefilter_by_metadata(question, available_metadata={})
    if config.ingest.enable_versioning:
        where_filter = {**where_filter, "is_current": True}
    if not where_filter:
        where_filter = None

    candidates = retrieve_candidates(collection, query_vector, retrieve_k, where_filter=where_filter)

    filtered = filter_candidates(
        candidates,
        path_contains=path_contains,
        ext_allow=ext_allow,
        ext_deny=ext_deny,
    )

    pinned_chunks = getattr(state, "pinned_chunks", None) or []
    if pinned_chunks:
        try:
            pinned_provider = collection if isinstance(collection, BaseVectorDBProvider) else None
            if pinned_provider is not None:
                pinned_result = pinned_provider.get(
                    ids=pinned_chunks,
                    include=["documents", "metadatas"],
                )
                if pinned_result.ids:
                    pinned_docs = pinned_result.documents or []
                    pinned_metas = pinned_result.metadatas or []
                    for i, chunk_id in enumerate(pinned_result.ids):
                        pinned_candidate = Candidate(
                            cid=chunk_id,
                            text=pinned_docs[i] if i < len(pinned_docs) else "",
                            meta=pinned_metas[i] if i < len(pinned_metas) else {},
                            distance=0.0,
                        )
                        if not any(c.cid == chunk_id for c in filtered):
                            filtered.insert(0, pinned_candidate)
        except Exception as exc:
            logger.error(f"Failed to load pinned chunks: {exc}")

    anchor_candidates: List[Candidate] = []
    if isinstance(collection, BaseVectorDBProvider):
        context_chunks = getattr(state, "context_chunks", None) or []
        if context_chunks:
            try:
                anchored = collection.get(
                    ids=context_chunks,
                    include=["documents", "metadatas"],
                )
                docs = anchored.documents or []
                metas = anchored.metadatas or []
                for i, chunk_id in enumerate(anchored.ids):
                    anchor_candidates.append(
                        Candidate(
                            cid=chunk_id,
                            text=docs[i] if i < len(docs) else "",
                            meta=metas[i] if i < len(metas) else {},
                            distance=0.0,
                        )
                    )
            except Exception as exc:
                logger.error(f"Failed to load context chunk anchors: {exc}")
        context_documents = getattr(state, "context_documents", None) or []
        if context_documents:
            for doc_path in context_documents:
                try:
                    anchored = collection.get(
                        filters={"source_path": doc_path},
                        limit=max(1, min(k, k_rerank)),
                        include=["documents", "metadatas"],
                    )
                    docs = anchored.documents or []
                    metas = anchored.metadatas or []
                    for i, chunk_id in enumerate(anchored.ids):
                        anchor_candidates.append(
                            Candidate(
                                cid=chunk_id,
                                text=docs[i] if i < len(docs) else "",
                                meta=metas[i] if i < len(metas) else {},
                                distance=0.0,
                            )
                        )
                except Exception as exc:
                    logger.error(f"Failed to load context document anchor '{doc_path}': {exc}")

    if anchor_candidates:
        existing_ids = {c.cid for c in filtered}
        prioritized = []
        for candidate in anchor_candidates:
            if candidate.cid not in existing_ids:
                prioritized.append(candidate)
                existing_ids.add(candidate.cid)
        filtered = prioritized + filtered

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
        metadata_filter=where_filter,
    )
