"""
RetrievalPipeline — three-stage query execution.

Stages:
    build_sources(request)       → ContextPackage
    synthesize_context(package)  → ContextPackage  (labels, prompt, context_block)
    execute(package)             → QueryResponse   (LLM synthesis)

run(request) chains all three stages.
"""

from __future__ import annotations

import asyncio
import re
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from lsm.config.models import LSMConfig
from lsm.config.models.modes import GROUNDED_MODE, ModeConfig
from lsm.logging import get_logger
from lsm.providers import create_provider
from lsm.providers.base import BaseLLMProvider
from lsm.query.context import (
    build_context_block,
    build_remote_candidates,
    fallback_answer,
    fetch_remote_sources,
    format_user_content,
)
from lsm.query.cost_tracking import (
    estimate_output_tokens,
    estimate_rerank_cost,
    estimate_tokens,
)
from lsm.query.pipeline_types import (
    Citation,
    ContextPackage,
    CostEntry,
    QueryRequest,
    QueryResponse,
    RemoteSource,
    RetrievalTrace,
    ScoreBreakdown,
    StageTimings,
)
from lsm.query.planning import LocalQueryPlan, prepare_local_candidates
from lsm.query.rerank import llm_rerank_candidates
from lsm.query.session import Candidate, SessionState
from lsm.query.stages.dense_recall import dense_recall
from lsm.query.stages.sparse_recall import sparse_recall
from lsm.query.stages.rrf_fusion import rrf_fuse
from lsm.query.stages.dedup import deduplicate_candidates
from lsm.query.stages.diversity import mmr_select, per_section_cap
from lsm.query.stages.temporal import apply_temporal_boost
from lsm.query.stages.cross_encoder import CrossEncoderReranker
from lsm.query.stages.hyde import hyde_recall
from lsm.query.stages.multi_vector import multi_vector_recall
from lsm.query.stages.graph_expansion import expand_via_graph
from lsm.vectordb.base import BaseVectorDBProvider

VALID_PROFILES = (
    "dense_only",
    "hybrid_rrf",
    "hyde_hybrid",
    "dense_cross_rerank",
    "llm_rerank",
    "multi_vector",
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class RetrievalPipeline:
    """Three-stage retrieval pipeline.

    Parameters
    ----------
    db : BaseVectorDBProvider
        Vector database provider for similarity search.
    embedder
        SentenceTransformer model (or compatible) for query embedding.
    config : LSMConfig
        Application configuration.
    llm_provider : BaseLLMProvider
        LLM provider for synthesis.
    """

    def __init__(
        self,
        db: BaseVectorDBProvider,
        embedder: Any,
        config: LSMConfig,
        llm_provider: BaseLLMProvider,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.config = config
        self.llm_provider = llm_provider

    # ------------------------------------------------------------------
    # Stage 1: build_sources
    # ------------------------------------------------------------------
    def build_sources(
        self,
        request: QueryRequest,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> ContextPackage:
        """Retrieve candidates from local and remote sources.

        Routes through the configured retrieval profile:
        - dense_only: vector similarity only
        - hybrid_rrf: dense + sparse (BM25) + RRF fusion
        - llm_rerank: dense + LLM reranking
        - hyde_hybrid, dense_cross_rerank: placeholders (Phase 11)

        Returns a ``ContextPackage`` with ``candidates``, ``remote_sources``,
        and ``retrieval_trace`` populated.
        """
        t0 = time.monotonic()
        trace = RetrievalTrace()
        costs: List[CostEntry] = []

        mode_config = self._resolve_mode(request)
        local_policy = mode_config.local_policy
        remote_policy = mode_config.remote_policy
        profile = mode_config.retrieval_profile or self.config.query.retrieval_profile

        # Build a lightweight SessionState for the planning module
        state = self._request_to_state(request)

        # --- Cluster pre-filtering ---
        cluster_ids: Optional[List[int]] = None
        if local_policy.enabled and self.config.query.cluster_enabled:
            try:
                conn = getattr(self.db, "connection", None)
                if conn is not None:
                    from lsm.db.clustering import get_top_clusters
                    from lsm.query.planning import embed_text

                    q_vec = embed_text(
                        self.embedder, request.question,
                        batch_size=self.config.batch_size,
                    )
                    cluster_ids = get_top_clusters(
                        q_vec, conn, top_n=self.config.query.cluster_top_n,
                    )
                    if cluster_ids:
                        trace.stages_executed.append("cluster_prefilter")
            except Exception as exc:
                logger.warning("Cluster pre-filtering failed: %s", exc)

        # --- Local retrieval via profile routing ---
        plan: Optional[LocalQueryPlan] = None
        local_candidates: List[Candidate] = []
        if local_policy.enabled:
            stage_t0 = time.monotonic()
            if progress_callback:
                progress_callback("retrieval", 0, 1, "Searching local knowledge base...")

            extra_filters = {"cluster_id": cluster_ids} if cluster_ids else None

            if profile == "multi_vector":
                local_candidates, plan = self._profile_multi_vector(
                    request, state, trace, extra_filters=extra_filters,
                )
            elif profile == "hybrid_rrf":
                local_candidates, plan = self._profile_hybrid_rrf(
                    request, state, trace, costs, extra_filters=extra_filters,
                )
            elif profile == "llm_rerank":
                local_candidates, plan = self._profile_llm_rerank(
                    request, state, trace, costs, extra_filters=extra_filters,
                )
            elif profile == "dense_cross_rerank":
                local_candidates, plan = self._profile_dense_cross_rerank(
                    request, state, trace, extra_filters=extra_filters,
                )
            elif profile == "hyde_hybrid":
                local_candidates, plan = self._profile_hyde_hybrid(
                    request, state, trace, costs, extra_filters=extra_filters,
                )
            else:
                # dense_only (default)
                local_candidates, plan = self._profile_dense_only(
                    request, state, trace, extra_filters=extra_filters,
                )

            trace.retrieval_profile = profile
            trace.timings.append(
                StageTimings(
                    stage="local_retrieval",
                    duration_ms=(time.monotonic() - stage_t0) * 1000,
                )
            )
            # --- Graph expansion (post-retrieval) ---
            if self.config.query.graph_expansion_enabled and local_candidates:
                try:
                    graph_t0 = time.monotonic()
                    local_candidates = expand_via_graph(
                        local_candidates,
                        self.db,
                        max_hops=self.config.query.graph_expansion_hops,
                    )
                    trace.stages_executed.append("graph_expansion")
                    trace.timings.append(
                        StageTimings(
                            stage="graph_expansion",
                            duration_ms=(time.monotonic() - graph_t0) * 1000,
                        )
                    )
                except Exception as exc:
                    logger.warning("Graph expansion failed: %s", exc)

            # --- Post-retrieval ranking stages ---
            if local_candidates:
                local_candidates = self._apply_post_retrieval_stages(
                    request=request,
                    candidates=local_candidates,
                    trace=trace,
                    final_k=plan.k if plan is not None else local_policy.k,
                )

            if progress_callback:
                progress_callback("retrieval", 1, 1, "Local retrieval complete")

        # --- Remote retrieval ---
        remote_sources: List[RemoteSource] = []
        remote_candidates: List[Candidate] = []
        if remote_policy.enabled:
            stage_t0 = time.monotonic()
            raw_remote = fetch_remote_sources(
                request.question,
                self.config,
                mode_config,
                progress_callback=progress_callback,
            )
            for r in raw_remote:
                remote_sources.append(
                    RemoteSource(
                        provider=r.get("provider", "remote"),
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("snippet", ""),
                        score=r.get("score", 0.5),
                        weight=r.get("weight", 1.0),
                        metadata=r.get("metadata", {}),
                    )
                )
            remote_candidates = build_remote_candidates(raw_remote)
            trace.stages_executed.append("remote_retrieval")
            trace.timings.append(
                StageTimings(
                    stage="remote_retrieval",
                    duration_ms=(time.monotonic() - stage_t0) * 1000,
                )
            )

        combined = local_candidates + remote_candidates

        return ContextPackage(
            request=request,
            candidates=combined,
            remote_sources=remote_sources,
            retrieval_trace=trace,
            costs=costs,
            all_candidates=plan.candidates if plan else [],
            filtered_candidates=plan.filtered if plan else [],
            relevance=plan.relevance if plan else 0.0,
            local_enabled=plan.local_enabled if plan else False,
            prior_response_id=request.prior_response_id,
        )

    # --- Profile implementations ---

    def _profile_dense_only(
        self,
        request: QueryRequest,
        state: SessionState,
        trace: RetrievalTrace,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Dense-only retrieval (default profile)."""
        plan = prepare_local_candidates(
            request.question, self.config, state, self.embedder, self.db,
            extra_filters=extra_filters,
        )
        dense_candidates = plan.filtered[: max(1, int(getattr(self.config.query, "k_dense", plan.k) or plan.k))]
        self._populate_dense_score_breakdown(dense_candidates)
        trace.stages_executed.append("dense_recall")
        trace.dense_candidates_count = len(plan.candidates)
        local_candidates = dense_candidates
        return local_candidates, plan

    def _profile_multi_vector(
        self,
        request: QueryRequest,
        state: SessionState,
        trace: RetrievalTrace,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Multi-vector retrieval across chunk, section, and file granularities."""
        from lsm.query.retrieval import embed_text

        query_embedding = embed_text(
            self.embedder, request.question,
            batch_size=getattr(self.config, "batch_size", 32),
        )

        k = max(
            self.config.query.k,
            int(getattr(self.config.query, "k_dense", self.config.query.k) or self.config.query.k),
        )
        filters = extra_filters or {}

        candidates = multi_vector_recall(
            query_embedding=query_embedding,
            db=self.db,
            top_k=k,
            filters=filters if filters else None,
        )

        # Build a lightweight plan for the ContextPackage
        plan = LocalQueryPlan(
            local_enabled=True,
            candidates=candidates,
            filtered=candidates[:k],
            relevance=1.0 if candidates else 0.0,
            filters_active=bool(extra_filters),
            retrieve_k=k,
            k=k,
            min_relevance=self.config.query.min_relevance,
        )

        trace.stages_executed.append("multi_vector_recall")
        trace.dense_candidates_count = len(candidates)

        local_candidates = candidates
        return local_candidates, plan

    def _profile_hybrid_rrf(
        self,
        request: QueryRequest,
        state: SessionState,
        trace: RetrievalTrace,
        costs: List[CostEntry],
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Hybrid RRF: dense + sparse (BM25) + RRF fusion."""
        if not self._is_sparse_recall_available():
            logger.warning(
                "FTS5 not available; falling back to dense_only profile"
            )
            return self._profile_dense_only(request, state, trace, extra_filters=extra_filters)

        # Dense recall
        plan = prepare_local_candidates(
            request.question, self.config, state, self.embedder, self.db,
            extra_filters=extra_filters,
        )
        dense_candidates = plan.filtered[
            : max(1, int(getattr(self.config.query, "k_dense", plan.k) or plan.k))
        ]
        self._populate_dense_score_breakdown(dense_candidates)
        trace.stages_executed.append("dense_recall")
        trace.dense_candidates_count = len(plan.candidates)

        # Sparse recall
        k_sparse = self.config.query.k_sparse
        sparse_candidates = sparse_recall(request.question, self.db, k_sparse)
        trace.stages_executed.append("sparse_recall")
        trace.sparse_candidates_count = len(sparse_candidates)

        # RRF fusion
        fused = rrf_fuse(
            dense_candidates,
            sparse_candidates,
            dense_weight=self.config.query.rrf_dense_weight,
            sparse_weight=self.config.query.rrf_sparse_weight,
        )
        trace.stages_executed.append("rrf_fusion")
        trace.fused_candidates_count = len(fused)

        local_candidates = fused
        return local_candidates, plan

    def _profile_llm_rerank(
        self,
        request: QueryRequest,
        state: SessionState,
        trace: RetrievalTrace,
        costs: List[CostEntry],
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Dense + LLM reranking profile."""
        plan = prepare_local_candidates(
            request.question, self.config, state, self.embedder, self.db,
            extra_filters=extra_filters,
        )
        trace.stages_executed.append("dense_recall")
        trace.dense_candidates_count = len(plan.candidates)

        # LLM reranking
        ranking_config = self.config.llm.resolve_service("ranking")
        ranking_provider = create_provider(ranking_config)
        rerank_dicts = [
            {"text": c.text, "metadata": c.meta, "distance": c.distance}
            for c in plan.filtered
        ]
        k_rerank = min(
            max(plan.k, int(getattr(self.config.query, "k_dense", plan.k) or plan.k)),
            len(plan.filtered),
        )
        reranked = llm_rerank_candidates(
            request.question, rerank_dicts, k=k_rerank, provider=ranking_provider
        )
        local_candidates = [
            Candidate(
                cid=item.get("cid", ""),
                text=item.get("text", ""),
                meta=item.get("metadata", {}),
                distance=item.get("distance"),
                score_breakdown=ScoreBreakdown(rerank_score=1.0 / (rank + 1)),
            )
            for rank, item in enumerate(reranked)
        ]

        # Cost tracking
        rerank_est = estimate_rerank_cost(
            ranking_provider, request.question, plan.filtered, k=k_rerank
        )
        costs.append(
            CostEntry(
                provider=ranking_provider.name,
                model=ranking_provider.model,
                input_tokens=rerank_est["input_tokens"],
                output_tokens=rerank_est["output_tokens"],
                cost=rerank_est["cost"],
                kind="rerank",
            )
        )

        trace.stages_executed.append("llm_rerank")
        trace.reranked_candidates_count = len(local_candidates)
        return local_candidates, plan

    def _profile_dense_cross_rerank(
        self,
        request: QueryRequest,
        state: SessionState,
        trace: RetrievalTrace,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Dense + cross-encoder reranking profile."""
        plan = prepare_local_candidates(
            request.question, self.config, state, self.embedder, self.db,
            extra_filters=extra_filters,
        )
        trace.stages_executed.append("dense_recall")
        trace.dense_candidates_count = len(plan.candidates)

        # Cross-encoder reranking
        model_name = getattr(
            self.config.query,
            "cross_encoder_model",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        device = getattr(self.config, "device", "cpu")
        cache_conn = getattr(self.db, "connection", getattr(self.db, "_conn", None))
        reranker = CrossEncoderReranker(
            model_name=model_name,
            device=device,
            cache_conn=cache_conn,
        )
        k = min(
            max(plan.k, int(getattr(self.config.query, "k_dense", plan.k) or plan.k)),
            len(plan.filtered),
        )
        local_candidates = reranker.rerank(request.question, plan.filtered, top_k=k)

        trace.stages_executed.append("cross_encoder_rerank")
        trace.reranked_candidates_count = len(local_candidates)
        return local_candidates, plan

    def _profile_hyde_hybrid(
        self,
        request: QueryRequest,
        state: SessionState,
        trace: RetrievalTrace,
        costs: List[CostEntry],
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """HyDE + hybrid RRF: generate hypothetical docs, embed, then hybrid_rrf."""
        query_config = self.config.query
        try:
            hyde_llm_config = self.config.llm.resolve_service("decomposition")
        except Exception:
            hyde_llm_config = self.config.llm.resolve_service("query")
        hyde_provider = create_provider(hyde_llm_config)

        pooled_embedding, hyp_docs = hyde_recall(
            request.question,
            hyde_provider,
            self.embedder,
            self.db,
            num_samples=query_config.hyde_num_samples,
            temperature=query_config.hyde_temperature,
            pooling=query_config.hyde_pooling,
            top_k=query_config.k_dense,
            batch_size=getattr(self.config, "batch_size", 32),
        )
        trace.stages_executed.append("hyde_generation")
        trace.hyde_documents = hyp_docs

        if not pooled_embedding:
            logger.warning("HyDE produced empty embedding; falling back to dense_only")
            return self._profile_dense_only(request, state, trace, extra_filters=extra_filters)

        # Use the pooled HyDE embedding for dense recall.
        plan = prepare_local_candidates(
            request.question, self.config, state, self.embedder, self.db,
            extra_filters=extra_filters,
            query_embedding=pooled_embedding,
        )
        dense_candidates = plan.filtered[
            : max(1, int(getattr(self.config.query, "k_dense", plan.k) or plan.k))
        ]
        self._populate_dense_score_breakdown(dense_candidates)
        trace.stages_executed.append("dense_recall")
        trace.dense_candidates_count = len(plan.candidates)

        # Also run sparse recall for hybrid fusion
        fts_available = self._is_sparse_recall_available()

        if fts_available:
            k_sparse = query_config.k_sparse
            sparse_candidates = sparse_recall(request.question, self.db, k_sparse)
            trace.stages_executed.append("sparse_recall")
            trace.sparse_candidates_count = len(sparse_candidates)

            fused = rrf_fuse(
                dense_candidates,
                sparse_candidates,
                dense_weight=query_config.rrf_dense_weight,
                sparse_weight=query_config.rrf_sparse_weight,
            )
            trace.stages_executed.append("rrf_fusion")
            local_candidates = fused
        else:
            local_candidates = dense_candidates

        # Cost tracking for HyDE generation
        input_tokens = len(request.question.split()) * 2
        output_tokens = sum(len(d.split()) for d in hyp_docs) * 2
        cost = hyde_provider.estimate_cost(input_tokens, output_tokens) or 0.0
        costs.append(
            CostEntry(
                provider=hyde_provider.name,
                model=hyde_provider.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                kind="hyde",
            )
        )

        return local_candidates, plan

    # ------------------------------------------------------------------
    # Stage 2: synthesize_context
    # ------------------------------------------------------------------
    def synthesize_context(self, package: ContextPackage) -> ContextPackage:
        """Assign citation labels and build the LLM context block.

        Populates ``context_block``, ``source_labels``, and ``starting_prompt``
        on the package.
        """
        context_block, sources_list = build_context_block(package.candidates)

        # Build label mapping
        source_labels: Dict[str, Any] = {}
        for i, entry in enumerate(sources_list, start=1):
            label = f"S{i}"
            source_labels[label] = entry
            # Also map to the candidate if in range
            if i <= len(package.candidates):
                source_labels[label] = {
                    **entry,
                    "candidate_index": i - 1,
                }

        # Resolve starting_prompt priority:
        # 1. Explicit from QueryRequest.starting_prompt
        # 2. Session cache continuation (prior_response_id)
        # 3. Mode-derived default
        mode_config = self._resolve_mode(package.request)
        starting_prompt: Optional[str]
        if package.request.starting_prompt is not None:
            starting_prompt = package.request.starting_prompt
        elif package.prior_response_id or package.request.prior_response_id:
            # Continuation requests rely on provider-side server cache.
            starting_prompt = None
        else:
            starting_prompt = mode_config.synthesis_instructions

        return replace(
            package,
            context_block=context_block,
            source_labels=source_labels,
            starting_prompt=starting_prompt,
        )

    # ------------------------------------------------------------------
    # Stage 3: execute
    # ------------------------------------------------------------------
    def execute(self, package: ContextPackage) -> QueryResponse:
        """Run LLM synthesis on the prepared context package.

        Returns a ``QueryResponse`` with the answer, citations, and costs.
        """
        if not package.context_block and not package.candidates:
            return QueryResponse(
                answer="No sources available to answer this question.",
                package=package,
            )

        mode_config = self._resolve_mode(package.request)

        # Resolve synthesis provider (respect model override)
        synthesis_provider = self.llm_provider
        if package.request.model_override:
            query_config = self.config.llm.resolve_service("query")
            overridden = replace(query_config, model=package.request.model_override)
            synthesis_provider = create_provider(overridden)

        # Build conversation payload
        question_payload = package.request.question
        if (
            package.request.chat_mode == "chat"
            and package.request.conversation_history
        ):
            history_lines: List[str] = []
            for turn in package.request.conversation_history[-10:]:
                role = (turn.get("role") or "user").upper()
                content = turn.get("content") or ""
                history_lines.append(f"{role}: {content}")
            question_payload = (
                "Conversation history:\n"
                + "\n".join(history_lines)
                + f"\n\nCurrent user question:\n{package.request.question}"
            )

        if package.starting_prompt is not None:
            instructions = package.starting_prompt
        elif package.prior_response_id:
            instructions = None
        else:
            instructions = mode_config.synthesis_instructions
        user_content = format_user_content(question_payload, package.context_block or "")

        # Build cache key for provider-side prompt caching
        conversation_scope = (
            package.request.conversation_id
            or package.prior_response_id
            or "single"
        )
        provider_cache_key = (
            f"{synthesis_provider.name}:{synthesis_provider.model}"
            f":{package.request.resolved_mode}:{conversation_scope}"
        )

        previous_response_id = package.prior_response_id
        enable_server_cache = self.config.query.enable_llm_server_cache
        prompt_cache_retention = getattr(
            self.config.query, "prompt_cache_retention", None
        )

        answer = synthesis_provider.send_message(
            input=user_content,
            instruction=instructions,
            temperature=self.config.llm.resolve_service("query").temperature,
            max_tokens=self.config.llm.resolve_service("query").max_tokens,
            reasoning_effort="medium",
            conversation_history=package.request.conversation_history or [],
            enable_server_cache=enable_server_cache,
            previous_response_id=previous_response_id,
            prompt_cache_key=provider_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

        # Capture response_id for next-turn chaining
        response_id = getattr(synthesis_provider, "last_response_id", None)
        if response_id:
            response_id = str(response_id)

        # Cost tracking
        costs: List[CostEntry] = []
        input_tokens = estimate_tokens(f"{question_payload}\n{package.context_block or ''}")
        output_tokens = estimate_output_tokens(
            answer, self.config.llm.resolve_service("query").max_tokens
        )
        cost = synthesis_provider.estimate_cost(input_tokens, output_tokens) or 0.0
        costs.append(
            CostEntry(
                provider=synthesis_provider.name,
                model=synthesis_provider.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                kind="synthesis",
            )
        )

        # Parse citations from answer
        citations = self._parse_citations(answer, package)

        # Model knowledge note
        model_knowledge_policy = mode_config.model_knowledge_policy
        if model_knowledge_policy.enabled:
            answer += (
                "\n\n---\n"
                "Note: Model knowledge is enabled for this mode. "
                "The answer may include information from the LLM's training data."
            )

        # Citation warning
        if "[S" not in answer:
            answer += (
                "\n\nNote: No inline citations were emitted. "
                "If this persists, reduce mode local_policy.k "
                "or reduce chunk size."
            )

        return QueryResponse(
            answer=answer,
            package=package,
            citations=citations,
            costs=costs,
            conversation_id=package.request.conversation_id,
            response_id=response_id,
        )

    # ------------------------------------------------------------------
    # run: chains all three stages
    # ------------------------------------------------------------------
    def run(
        self,
        request: QueryRequest,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> QueryResponse:
        """Execute the full pipeline: build_sources → synthesize_context → execute."""
        package = self.build_sources(request, progress_callback=progress_callback)

        # Relevance gate: if below threshold and no remote sources, use fallback
        mode_config = self._resolve_mode(request)
        local_policy = mode_config.local_policy
        remote_policy = mode_config.remote_policy
        if (
            package.local_enabled
            and package.relevance < local_policy.min_relevance
            and not remote_policy.enabled
        ):
            chosen = package.filtered_candidates[: local_policy.k]
            answer = fallback_answer(request.question, chosen)
            return QueryResponse(
                answer=answer,
                package=replace(package, candidates=chosen),
            )

        # Early exit: no candidates
        if package.local_enabled and not package.all_candidates and not remote_policy.enabled:
            return QueryResponse(
                answer="No results found in the knowledge base for this query.",
                package=package,
            )
        if package.local_enabled and not package.filtered_candidates and not remote_policy.enabled:
            return QueryResponse(
                answer="No results matched the configured filters.",
                package=package,
            )

        if progress_callback:
            progress_callback("synthesis", 0, 1, "Generating answer...")

        package = self.synthesize_context(package)
        response = self.execute(package)

        if progress_callback:
            progress_callback("synthesis", 1, 1, "Answer generation complete")

        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_mode(self, request: QueryRequest) -> ModeConfig:
        """Resolve mode config from request."""
        try:
            return self.config.get_mode_config(request.resolved_mode)
        except Exception:
            return GROUNDED_MODE

    def _request_to_state(self, request: QueryRequest) -> SessionState:
        """Build a minimal SessionState from a QueryRequest for planning."""
        filters = request.filters
        return SessionState(
            path_contains=filters.path_contains if filters else None,
            ext_allow=filters.ext_allow if filters else None,
            ext_deny=filters.ext_deny if filters else None,
            pinned_chunks=request.pinned_chunks or [],
            context_documents=request.context_documents or [],
            context_chunks=request.context_chunks or [],
        )

    def _parse_citations(
        self, answer: str, package: ContextPackage
    ) -> List[Citation]:
        """Extract citation references from the answer text."""
        citations: List[Citation] = []
        seen: set = set()
        for match in re.finditer(r"\[S(\d+)\]", answer):
            idx = int(match.group(1)) - 1
            if idx in seen or idx >= len(package.candidates):
                continue
            seen.add(idx)
            cand = package.candidates[idx]
            meta = cand.meta or {}
            citations.append(
                Citation(
                    chunk_id=cand.cid,
                    source_path=meta.get("source_path", "unknown"),
                    heading=meta.get("heading"),
                    page_number=meta.get("page_number"),
                    url_or_doi=meta.get("url") or meta.get("doi"),
                    snippet=(cand.text or "")[:200],
                )
            )
        return citations

    def _apply_post_retrieval_stages(
        self,
        request: QueryRequest,
        candidates: List[Candidate],
        trace: RetrievalTrace,
        final_k: int,
    ) -> List[Candidate]:
        """Apply dedup, temporal ranking, MMR, and section caps."""
        result = list(candidates)

        # 1) MinHash near-duplicate suppression.
        if result:
            dedup_t0 = time.monotonic()
            threshold = float(getattr(self.config.query, "dedup_threshold", 0.8))
            result = deduplicate_candidates(result, threshold=threshold)
            trace.stages_executed.append("minhash_dedup")
            trace.timings.append(
                StageTimings(
                    stage="minhash_dedup",
                    duration_ms=(time.monotonic() - dedup_t0) * 1000,
                )
            )

        # 2) Optional temporal boost.
        if result and bool(getattr(self.config.query, "temporal_boost_enabled", False)):
            temporal_t0 = time.monotonic()
            result = apply_temporal_boost(
                result,
                boost_days=int(getattr(self.config.query, "temporal_boost_days", 30)),
                boost_factor=float(
                    getattr(self.config.query, "temporal_boost_factor", 1.5)
                ),
            )
            trace.stages_executed.append("temporal_boost")
            trace.timings.append(
                StageTimings(
                    stage="temporal_boost",
                    duration_ms=(time.monotonic() - temporal_t0) * 1000,
                )
            )

        # 3) MMR diversity selection and per-section cap.
        final_k = max(1, int(final_k))
        if result:
            mmr_t0 = time.monotonic()
            try:
                query_vec = self.embedder.encode(
                    [request.question],
                    batch_size=getattr(self.config, "batch_size", 32),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )[0]
                query_embedding = (
                    query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
                )
                candidate_texts = [c.text or "" for c in result]
                candidate_vecs = self.embedder.encode(
                    candidate_texts,
                    batch_size=getattr(self.config, "batch_size", 32),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                for cand, emb in zip(result, candidate_vecs):
                    cand.embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            except Exception as exc:
                logger.debug("MMR embedding preparation failed: %s", exc)
                query_embedding = None

            result = mmr_select(
                result,
                query_embedding=query_embedding,
                lambda_param=float(getattr(self.config.query, "mmr_lambda", 0.7)),
                k=min(len(result), max(final_k, int(getattr(self.config.query, "k_dense", final_k) or final_k))),
            )
            trace.stages_executed.append("mmr_diversity")
            trace.timings.append(
                StageTimings(
                    stage="mmr_diversity",
                    duration_ms=(time.monotonic() - mmr_t0) * 1000,
                )
            )

            max_per_section = getattr(self.config.query, "max_per_section", None)
            if max_per_section is not None:
                cap_t0 = time.monotonic()
                result = per_section_cap(
                    result,
                    max_per_section=int(max_per_section),
                    heading_depth=2,
                )
                trace.stages_executed.append("per_section_cap")
                trace.timings.append(
                    StageTimings(
                        stage="per_section_cap",
                        duration_ms=(time.monotonic() - cap_t0) * 1000,
                    )
                )

        return result[: min(final_k, len(result))]

    @staticmethod
    def _populate_dense_score_breakdown(candidates: List[Candidate]) -> None:
        """Ensure dense recall candidates carry dense score/rank metadata."""
        for rank, candidate in enumerate(candidates, start=1):
            dense_score = None
            if candidate.distance is not None:
                dense_score = 1.0 - float(candidate.distance)
            prior = candidate.score_breakdown or ScoreBreakdown()
            candidate.score_breakdown = ScoreBreakdown(
                dense_score=dense_score if dense_score is not None else prior.dense_score,
                dense_rank=rank,
                sparse_score=prior.sparse_score,
                sparse_rank=prior.sparse_rank,
                fused_score=prior.fused_score,
                rerank_score=prior.rerank_score,
                temporal_boost=prior.temporal_boost,
            )

    def _is_sparse_recall_available(self) -> bool:
        """Detect whether provider-backed sparse recall is available."""
        provider_method = getattr(type(self.db), "fts_query", None)
        if provider_method is None:
            return False
        if provider_method is BaseVectorDBProvider.fts_query:
            return False
        if hasattr(self.db, "_extension_loaded") and not bool(
            getattr(self.db, "_extension_loaded")
        ):
            return False
        try:
            self.db.fts_query("lsm_sparse_probe", top_k=1)
        except Exception:
            return False
        return True
