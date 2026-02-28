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
    StageTimings,
)
from lsm.query.planning import LocalQueryPlan, prepare_local_candidates
from lsm.query.rerank import llm_rerank_candidates
from lsm.query.session import Candidate, SessionState
from lsm.vectordb.base import BaseVectorDBProvider

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

        Returns a ``ContextPackage`` with ``candidates``, ``remote_sources``,
        and ``retrieval_trace`` populated.
        """
        t0 = time.monotonic()
        trace = RetrievalTrace()

        mode_config = self._resolve_mode(request)
        local_policy = mode_config.local_policy
        remote_policy = mode_config.remote_policy

        # Build a lightweight SessionState for the planning module
        state = self._request_to_state(request)

        # --- Local retrieval ---
        plan: Optional[LocalQueryPlan] = None
        local_candidates: List[Candidate] = []
        if local_policy.enabled:
            stage_t0 = time.monotonic()
            if progress_callback:
                progress_callback("retrieval", 0, 1, "Searching local knowledge base...")
            plan = prepare_local_candidates(
                request.question,
                self.config,
                state,
                self.embedder,
                self.db,
            )
            local_candidates = plan.filtered
            trace.stages_executed.append("local_retrieval")
            trace.dense_candidates_count = len(plan.candidates)
            trace.timings.append(
                StageTimings(
                    stage="local_retrieval",
                    duration_ms=(time.monotonic() - stage_t0) * 1000,
                )
            )
            if progress_callback:
                progress_callback("retrieval", 1, 1, "Local retrieval complete")

        # --- Local reranking (LLM) ---
        if plan and plan.should_llm_rerank:
            stage_t0 = time.monotonic()
            if progress_callback:
                progress_callback("rerank", 0, 1, "Reranking candidates...")
            ranking_config = self.config.llm.resolve_service("ranking")
            ranking_provider = create_provider(ranking_config)
            rerank_dicts = [
                {"text": c.text, "metadata": c.meta, "distance": c.distance}
                for c in plan.filtered
            ]
            k_rerank = min(plan.k, len(plan.filtered))
            reranked = llm_rerank_candidates(
                request.question, rerank_dicts, k=k_rerank, provider=ranking_provider
            )
            local_candidates = [
                Candidate(
                    cid=item.get("cid", ""),
                    text=item.get("text", ""),
                    meta=item.get("metadata", {}),
                    distance=item.get("distance"),
                )
                for item in reranked
            ]
            # Cost tracking for rerank
            rerank_est = estimate_rerank_cost(
                ranking_provider, request.question, plan.filtered, k=k_rerank
            )
            trace.stages_executed.append("llm_rerank")
            trace.reranked_candidates_count = len(local_candidates)
            trace.timings.append(
                StageTimings(
                    stage="llm_rerank",
                    duration_ms=(time.monotonic() - stage_t0) * 1000,
                )
            )
            if progress_callback:
                progress_callback("rerank", 1, 1, "Reranking complete")
        elif plan and not plan.should_llm_rerank:
            local_candidates = plan.filtered[: min(plan.k, len(plan.filtered))]

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
        trace.retrieval_profile = mode_config.retrieval_profile

        costs: List[CostEntry] = []
        # Record rerank cost if applicable
        if plan and plan.should_llm_rerank:
            ranking_config = self.config.llm.resolve_service("ranking")
            ranking_provider = create_provider(ranking_config)
            rerank_est = estimate_rerank_cost(
                ranking_provider,
                request.question,
                plan.filtered,
                k=min(plan.k, len(plan.filtered)),
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
        starting_prompt: str
        if package.request.starting_prompt is not None:
            starting_prompt = package.request.starting_prompt
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

        instructions = package.starting_prompt or mode_config.synthesis_instructions
        user_content = format_user_content(question_payload, package.context_block or "")

        # Build cache key for provider-side prompt caching
        provider_cache_key = (
            f"{synthesis_provider.name}:{synthesis_provider.model}"
            f":{package.request.resolved_mode}"
        )

        previous_response_id = package.prior_response_id
        enable_server_cache = self.config.query.enable_llm_server_cache

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
                "If this persists, reduce mode local_policy.k / query.local_pool "
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
