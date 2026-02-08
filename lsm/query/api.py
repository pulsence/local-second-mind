"""
Clean public API for query operations.

This module provides the primary interface for executing queries.
All query operations should go through these functions.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable, Dict, Any, List, Optional

from lsm.config.models import LSMConfig
from lsm.providers import create_provider
from lsm.query.cache import QueryCache
from lsm.query.session import (
    Candidate,
    SessionState,
    append_chat_turn,
    save_conversation_markdown,
    serialize_conversation,
)
from lsm.query.context import (
    build_combined_context_async,
    build_remote_candidates,
    build_context_block,
    fallback_answer,
    ContextResult,
)
from lsm.ui.utils import format_source_list
from lsm.query.cost_tracking import (
    estimate_tokens,
    estimate_output_tokens,
    estimate_rerank_cost,
)
from lsm.logging import get_logger
from lsm.paths import get_mode_chats_folder

logger = get_logger(__name__)
_QUERY_CACHES: Dict[int, QueryCache] = {}


@dataclass
class QueryResult:
    """Result from a query operation."""

    answer: str
    """Synthesized answer text."""

    candidates: List[Candidate]
    """All candidates used in the answer."""

    sources_display: str
    """Formatted source list for display."""

    cost: float
    """Estimated cost for this query."""

    remote_sources: List[Dict[str, Any]]
    """Raw remote source data."""

    debug_info: Dict[str, Any]
    """Debug information from retrieval."""


@dataclass
class QueryProgress:
    """Progress update from query operation."""

    stage: str
    current: int
    total: int
    message: str


async def query(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
    progress_callback: Optional[Callable[[QueryProgress], None]] = None,
) -> QueryResult:
    """
    Execute a query and return results.

    This is the primary entry point for all query operations.
    Handles context building, reranking, and synthesis.

    Args:
        question: User's question
        config: Global configuration
        state: Session state (will be updated with results)
        embedder: SentenceTransformer model
        collection: Vector DB provider

    Returns:
        QueryResult with answer, candidates, and metadata

    Raises:
        ValueError: If question is empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    logger.info(f"Running query: {question[:50]}...")

    def emit(stage: str, current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(QueryProgress(stage=stage, current=current, total=total, message=message))

    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local
    remote_policy = mode_config.source_policy.remote
    model_knowledge_policy = mode_config.source_policy.model_knowledge
    chat_mode = config.query.chat_mode

    state.last_question = question
    total_cost = 0.0

    cache = _get_query_cache(config)
    cache_key = None
    if cache is not None:
        cache_filters = {
            "path_contains": state.path_contains,
            "ext_allow": state.ext_allow,
            "ext_deny": state.ext_deny,
            "context_documents": state.context_documents,
            "context_chunks": state.context_chunks,
        }
        cache_key = cache.build_key(
            query_text=question,
            mode=config.query.mode,
            filters=cache_filters,
            k=local_policy.k,
            k_rerank=local_policy.k_rerank,
            conversation=serialize_conversation(state) if chat_mode == "chat" else None,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            result = deepcopy(cached)
            state.last_debug = dict(result.debug_info or {})
            state.last_debug["cache_hit"] = True
            state.last_answer = result.answer
            state.last_chosen = list(result.candidates or [])
            state.last_label_to_candidate = {
                f"S{i}": candidate for i, candidate in enumerate(result.candidates or [], start=1)
            }
            state.last_remote_sources = list(result.remote_sources or [])
            state.last_local_sources_for_notes = [
                {
                    "text": c.text,
                    "meta": c.meta,
                    "distance": c.distance,
                }
                for c in (result.candidates or [])
                if not (c.meta or {}).get("remote")
            ]
            return result

    if chat_mode == "chat":
        append_chat_turn(state, "user", question)

    context_result = await build_combined_context_async(
        question,
        config,
        state,
        embedder,
        collection,
        progress_callback=lambda stage, current, total, message: emit(
            stage, current, total, message
        ),
    )

    plan = context_result.plan
    local_enabled = plan.local_enabled if plan else False
    filtered = context_result.local_candidates

    if local_enabled:
        state.last_all_candidates = plan.candidates
        state.last_filtered_candidates = plan.filtered

        if not plan.candidates and not remote_policy.enabled:
            return QueryResult(
                answer="No results found in the knowledge base for this query.",
                candidates=[],
                sources_display="",
                cost=0.0,
                remote_sources=[],
                debug_info={"local_enabled": True, "no_candidates": True},
            )

        if not plan.filtered and not remote_policy.enabled:
            return QueryResult(
                answer="No results matched the configured filters.",
                candidates=[],
                sources_display="",
                cost=0.0,
                remote_sources=[],
                debug_info={"local_enabled": True, "no_filtered": True},
            )

        state.last_debug = {
            "question": question,
            "retrieve_k": plan.retrieve_k,
            "k": plan.k,
            "k_rerank": plan.k_rerank,
            "filters_active": plan.filters_active,
            "path_contains": state.path_contains,
            "ext_allow": state.ext_allow,
            "ext_deny": state.ext_deny,
            "context_documents": state.context_documents,
            "context_chunks": state.context_chunks,
            "best_relevance": plan.relevance,
            "min_relevance": plan.min_relevance,
            "rerank_strategy": plan.rerank_strategy,
            "no_rerank": plan.no_rerank,
            "model": state.model,
            "max_per_file": plan.max_per_file,
            "local_pool": plan.local_pool,
            "post_local_count": len(plan.filtered),
            "local_enabled": local_enabled,
            "remote_enabled": remote_policy.enabled,
            "metadata_prefilter": getattr(plan, "metadata_filter", None),
        }

        if plan.relevance < plan.min_relevance and not remote_policy.enabled:
            chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
            state.last_chosen = chosen
            state.last_label_to_candidate = {
                f"S{i}": c for i, c in enumerate(chosen, start=1)
            }

            answer = fallback_answer(question, chosen)
            _, sources = build_context_block(chosen)

            return QueryResult(
                answer=answer,
                candidates=chosen,
                sources_display=format_source_list(sources),
                cost=0.0,
                remote_sources=[],
                debug_info=state.last_debug,
            )

        chosen = await _apply_reranking(
            question,
            plan,
            config,
            state,
            progress_callback=emit,
        )
    else:
        state.last_all_candidates = []
        state.last_filtered_candidates = []
        state.last_debug = {
            "question": question,
            "model": state.model,
            "local_enabled": local_enabled,
            "remote_enabled": remote_policy.enabled,
        }
        chosen = []

    state.last_chosen = chosen

    remote_candidates = context_result.remote_candidates
    remote_sources = context_result.remote_sources

    combined_candidates = chosen + remote_candidates
    state.last_label_to_candidate = {
        f"S{i}": c for i, c in enumerate(combined_candidates, start=1)
    }
    context_block, sources = build_context_block(combined_candidates)

    emit("synthesis", 0, 1, "Generating answer...")
    synthesis_failed = False
    try:
        answer, synthesis_cost = await _synthesize_answer(
            question, context_block, config, state, mode_config
        )
        total_cost += synthesis_cost
        emit("synthesis", 1, 1, "Answer generation complete")
    except Exception as exc:
        synthesis_failed = True
        logger.error(f"Synthesis failed, returning fallback answer: {exc}")
        answer = fallback_answer(question, combined_candidates)
        emit("synthesis", 1, 1, f"Synthesis failed, returned fallback: {exc}")

    if "[S" not in answer:
        answer += (
            "\n\nNote: No inline citations were emitted. "
            "If this persists, tighten query.k / query.k_rerank or reduce chunk size."
        )

    if model_knowledge_policy.enabled:
        answer += (
            "\n\n---\n"
            "Note: Model knowledge is enabled for this mode. "
            "The answer may include information from the LLM's training data."
        )

    _update_state(state, question, answer, chosen, remote_sources)
    if chat_mode == "chat":
        append_chat_turn(state, "assistant", answer)
        _maybe_auto_save_chat(config, state)

    source_list = format_source_list(sources)
    if synthesis_failed:
        state.last_debug["synthesis_fallback"] = True

    result = QueryResult(
        answer=answer,
        candidates=combined_candidates,
        sources_display=source_list,
        cost=total_cost,
        remote_sources=remote_sources,
        debug_info=state.last_debug,
    )
    if cache is not None and cache_key is not None:
        cache.set(cache_key, deepcopy(result))
    return result


def query_sync(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
    progress_callback: Optional[Callable[[QueryProgress], None]] = None,
) -> QueryResult:
    """
    Synchronous wrapper for query().

    For CLI or non-async contexts.

    Args:
        question: User's question
        config: Global configuration
        state: Session state
        embedder: SentenceTransformer model
        collection: Vector DB provider

    Returns:
        QueryResult with answer, candidates, and metadata
    """
    return asyncio.run(
        query(
            question,
            config,
            state,
            embedder,
            collection,
            progress_callback=progress_callback,
        )
    )


async def _apply_reranking(
    question: str,
    plan,
    config: LSMConfig,
    state: SessionState,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> List[Candidate]:
    """
    Apply reranking to candidates based on plan settings.

    Args:
        question: User's question
        plan: LocalQueryPlan from context building
        config: Global configuration
        state: Session state

    Returns:
        List of reranked candidates
    """
    if not plan.should_llm_rerank:
        return plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
    if progress_callback:
        progress_callback("rerank", 0, 1, "Reranking candidates...")

    ranking_config = config.llm.resolve_service("ranking")
    provider = create_provider(ranking_config)

    rerank_candidates = [
        {
            "text": c.text,
            "metadata": c.meta,
            "distance": c.distance,
        }
        for c in plan.filtered
    ]

    rerank_est = estimate_rerank_cost(
        provider,
        question,
        plan.filtered,
        k=min(plan.k_rerank, len(plan.filtered)),
    )

    loop = asyncio.get_event_loop()
    reranked = await loop.run_in_executor(
        None,
        lambda: provider.rerank(
            question,
            rerank_candidates,
            k=min(plan.k_rerank, len(plan.filtered)),
        )
    )

    chosen = []
    for item in reranked:
        chosen.append(
            Candidate(
                cid=item.get("cid", ""),
                text=item.get("text", ""),
                meta=item.get("metadata", {}),
                distance=item.get("distance"),
            )
        )

    if state.cost_tracker:
        cost = rerank_est["cost"]
        state.cost_tracker.add_entry(
            provider=provider.name,
            model=provider.model,
            input_tokens=rerank_est["input_tokens"],
            output_tokens=rerank_est["output_tokens"],
            cost=cost,
            kind="rerank",
        )

    if progress_callback:
        progress_callback("rerank", 1, 1, "Reranking complete")

    return chosen


async def _synthesize_answer(
    question: str,
    context_block: str,
    config: LSMConfig,
    state: SessionState,
    mode_config,
) -> tuple[str, float]:
    """
    Synthesize answer using LLM.

    Args:
        question: User's question
        context_block: Formatted context string
        config: Global configuration
        state: Session state
        mode_config: Mode configuration

    Returns:
        Tuple of (answer, cost)
    """
    query_config = config.llm.resolve_service("query")
    if state.model and state.model != query_config.model:
        query_config = replace(query_config, model=state.model)

    synthesis_provider = create_provider(query_config)
    provider_cache_key = f"{synthesis_provider.name}:{synthesis_provider.model}:{config.query.mode}"
    previous_response_id = None
    if config.query.chat_mode == "chat" and config.query.enable_llm_server_cache:
        previous_response_id = state.llm_server_cache_ids.get(provider_cache_key)

    loop = asyncio.get_event_loop()
    question_payload = question
    if config.query.chat_mode == "chat" and state.conversation_history:
        history_lines: List[str] = []
        for turn in state.conversation_history[-10:]:
            role = (turn.get("role") or "user").upper()
            content = turn.get("content") or ""
            history_lines.append(f"{role}: {content}")
        question_payload = (
            "Conversation history:\n"
            + "\n".join(history_lines)
            + f"\n\nCurrent user question:\n{question}"
        )

    answer = await loop.run_in_executor(
        None,
        lambda: synthesis_provider.synthesize(
            question_payload,
            context_block,
            mode=mode_config.synthesis_style,
            conversation_history=state.conversation_history,
            enable_server_cache=config.query.enable_llm_server_cache,
            previous_response_id=previous_response_id,
            prompt_cache_key=provider_cache_key,
        )
    )

    if config.query.enable_llm_server_cache:
        response_id = getattr(synthesis_provider, "last_response_id", None)
        if response_id:
            state.llm_server_cache_ids[provider_cache_key] = str(response_id)

    cost = 0.0
    if state.cost_tracker:
        input_tokens = estimate_tokens(f"{question_payload}\n{context_block}")
        output_tokens = estimate_output_tokens(answer, query_config.max_tokens)
        cost = synthesis_provider.estimate_cost(input_tokens, output_tokens) or 0.0
        state.cost_tracker.add_entry(
            provider=synthesis_provider.name,
            model=synthesis_provider.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            kind="synthesize",
        )

    return answer, cost


def _update_state(
    state: SessionState,
    question: str,
    answer: str,
    chosen: List[Candidate],
    remote_sources: List[Dict[str, Any]],
) -> None:
    """
    Update session state with query results.

    Args:
        state: Session state to update
        question: User's question
        answer: Generated answer
        chosen: Chosen local candidates
        remote_sources: Remote source data
    """
    state.last_answer = answer
    state.last_remote_sources = remote_sources
    state.last_local_sources_for_notes = [
        {
            "text": c.text,
            "meta": c.meta,
            "distance": c.distance,
        }
        for c in chosen
    ]


def _get_query_cache(config: LSMConfig) -> QueryCache | None:
    """Get or initialize a cache for this config instance."""
    if not config.query.enable_query_cache:
        return None
    key = id(config)
    cache = _QUERY_CACHES.get(key)
    if cache is None:
        cache = QueryCache(
            ttl_seconds=config.query.query_cache_ttl,
            max_size=config.query.query_cache_size,
        )
        _QUERY_CACHES[key] = cache
    return cache


def _maybe_auto_save_chat(config: LSMConfig, state: SessionState) -> None:
    """Auto-save chat transcript if enabled."""
    if config.query.chat_mode != "chat":
        return
    mode_chat_auto_save = config.chats.auto_save
    mode_chat_dir = config.chats.dir
    try:
        mode_config = config.get_mode_config(config.query.mode)
        if mode_config.chats is not None:
            if mode_config.chats.auto_save is not None:
                mode_chat_auto_save = mode_config.chats.auto_save
            if mode_config.chats.dir:
                mode_chat_dir = mode_config.chats.dir
    except Exception as exc:
        logger.debug(f"Could not resolve mode chat overrides for auto-save: {exc}")

    if not config.chats.enabled or not mode_chat_auto_save:
        return
    if not state.conversation_history:
        return
    try:
        chats_dir = get_mode_chats_folder(
            mode_name=config.query.mode,
            global_folder=config.global_folder,
            base_dir=mode_chat_dir,
        )
        save_conversation_markdown(state, chats_dir=chats_dir, mode_name=config.query.mode)
    except Exception as exc:
        logger.warning(f"Failed to auto-save chat transcript: {exc}")
