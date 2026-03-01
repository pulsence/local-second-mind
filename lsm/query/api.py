"""
Clean public API for query operations.

This module provides the primary interface for executing queries.
All query operations should go through these functions.

Internally delegates to RetrievalPipeline for the three-stage execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

from lsm.config.models import LSMConfig
from lsm.providers import create_provider
from lsm.query.session import (
    Candidate,
    SessionState,
    append_chat_turn,
    save_conversation_markdown,
)
from lsm.query.context import build_context_block
from lsm.query.pipeline import RetrievalPipeline
from lsm.query.pipeline_types import FilterSet, QueryRequest
from lsm.ui.utils import format_source_list
from lsm.logging import get_logger
from lsm.paths import get_mode_chats_folder

logger = get_logger(__name__)


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
    Delegates to RetrievalPipeline for three-stage execution.

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
            progress_callback(
                QueryProgress(stage=stage, current=current, total=total, message=message)
            )

    mode_config = config.get_mode_config()
    local_policy = getattr(mode_config, "local_policy", mode_config.source_policy.local)
    chat_mode = config.query.chat_mode

    state.last_question = question
    _check_conversation_invalidation(config, state)

    if chat_mode == "chat":
        append_chat_turn(state, "user", question)

    # --- Build QueryRequest from session state ---
    request = QueryRequest(
        question=question,
        mode=config.query.mode,
        filters=FilterSet(
            path_contains=state.path_contains
            if isinstance(state.path_contains, list)
            else ([state.path_contains] if state.path_contains else None),
            ext_allow=state.ext_allow,
            ext_deny=state.ext_deny,
        ),
        k=local_policy.k,
        conversation_id=state.conversation_id,
        prior_response_id=state.prior_response_id,
        conversation_history=state.conversation_history if chat_mode == "chat" else None,
        chat_mode=chat_mode,
        pinned_chunks=state.pinned_chunks or None,
        context_documents=state.context_documents or None,
        context_chunks=state.context_chunks or None,
        model_override=state.model if state.model else None,
    )

    # --- Create pipeline and run ---
    query_config = config.llm.resolve_service("query")
    llm_provider = create_provider(query_config)

    pipeline = RetrievalPipeline(
        db=collection,
        embedder=embedder,
        config=config,
        llm_provider=llm_provider,
    )

    loop = asyncio.get_event_loop()

    def _pipeline_progress(stage, current, total, message):
        emit(stage, current, total, message)

    try:
        response = await loop.run_in_executor(
            None,
            lambda: pipeline.run(request, progress_callback=_pipeline_progress),
        )
    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}", exc_info=True)
        raise

    # --- Map QueryResponse back to SessionState and QueryResult ---
    package = response.package

    # Update session state
    state.last_all_candidates = package.all_candidates
    state.last_filtered_candidates = package.filtered_candidates
    state.last_chosen = list(package.candidates)
    state.last_label_to_candidate = {
        f"S{i}": c for i, c in enumerate(package.candidates, start=1)
    }
    state.last_answer = response.answer
    state.last_remote_sources = [rs.to_dict() for rs in package.remote_sources]
    state.last_local_sources_for_notes = [
        {"text": c.text, "meta": c.meta, "distance": c.distance}
        for c in package.candidates
        if not (c.meta or {}).get("remote")
    ]
    state.last_retrieval_trace = package.retrieval_trace.to_dict()

    # Conversation chain state
    if response.response_id:
        state.prior_response_id = response.response_id
    if response.conversation_id:
        state.conversation_id = response.conversation_id

    # Debug info
    state.last_debug = {
        "question": question,
        "k": local_policy.k,
        "model": state.model,
        "local_enabled": package.local_enabled,
        "remote_enabled": getattr(
            mode_config, "remote_policy", mode_config.source_policy.remote
        ).enabled,
        "relevance": package.relevance,
        "retrieval_profile": package.retrieval_trace.retrieval_profile,
        "stages": package.retrieval_trace.stages_executed,
        "total_duration_ms": package.retrieval_trace.total_duration_ms(),
    }

    if chat_mode == "chat":
        append_chat_turn(state, "assistant", response.answer)
        _maybe_auto_save_chat(config, state)

    # Cost tracking
    total_cost = response.total_cost()
    if state.cost_tracker:
        for entry in list(package.costs) + list(response.costs):
            state.cost_tracker.add_entry(
                provider=entry.provider,
                model=entry.model,
                input_tokens=entry.input_tokens,
                output_tokens=entry.output_tokens,
                cost=entry.cost,
                kind=entry.kind,
            )

    _, sources = build_context_block(package.candidates)
    source_list = format_source_list(sources)

    return QueryResult(
        answer=response.answer,
        candidates=list(package.candidates),
        sources_display=source_list,
        cost=total_cost,
        remote_sources=[rs.to_dict() for rs in package.remote_sources],
        debug_info=state.last_debug,
    )


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


def _check_conversation_invalidation(config: LSMConfig, state: SessionState) -> None:
    """Reset conversation chain state on model/mode changes."""
    query_service = config.llm.resolve_service("query")
    active_provider = str(getattr(query_service, "provider", ""))
    active_model = str(getattr(state, "model", "") or getattr(query_service, "model", ""))
    cache_key = f"{config.query.mode}:{active_provider}:{active_model}"
    previous_key = getattr(state, "_last_conversation_key", None)
    if previous_key is not None and previous_key != cache_key:
        state.prior_response_id = None
        state.conversation_id = None
    if config.query.chat_mode == "single":
        state.prior_response_id = None
        state.conversation_id = None
    state._last_conversation_key = cache_key
