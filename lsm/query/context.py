"""
Context building utilities for query execution.

Provides helper functions for building query context from various sources:
- Local knowledge base (ChromaDB)
- Remote sources (web search, APIs)
- Combined context with proper citation labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from lsm.config.models import LSMConfig
from lsm.query.session import Candidate, SessionState
from lsm.query.planning import prepare_local_candidates, LocalQueryPlan
from lsm.query.synthesis import build_context_block
from lsm.remote import create_remote_provider
from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContextResult:
    """Result from building query context."""

    candidates: List[Candidate]
    """Combined candidates (local + remote)."""

    context_block: str
    """Formatted context string for LLM."""

    sources: List[Dict[str, Any]]
    """Source metadata for display."""

    local_candidates: List[Candidate]
    """Local candidates only."""

    remote_candidates: List[Candidate]
    """Remote candidates only."""

    remote_sources: List[Dict[str, Any]]
    """Raw remote source data."""

    plan: Optional[LocalQueryPlan]
    """Local query plan if local sources were enabled."""


def fetch_remote_sources(
    question: str,
    config: LSMConfig,
    mode_config: Any,
) -> List[Dict[str, Any]]:
    """
    Fetch remote sources if enabled in mode configuration.

    Args:
        question: User's question
        config: Global configuration
        mode_config: Mode configuration with source policies

    Returns:
        List of remote source dicts, sorted by weighted score if rank_strategy is 'weighted'
    """
    remote_policy = mode_config.source_policy.remote

    if not remote_policy.enabled:
        return []

    allowed_names = None
    if remote_policy.remote_providers is not None:
        allowed_names = set(remote_policy.get_provider_names())

    active_providers = config.get_active_remote_providers(allowed_names=allowed_names)

    if allowed_names:
        known_names = {provider.name.lower() for provider in config.remote_providers or []}
        for name in allowed_names:
            if name.lower() not in known_names:
                logger.warning(f"Mode requested unknown remote provider: {name}")

    if not active_providers:
        logger.warning("Remote sources enabled but no providers configured")
        return []

    all_remote_results = []

    for provider_config in active_providers:
        provider_name = provider_config.name
        try:
            logger.info(f"Fetching from remote provider: {provider_name}")

            mode_weight = remote_policy.get_provider_weight(provider_name)
            effective_weight = mode_weight if mode_weight is not None else provider_config.weight

            provider = create_remote_provider(
                provider_config.type,
                {
                    "type": provider_config.type,
                    "enabled": provider_config.enabled,
                    "weight": effective_weight,
                    "api_key": provider_config.api_key,
                    "endpoint": provider_config.endpoint,
                    "max_results": provider_config.max_results,
                    "language": provider_config.language,
                    "user_agent": provider_config.user_agent,
                    "timeout": provider_config.timeout,
                    "min_interval_seconds": provider_config.min_interval_seconds,
                    "section_limit": provider_config.section_limit,
                    "snippet_max_chars": provider_config.snippet_max_chars,
                    "include_disambiguation": provider_config.include_disambiguation,
                }
            )

            max_results = provider_config.max_results or remote_policy.max_results

            results = provider.search(question, max_results=max_results)

            for result in results:
                base_score = result.score if result.score is not None else 0.5
                weighted_score = base_score * effective_weight
                all_remote_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "score": result.score,
                    "weight": effective_weight,
                    "weighted_score": weighted_score,
                    "provider": provider_name,
                    "metadata": result.metadata,
                })

        except Exception as e:
            logger.error(f"Failed to fetch from {provider_name}: {e}")

    if remote_policy.rank_strategy == "weighted" and all_remote_results:
        all_remote_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        logger.info(f"Applied weighted ranking to {len(all_remote_results)} results")

    logger.info(f"Fetched {len(all_remote_results)} remote results")
    return all_remote_results


def build_remote_candidates(remote_sources: List[Dict[str, Any]]) -> List[Candidate]:
    """
    Convert remote sources to Candidate objects for context building.

    Args:
        remote_sources: List of remote source dicts from fetch_remote_sources()

    Returns:
        List of Candidate objects
    """
    candidates: List[Candidate] = []
    for idx, source in enumerate(remote_sources, start=1):
        title = source.get("title") or ""
        snippet = source.get("snippet") or ""
        url = source.get("url") or ""
        provider = source.get("provider") or "remote"

        text_parts = [part for part in (title, snippet) if part]
        text = "\n".join(text_parts).strip() or url or title or provider

        meta = {
            "source_path": url or f"{provider}_result_{idx}",
            "source_name": title or provider,
            "title": title or None,
            "author": None,
            "remote_provider": provider,
            "remote": True,
        }

        candidates.append(
            Candidate(
                cid=f"remote:{provider}:{idx}",
                text=text,
                meta=meta,
                distance=None,
            )
        )

    return candidates


def build_local_context(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> Tuple[List[Candidate], Optional[LocalQueryPlan]]:
    """
    Build context from local knowledge base.

    Args:
        question: User's question
        config: Global configuration
        state: Session state
        embedder: SentenceTransformer model
        collection: Vector DB provider

    Returns:
        Tuple of (candidates, plan)
    """
    plan = prepare_local_candidates(question, config, state, embedder, collection)

    if not plan.local_enabled:
        return [], plan

    return plan.filtered, plan


def build_remote_context(
    question: str,
    config: LSMConfig,
    mode_config: Any,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Build context from remote sources.

    Args:
        question: User's question
        config: Global configuration
        mode_config: Mode configuration with remote policy

    Returns:
        Tuple of (candidates, raw_remote_sources)
    """
    remote_sources = fetch_remote_sources(question, config, mode_config)
    candidates = build_remote_candidates(remote_sources)
    return candidates, remote_sources


def build_combined_context(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> ContextResult:
    """
    Build combined context from all enabled sources.

    This is the main entry point for context building.
    Orchestrates local and remote source fetching based on mode config.

    Args:
        question: User's question
        config: Global configuration
        state: Session state
        embedder: SentenceTransformer model
        collection: Vector DB provider

    Returns:
        ContextResult with all context data
    """
    mode_config = config.get_mode_config()

    local_candidates, plan = build_local_context(
        question, config, state, embedder, collection
    )

    remote_candidates, remote_sources = build_remote_context(
        question, config, mode_config
    )

    combined = local_candidates + remote_candidates
    context_block, sources = build_context_block(combined)

    return ContextResult(
        candidates=combined,
        context_block=context_block,
        sources=sources,
        local_candidates=local_candidates,
        remote_candidates=remote_candidates,
        remote_sources=remote_sources,
        plan=plan,
    )


async def build_combined_context_async(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> ContextResult:
    """
    Async version of build_combined_context.

    Runs local context building in a thread to avoid blocking.

    Args:
        question: User's question
        config: Global configuration
        state: Session state
        embedder: SentenceTransformer model
        collection: Vector DB provider

    Returns:
        ContextResult with all context data
    """
    import asyncio

    loop = asyncio.get_event_loop()
    mode_config = config.get_mode_config()

    local_candidates, plan = await loop.run_in_executor(
        None,
        lambda: build_local_context(question, config, state, embedder, collection)
    )

    remote_candidates, remote_sources = await loop.run_in_executor(
        None,
        lambda: build_remote_context(question, config, mode_config)
    )

    combined = local_candidates + remote_candidates
    context_block, sources = build_context_block(combined)

    return ContextResult(
        candidates=combined,
        context_block=context_block,
        sources=sources,
        local_candidates=local_candidates,
        remote_candidates=remote_candidates,
        remote_sources=remote_sources,
        plan=plan,
    )
