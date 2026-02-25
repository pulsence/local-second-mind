"""
Context building utilities for query execution.

Provides helper functions for building query context from various sources:
- Local knowledge base (ChromaDB)
- Remote sources (web search, APIs)
- Combined context with proper citation labels
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple

from lsm.config.models import LSMConfig
from lsm.query.session import Candidate, SessionState
from lsm.query.planning import prepare_local_candidates, LocalQueryPlan
from lsm.remote import create_remote_provider
from lsm.remote.storage import load_cached_results, save_results
from lsm.logging import get_logger

logger = get_logger(__name__)


def _remote_provider_runtime_config(
    provider_config: Any,
    effective_weight: float,
    global_folder: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build provider runtime config preserving provider-specific passthrough fields."""
    runtime_config: Dict[str, Any] = {
        "type": provider_config.type,
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
    if global_folder is not None:
        runtime_config["global_folder"] = str(global_folder)
    if getattr(provider_config, "extra", None):
        runtime_config.update(provider_config.extra)
    return runtime_config


# -----------------------------
# Context Block Building
# -----------------------------
def build_context_block(candidates: List[Candidate]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build context block for LLM with source citations.

    Creates:
    - Context string for model input (sources prefixed [S1], [S2], ...)
    - Sources list for user display with metadata

    Args:
        candidates: List of candidates to include in context

    Returns:
        Tuple of (context_block, sources_list)
        - context_block: Formatted string with [S#] citations
        - sources_list: List of source metadata dicts

    Example:
        >>> context, sources = build_context_block(candidates)
        >>> context.startswith("[S1]")
        True
    """
    logger.debug(f"Building context block from {len(candidates)} candidates")

    sources_for_model: List[str] = []
    sources_for_user: List[Dict[str, Any]] = []

    for i, c in enumerate(candidates, start=1):
        label = f"S{i}"
        meta = c.meta or {}

        # Extract metadata
        source_path = meta.get("source_path", "unknown")
        source_name = meta.get("source_name")
        chunk_index = meta.get("chunk_index")
        ext = meta.get("ext")
        mtime_ns = meta.get("mtime_ns")
        file_hash = meta.get("file_hash")
        ingested_at = meta.get("ingested_at")
        title = meta.get("title")
        author = meta.get("author")

        # Build locator string with available metadata
        locator_bits = []
        if source_name:
            locator_bits.append(f"name={source_name}")
        if chunk_index is not None:
            locator_bits.append(f"chunk_index={chunk_index}")
        if ext:
            locator_bits.append(f"ext={ext}")
        if mtime_ns is not None:
            locator_bits.append(f"mtime_ns={mtime_ns}")
        if file_hash:
            locator_bits.append(f"file_hash={file_hash}")
        if ingested_at:
            locator_bits.append(f"ingested_at={ingested_at}")

        locator = " ".join(locator_bits)
        header = f"[{label}] {source_path}"
        if locator:
            header += f" ({locator})"

        # Add to model context
        sources_for_model.append(f"{header}\n{(c.text or '').strip()}\n")

        # Add to user sources list
        sources_for_user.append(
            {
                "label": label,
                "source_path": source_path,
                "source_name": source_name,
                "chunk_index": chunk_index,
                "ext": ext,
                "mtime_ns": mtime_ns,
                "file_hash": file_hash,
                "ingested_at": ingested_at,
                "title": title,
                "author": author,
            }
        )

    context_block = "\n\n".join(sources_for_model)

    logger.info(
        f"Built context block with {len(candidates)} sources "
        f"({len(context_block)} chars)"
    )

    return context_block, sources_for_user


# -----------------------------
# Fallback Answer Generation
# -----------------------------
def fallback_answer(
    question: str,
    candidates: List[Candidate],
    max_chars: int = 1200,
) -> str:
    """
    Generate minimal offline fallback answer.

    Returns top passages with citations when LLM is unavailable.
    This is not true synthesis, but useful when API is down.

    Args:
        question: User's question
        candidates: List of candidates to include
        max_chars: Maximum characters per excerpt

    Returns:
        Formatted fallback answer string

    Example:
        >>> answer = fallback_answer(
        ...     "What is Python?",
        ...     candidates,
        ...     max_chars=500
        ... )
    """
    logger.warning("Generating fallback answer (LLM unavailable)")

    lines = [
        "OpenAI is unavailable (quota/credentials). "
        "Showing the most relevant excerpts instead.",
        "",
        f"Question: {question}",
        "",
        "Top excerpts:",
    ]

    for i, c in enumerate(candidates, start=1):
        label = f"S{i}"
        excerpt = (c.text or "").strip()

        # Truncate long excerpts
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 50] + "\n...[truncated]..."

        meta = c.meta or {}
        source_path = meta.get("source_path", "unknown")
        chunk_index = meta.get("chunk_index", "NA")

        lines.append(
            f"\n[{label}] {source_path} (chunk_index={chunk_index})\n{excerpt}"
        )

    return "\n".join(lines)


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
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
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
    total_providers = len(active_providers)
    completed_providers = 0

    for provider_config in active_providers:
        provider_name = provider_config.name
        try:
            if progress_callback:
                progress_callback(
                    "remote",
                    completed_providers,
                    total_providers,
                    f"Fetching from {provider_name}...",
                )
            logger.info(f"Fetching from remote provider: {provider_name}")

            mode_weight = remote_policy.get_provider_weight(provider_name)
            effective_weight = mode_weight if mode_weight is not None else provider_config.weight
            cache_ttl = provider_config.cache_ttl if provider_config.cache_ttl is not None else 86400

            raw_results: List[Dict[str, Any]] | None = None
            if provider_config.cache_results:
                cached = load_cached_results(
                    provider_name=provider_name,
                    query=question,
                    global_folder=config.global_folder,
                    max_age=int(cache_ttl),
                )
                if cached is not None:
                    raw_results = []
                    for item in cached:
                        if not isinstance(item, dict):
                            continue
                        raw_results.append(
                            {
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "snippet": item.get("snippet", ""),
                                "score": item.get("score", 0.5),
                                "metadata": item.get("metadata") or {},
                            }
                        )
                    logger.info(
                        f"Using cached remote results for provider '{provider_name}' "
                        f"({len(raw_results)} results)"
                    )

            if raw_results is None:
                provider = create_remote_provider(
                    provider_config.type,
                    _remote_provider_runtime_config(
                        provider_config,
                        effective_weight,
                        config.global_folder,
                    ),
                )

                max_results = provider_config.max_results or remote_policy.max_results

                timeout_seconds = provider_config.timeout if provider_config.timeout is not None else 30
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(provider.search, question, max_results=max_results)
                    try:
                        results = future.result(timeout=max(1, int(timeout_seconds)))
                    except FuturesTimeoutError:
                        future.cancel()
                        raise TimeoutError(
                            f"remote provider timed out after {timeout_seconds}s"
                        )

                raw_results = [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "score": result.score,
                        "metadata": result.metadata or {},
                    }
                    for result in results
                ]

                if provider_config.cache_results:
                    try:
                        save_results(
                            provider_name=provider_name,
                            query=question,
                            results=raw_results,
                            global_folder=config.global_folder,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Failed saving remote cache for provider '{provider_name}': {exc}"
                        )

            for result in raw_results:
                base_score = result.get("score")
                base_score = base_score if base_score is not None else 0.5
                weighted_score = base_score * effective_weight
                all_remote_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "score": base_score,
                    "weight": effective_weight,
                    "weighted_score": weighted_score,
                    "provider": provider_name,
                    "metadata": result.get("metadata") or {},
                })
            completed_providers += 1

        except Exception as e:
            logger.error(f"Failed to fetch from {provider_name}: {e}")
            completed_providers += 1
            if progress_callback:
                progress_callback(
                    "remote",
                    completed_providers,
                    total_providers,
                    f"{provider_name} failed: {e}",
                )

    if remote_policy.rank_strategy == "weighted" and all_remote_results:
        all_remote_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        logger.info(f"Applied weighted ranking to {len(all_remote_results)} results")

    logger.info(f"Fetched {len(all_remote_results)} remote results")
    if progress_callback:
        progress_callback(
            "remote",
            total_providers,
            total_providers,
            f"Fetched {len(all_remote_results)} remote results",
        )
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
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
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
    remote_sources = fetch_remote_sources(
        question,
        config,
        mode_config,
        progress_callback=progress_callback,
    )
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
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
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

    if progress_callback:
        progress_callback("retrieval", 0, 1, "Searching local knowledge base...")
    local_candidates, plan = await loop.run_in_executor(
        None,
        lambda: build_local_context(question, config, state, embedder, collection)
    )
    if progress_callback:
        progress_callback("retrieval", 1, 1, "Local retrieval complete")

    remote_candidates, remote_sources = await loop.run_in_executor(
        None,
        lambda: build_remote_context(
            question,
            config,
            mode_config,
            progress_callback=progress_callback,
        )
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
