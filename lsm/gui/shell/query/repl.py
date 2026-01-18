"""
REPL (Read-Eval-Print Loop) for interactive query sessions.

Provides the main loop and query execution orchestration.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Dict, Any

from lsm.config.models import LSMConfig
from lsm.gui.shell.logging import get_logger
from lsm.providers import create_provider
from lsm.vectordb.utils import require_chroma_collection
from lsm.query.session import Candidate, SessionState
from lsm.query.retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from lsm.query.rerank import apply_local_reranking
from lsm.query.synthesis import build_context_block, fallback_answer, format_source_list
from lsm.remote import create_remote_provider
from lsm.query.cost_tracking import CostTracker, estimate_tokens, estimate_output_tokens
from .display import print_banner, stream_output
from .commands import (
    handle_command,
    COMMAND_HINTS,
    estimate_synthesis_cost,
    estimate_rerank_cost,
)

logger = get_logger(__name__)


# -----------------------------
# Remote Source Fetching
# -----------------------------
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

    # Get active remote providers (optionally filtered per mode)
    # Extract provider names from refs (handles both string and dict formats)
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

            # Get mode-specific weight override, or fall back to global weight
            mode_weight = remote_policy.get_provider_weight(provider_name)
            effective_weight = mode_weight if mode_weight is not None else provider_config.weight

            # Create provider instance
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

            # Determine max_results
            max_results = provider_config.max_results or remote_policy.max_results

            # Fetch results
            results = provider.search(question, max_results=max_results)

            # Convert to dict format with weighted score
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
            # Continue with other providers

    # Apply ranking strategy
    if remote_policy.rank_strategy == "weighted" and all_remote_results:
        # Sort by weighted score (highest first)
        all_remote_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        logger.info(f"Applied weighted ranking to {len(all_remote_results)} results")

    logger.info(f"Fetched {len(all_remote_results)} remote results")
    return all_remote_results


def _build_remote_candidates(remote_sources: List[Dict[str, Any]]) -> List[Candidate]:
    """
    Convert remote sources to Candidate objects for context building.
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


# -----------------------------
# Query Execution
# -----------------------------
def run_query_turn(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> None:
    """
    Execute one query turn end-to-end.

    Args:
        question: User's question
        config: Global configuration
        state: Session state
        embedder: SentenceTransformer model
        collection: ChromaDB collection
    """
    logger.info(f"Running query: {question[:50]}...")

    # Get mode configuration
    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local
    model_knowledge_policy = mode_config.source_policy.model_knowledge
    notes_config = mode_config.notes
    remote_policy = mode_config.source_policy.remote

    # Get configuration values (use mode-specific or fallback to query config)
    batch_size = config.batch_size
    k = local_policy.k
    k_rerank = local_policy.k_rerank
    no_rerank = config.query.no_rerank
    min_relevance = local_policy.min_relevance
    max_per_file = config.query.max_per_file
    local_pool = config.query.local_pool or max(k * 3, k_rerank * 4)

    state.last_question = question

    chosen: List[Candidate] = []
    filtered: List[Candidate] = []
    local_enabled = local_policy.enabled

    if local_enabled:
        # Embed query
        query_vector = embed_text(embedder, question, batch_size=batch_size)

        # Get session filters
        path_contains = state.path_contains
        ext_allow = state.ext_allow
        ext_deny = state.ext_deny

        # Retrieve more candidates if filters are active
        filters_active = bool(path_contains) or bool(ext_allow) or bool(ext_deny)
        retrieve_k = config.query.retrieve_k or (max(k, k * 3) if filters_active else k)

        candidates = retrieve_candidates(collection, query_vector, retrieve_k)
        state.last_all_candidates = candidates

        if not candidates and not remote_policy.enabled:
            print("No results found in Chroma for this query.\n")
            return

        # Apply filters
        filtered = filter_candidates(
            candidates,
            path_contains=path_contains,
            ext_allow=ext_allow,
            ext_deny=ext_deny,
        )
        state.last_filtered_candidates = filtered

        # Add pinned chunks if any
        if state.pinned_chunks:
            logger.info(f"Including {len(state.pinned_chunks)} pinned chunks")
            print(f"Including {len(state.pinned_chunks)} pinned chunks in context...\n")

            # Fetch pinned chunks from collection
            try:
                chroma = require_chroma_collection(collection, "pinned chunk retrieval")
                pinned_results = chroma.get(
                    ids=state.pinned_chunks,
                    include=["documents", "metadatas", "distances"],
                )

                if pinned_results and pinned_results.get("ids"):
                    # Convert to Candidate objects
                    for i, chunk_id in enumerate(pinned_results["ids"]):
                        pinned_candidate = Candidate(
                            cid=chunk_id,
                            text=pinned_results["documents"][i],
                            meta=pinned_results["metadatas"][i],
                            distance=0.0,  # Force high relevance
                        )

                        # Add to front of filtered list if not already present
                        if not any(c.cid == chunk_id for c in filtered):
                            filtered.insert(0, pinned_candidate)

            except Exception as e:
                logger.error(f"Failed to load pinned chunks: {e}")
                print(f"Warning: Could not load pinned chunks: {e}\n")

        if not filtered and not remote_policy.enabled:
            print("No results matched the configured filters.\n")
            return

        # Determine reranking strategy
        rerank_strategy = config.query.rerank_strategy.lower()

        # Apply local reranking based on strategy
        if rerank_strategy in ("lexical", "hybrid"):
            # Local quality passes: dedupe, lexical rerank, diversity
            local = apply_local_reranking(
                question,
                filtered,
                max_per_file=max_per_file,
                local_pool=local_pool,
            )
            # Trim to k for downstream steps
            filtered = local[: min(k, len(local))]
        elif rerank_strategy == "none":
            # No local reranking, just use raw similarity order
            # Apply basic diversity enforcement
            from lsm.query.rerank import enforce_diversity
            filtered = enforce_diversity(filtered, max_per_file=max_per_file)
            filtered = filtered[: min(k, len(filtered))]
        # else: "llm" strategy - skip local reranking, will do LLM only

        # Relevance gating
        relevance = compute_relevance(filtered) if filtered else 0.0

        # Save debug info
        state.last_debug = {
            "question": question,
            "retrieve_k": retrieve_k,
            "k": k,
            "k_rerank": k_rerank,
            "filters_active": filters_active,
            "path_contains": path_contains,
            "ext_allow": ext_allow,
            "ext_deny": ext_deny,
            "best_relevance": relevance,
            "min_relevance": min_relevance,
            "rerank_strategy": rerank_strategy,
            "no_rerank": no_rerank,
            "model": state.model,
            "max_per_file": max_per_file,
            "local_pool": local_pool,
            "post_local_count": len(filtered),
            "local_enabled": local_enabled,
            "remote_enabled": remote_policy.enabled,
        }

        # If relevance is too low and no remote sources, skip LLM and show fallback
        if relevance < min_relevance and not remote_policy.enabled:
            chosen = filtered[: min(k_rerank, len(filtered))]
            state.last_chosen = chosen
            state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

            answer = fallback_answer(question, chosen)
            _, sources = build_context_block(chosen)

            print("\n" + answer)
            print(format_source_list(sources))
            print()
            return
    else:
        state.last_all_candidates = []
        state.last_filtered_candidates = []
        state.last_debug = {
            "question": question,
            "k": k,
            "k_rerank": k_rerank,
            "min_relevance": min_relevance,
            "model": state.model,
            "local_enabled": local_enabled,
            "remote_enabled": remote_policy.enabled,
        }

        if not filtered:
            chosen = []
        else:
            # Use ranking-specific LLM config if available
            ranking_config = config.llm.get_ranking_config()
            provider = create_provider(ranking_config)

            # Apply LLM reranking based on strategy
            rerank_strategy = config.query.rerank_strategy.lower()
            should_llm_rerank = rerank_strategy in ("llm", "hybrid") and not no_rerank

            if should_llm_rerank:
                # Convert candidates to provider format
                rerank_candidates = [
                    {
                        "text": c.text,
                        "metadata": c.meta,
                        "distance": c.distance,
                    }
                    for c in filtered
                ]

                rerank_est = estimate_rerank_cost(
                    provider,
                    question,
                    filtered,
                    k=min(k_rerank, len(filtered)),
                )
                if rerank_est["cost"] is not None:
                    print(f"\nEstimated rerank cost: ${rerank_est['cost']:.4f}")

                reranked = provider.rerank(
                    question,
                    rerank_candidates,
                    k=min(k_rerank, len(filtered)),
                )

                # Convert back to Candidate objects
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
            else:
                # No LLM reranking - use filtered results
                chosen = filtered[: min(k_rerank, len(filtered))]

    state.last_chosen = chosen

    # Fetch remote sources if enabled
    remote_sources = fetch_remote_sources(question, config, mode_config)
    remote_candidates = _build_remote_candidates(remote_sources)

    # Generate answer with citations
    combined_candidates = chosen + remote_candidates
    state.last_label_to_candidate = {
        f"S{i}": c for i, c in enumerate(combined_candidates, start=1)
    }
    context_block, sources = build_context_block(combined_candidates)

    # Use query-specific LLM config for synthesis
    query_config = config.llm.get_query_config()
    if state.model and state.model != query_config.model:
        query_config = replace(query_config, model=state.model)
    synthesis_provider = create_provider(query_config)

    synth_est = estimate_synthesis_cost(
        synthesis_provider,
        question,
        context_block,
        query_config.max_tokens,
    )
    if synth_est["cost"] is not None:
        print(f"\nEstimated synthesis cost: ${synth_est['cost']:.4f}")

    used_streaming = False
    try:
        answer = stream_output(
            synthesis_provider.stream_synthesize(
                question,
                context_block,
                mode=mode_config.synthesis_style,
            )
        )
        used_streaming = True
    except Exception as e:
        logger.warning(f"Streaming failed, falling back to non-streaming: {e}")
        answer = synthesis_provider.synthesize(
            question,
            context_block,
            mode=mode_config.synthesis_style,
        )

    # Warn if no citations
    if "[S" not in answer:
        answer += (
            "\n\nNote: No inline citations were emitted. "
            "If this persists, tighten query.k / query.k_rerank or reduce chunk size."
        )

    # Display answer and sources
    if not used_streaming:
        print("\n" + answer)
    print(format_source_list(sources))

    # Display remote sources if any
    if remote_sources:
        print("\n" + "=" * 60)
        print("REMOTE SOURCES")
        print("=" * 60)
        for i, remote in enumerate(remote_sources, 1):
            print(f"\n{i}. {remote['title']}")
            print(f"   {remote['url']}")
            print(f"   {remote['snippet'][:150]}...")

    # Display model knowledge note if enabled
    if model_knowledge_policy.enabled:
        print("\n" + "=" * 60)
        print("Note: Model knowledge is enabled for this mode.")
        print("The answer may include information from the LLM's training data.")
        print("=" * 60)

    print()

    # Store last query details for potential note saving
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

    if state.cost_tracker:
        input_tokens = estimate_tokens(f"{question}\n{context_block}")
        output_tokens = estimate_output_tokens(answer, query_config.max_tokens)
        cost = synthesis_provider.estimate_cost(input_tokens, output_tokens)
        state.cost_tracker.add_entry(
            provider=synthesis_provider.name,
            model=synthesis_provider.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            kind="synthesize",
        )
        status = state.cost_tracker.budget_status()
        if status and ("warning" in status.lower() or "exceeded" in status.lower()):
            print(status + "\n")


# -----------------------------
# Main REPL Loop
# -----------------------------
def run_repl(
    config: LSMConfig,
    embedder,
    collection,
) -> None:
    """
    Run the interactive REPL loop.

    Args:
        config: Global configuration
        embedder: SentenceTransformer model
        collection: ChromaDB collection
    """
    logger.info("Starting REPL session")

    # Initialize session state from config
    query_config = config.llm.get_query_config()
    state = SessionState(
        path_contains=config.query.path_contains,
        ext_allow=config.query.ext_allow,
        ext_deny=config.query.ext_deny,
        model=query_config.model,
        cost_tracker=CostTracker(),
    )

    print_banner()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not line:
            continue

        if not line.startswith("/"):
            token = line.split(maxsplit=1)[0].lower()
            if token in COMMAND_HINTS:
                print(f"Did you mean '/{token}'? Commands must start with '/'.\n")
                continue

        try:
            if handle_command(line, state, config, embedder, collection):
                continue
        except SystemExit:
            print("Exiting.")
            return

        run_query_turn(line, config, state, embedder, collection)
