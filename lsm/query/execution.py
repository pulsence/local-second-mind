"""
Query execution orchestration.

Contains the core query execution functions:
- fetch_remote_sources: Fetches sources from remote providers
- run_query: CLI query handler
- run_query_turn: Synchronous query execution (for CLI)
- run_query_turn_async: Asynchronous query execution (for TUI)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Any

from lsm.config import load_config_from_file
from lsm.config.models import LSMConfig
from lsm.logging import get_logger
from lsm.providers import create_provider
from lsm.vectordb import create_vectordb_provider
from lsm.query.session import Candidate, SessionState
from lsm.query.retrieval import init_embedder
from lsm.query.synthesis import build_context_block, fallback_answer, format_source_list
from lsm.remote import create_remote_provider
from lsm.query.cost_tracking import (
    estimate_tokens,
    estimate_output_tokens,
    estimate_synthesis_cost,
    estimate_rerank_cost,
)
from lsm.query.planning import prepare_local_candidates

logger = get_logger(__name__)

def _stream_output(chunks) -> str:
    print("\nTyping...")
    parts: List[str] = []
    for chunk in chunks:
        if chunk:
            parts.append(chunk)
            print(chunk, end="", flush=True)
    print()
    return "".join(parts).strip()


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
def run_query(args: Any) -> int:
    """
    Run the query command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    # Load configuration
    cfg_path = Path(args.config).expanduser().resolve()

    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(f"Query config not found: {cfg_path}")

    logger.info(f"Loading configuration from: {cfg_path}")
    config = load_config_from_file(cfg_path)

    # Apply CLI overrides to config
    if hasattr(args, "mode") and args.mode:
        config.query.mode = args.mode
        logger.info(f"Overriding query mode to: {args.mode}")

    if hasattr(args, "model") and args.model:
        config.llm.override_feature_model("query", args.model)
        logger.info(f"Overriding LLM model to: {args.model}")

    if hasattr(args, "no_rerank") and args.no_rerank:
        config.query.no_rerank = True
        logger.info("Disabling reranking")

    if hasattr(args, "k") and args.k:
        config.query.k = args.k
        logger.info(f"Overriding retrieval depth to: {args.k}")

    logger.debug("Query configuration:")
    logger.debug(f"  Collection: {config.collection}")
    query_config = config.llm.get_query_config()
    logger.debug(f"  LLM: {query_config.provider}/{query_config.model}")
    logger.debug(f"  Retrieval: k={config.query.k}")
    logger.debug(f"  Mode: {config.query.mode}")

    # Single-shot mode only (TUI handles interactivity)
    question = getattr(args, "question", None)
    if not question:
        logger.info("Interactive query disabled for CLI; use TUI")
        print("Interactive query is now TUI-only. Run `lsm` to launch the TUI.")
        return 2

    try:
        logger.info(f"Running single-shot query: {question}")

        # Initialize embedder
        logger.info(f"Initializing embedder: {config.embed_model}")
        embedder = init_embedder(config.embed_model, device=config.device)

        # Initialize vector DB provider
        if config.vectordb.provider == "chromadb":
            persist_dir = Path(config.persist_dir)
            if not persist_dir.exists():
                print(f"Error: ChromaDB directory not found: {persist_dir}")
                print("Run 'lsm ingest' first to create the database.")
                return 1

        logger.info(f"Initializing vector DB provider: {config.vectordb.provider}")
        provider = create_vectordb_provider(config.vectordb)

        # Check collection has data
        count = provider.count()
        if count == 0:
            print(f"Warning: Collection '{config.collection}' is empty.")
            print("Run 'lsm ingest' to populate the database.")
            return 1

        logger.info(f"Collection ready with {count} chunks")

        # Initialize session state
        state = SessionState(
            path_contains=config.query.path_contains,
            ext_allow=config.query.ext_allow,
            ext_deny=config.query.ext_deny,
            model=query_config.model,
        )

        # Run single query
        run_query_turn(question, config, state, embedder, provider)

        return 0

    except Exception as e:
        logger.error(f"Single-shot query failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


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
    notes_config = mode_config.notes
    remote_policy = mode_config.source_policy.remote

    # Get configuration values (use mode-specific or fallback to query config)
    k = local_policy.k
    k_rerank = local_policy.k_rerank
    no_rerank = config.query.no_rerank
    min_relevance = local_policy.min_relevance

    state.last_question = question

    chosen: List[Candidate] = []
    filtered: List[Candidate] = []
    plan = prepare_local_candidates(question, config, state, embedder, collection)
    local_enabled = plan.local_enabled
    filtered = plan.filtered

    if local_enabled:
        state.last_all_candidates = plan.candidates
        state.last_filtered_candidates = plan.filtered

        if not plan.candidates and not remote_policy.enabled:
            print("No results found in Chroma for this query.\n")
            return

        if not plan.filtered and not remote_policy.enabled:
            print("No results matched the configured filters.\n")
            return

        state.last_debug = {
            "question": question,
            "retrieve_k": plan.retrieve_k,
            "k": plan.k,
            "k_rerank": plan.k_rerank,
            "filters_active": plan.filters_active,
            "path_contains": state.path_contains,
            "ext_allow": state.ext_allow,
            "ext_deny": state.ext_deny,
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
        }

        if plan.relevance < plan.min_relevance and not remote_policy.enabled:
            chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
            state.last_chosen = chosen
            state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

            answer = fallback_answer(question, chosen)
            _, sources = build_context_block(chosen)

            print("\n" + answer)
            print(format_source_list(sources))
            print()
            return

        if plan.should_llm_rerank:
            ranking_config = config.llm.get_ranking_config()
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
            if rerank_est["cost"] is not None:
                print(f"\nEstimated rerank cost: ${rerank_est['cost']:.4f}")

            reranked = provider.rerank(
                question,
                rerank_candidates,
                k=min(plan.k_rerank, len(plan.filtered)),
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
        else:
            chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
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
        answer = _stream_output(
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
# Async Query Turn (for TUI)
# -----------------------------
async def run_query_turn_async(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> Dict[str, Any]:
    """
    Execute one query turn and return results (for TUI use).

    This is an async-friendly version that returns results instead of printing.

    Args:
        question: User's question
        config: Global configuration
        state: Session state
        embedder: SentenceTransformer model
        collection: Vector DB provider

    Returns:
        Dict with 'response', 'candidates', 'cost', and 'remote_sources'
    """
    import asyncio

    logger.info(f"Running async query: {question[:50]}...")

    # Get mode configuration
    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local
    model_knowledge_policy = mode_config.source_policy.model_knowledge
    remote_policy = mode_config.source_policy.remote

    # Get configuration values

    state.last_question = question

    chosen: List[Candidate] = []
    filtered: List[Candidate] = []
    total_cost = 0.0
    loop = asyncio.get_event_loop()
    plan = await loop.run_in_executor(
        None, lambda: prepare_local_candidates(question, config, state, embedder, collection)
    )
    local_enabled = plan.local_enabled
    filtered = plan.filtered

    if local_enabled:
        state.last_all_candidates = plan.candidates
        state.last_filtered_candidates = plan.filtered

        if not plan.candidates and not remote_policy.enabled:
            return {
                "response": "No results found in the knowledge base for this query.",
                "candidates": [],
                "cost": 0.0,
                "remote_sources": [],
            }

        if not plan.filtered and not remote_policy.enabled:
            return {
                "response": "No results matched the configured filters.",
                "candidates": [],
                "cost": 0.0,
                "remote_sources": [],
            }

        state.last_debug = {
            "question": question,
            "retrieve_k": plan.retrieve_k,
            "k": plan.k,
            "k_rerank": plan.k_rerank,
            "filters_active": plan.filters_active,
            "path_contains": state.path_contains,
            "ext_allow": state.ext_allow,
            "ext_deny": state.ext_deny,
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
        }

        if plan.relevance < plan.min_relevance and not remote_policy.enabled:
            chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
            state.last_chosen = chosen
            state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

            answer = fallback_answer(question, chosen)
            return {
                "response": answer,
                "candidates": chosen,
                "cost": 0.0,
                "remote_sources": [],
            }

        if plan.should_llm_rerank:
            ranking_config = config.llm.get_ranking_config()
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

            reranked = provider.rerank(
                question,
                rerank_candidates,
                k=min(plan.k_rerank, len(plan.filtered)),
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
        else:
            chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
    else:
        state.last_all_candidates = []
        state.last_filtered_candidates = []

    state.last_chosen = chosen

    # Fetch remote sources if enabled (run in thread)
    loop = asyncio.get_event_loop()
    remote_sources = await loop.run_in_executor(
        None, lambda: fetch_remote_sources(question, config, mode_config)
    )
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

    # Run synthesis in thread
    answer = await loop.run_in_executor(
        None,
        lambda: synthesis_provider.synthesize(
            question,
            context_block,
            mode=mode_config.synthesis_style,
        )
    )

    # Warn if no citations
    if "[S" not in answer:
        answer += (
            "\n\nNote: No inline citations were emitted. "
            "If this persists, tighten query.k / query.k_rerank or reduce chunk size."
        )

    # Store state
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

    # Track costs
    if state.cost_tracker:
        input_tokens = estimate_tokens(f"{question}\n{context_block}")
        output_tokens = estimate_output_tokens(answer, query_config.max_tokens)
        cost = synthesis_provider.estimate_cost(input_tokens, output_tokens)
        total_cost = cost or 0.0
        state.cost_tracker.add_entry(
            provider=synthesis_provider.name,
            model=synthesis_provider.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            kind="synthesize",
        )

    # Format source list for display
    source_list = format_source_list(sources)

    return {
        "response": f"{answer}\n{source_list}",
        "candidates": combined_candidates,
        "cost": total_cost,
        "remote_sources": remote_sources,
    }


