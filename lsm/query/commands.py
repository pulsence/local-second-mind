"""
Command handlers for the query module.

Contains business logic handlers that return results as strings.
UI layers (TUI, shell) are responsible for display.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Any, Optional

from lsm.config.models import LSMConfig, LLMConfig
from lsm.logging import get_logger
from lsm.providers import create_provider
from lsm.vectordb.utils import require_chroma_collection
from lsm.vectordb import create_vectordb_provider, list_available_providers
from lsm.query.session import Candidate, SessionState
from lsm.query.retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from lsm.query.rerank import apply_local_reranking
from lsm.query.synthesis import build_context_block
from lsm.remote import create_remote_provider, get_registered_providers
from lsm.query.notes import get_note_filename, generate_note_content
from lsm.query.cost_tracking import estimate_tokens
from lsm.query.citations import export_citations_from_note, export_citations_from_sources
from .display import (
    get_help,
    format_source_chunk,
    format_debug,
    format_costs,
    display_provider_name,
    format_feature_label,
)

logger = get_logger(__name__)

COMMAND_HINTS = {
    "exit",
    "help",
    "show",
    "expand",
    "open",
    "debug",
    "model",
    "models",
    "providers",
    "provider-status",
    "vectordb-providers",
    "vectordb-status",
    "remote-providers",
    "remote-provider",
    "remote-search",
    "remote-search-all",
    "mode",
    "note",
    "notes",
    "load",
    "set",
    "clear",
    "costs",
    "budget",
    "cost-estimate",
    "export-citations",
}


# Field-specific provider recommendations
PROVIDER_RECOMMENDATIONS = {
    "stem": {
        "physics": ["arxiv", "semantic_scholar", "openalex"],
        "math": ["arxiv", "semantic_scholar", "openalex"],
        "cs": ["arxiv", "semantic_scholar", "openalex", "core"],
        "biology": ["semantic_scholar", "openalex", "core"],
        "medicine": ["semantic_scholar", "openalex", "core"],
        "general": ["openalex", "crossref", "core", "semantic_scholar"],
    },
    "humanities": {
        "philosophy": ["philpapers", "openalex"],
        "theology": ["ixtheo", "philpapers", "openalex"],
        "religious_studies": ["ixtheo", "philpapers", "openalex"],
        "history": ["openalex", "crossref"],
        "classics": ["openalex", "crossref"],
    },
    "social_sciences": {
        "economics": ["openalex", "crossref"],
        "psychology": ["semantic_scholar", "openalex"],
        "sociology": ["openalex", "crossref", "core"],
        "general": ["openalex", "crossref", "core"],
    },
    "cross_disciplinary": {
        "citation_lookup": ["crossref"],
        "open_access": ["core", "openalex"],
        "comprehensive": ["openalex", "crossref", "semantic_scholar"],
    },
}


# -----------------------------
# Model Management
# -----------------------------
def list_models(provider) -> List[str]:
    """
    List models available to the current API key.

    Args:
        provider: LLM provider instance

    Returns:
        List of model IDs (sorted)
    """
    ids = provider.list_models()
    ids.sort()
    logger.info(f"Found {len(ids)} available models")
    return ids


def format_models(state: SessionState, provider) -> str:
    """
    Format available models as a string.

    Args:
        state: Session state (to cache model list)
        provider: LLM provider instance

    Returns:
        Formatted model list string
    """
    try:
        ids = list_models(provider)
        state.available_models = ids

        if not ids:
            return "  (no models returned or listing unsupported)"

        lines = [f"  - {model_id}" for model_id in ids]
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return f"Failed to list models: {e}\n"


def _get_feature_configs(config: LSMConfig) -> dict[str, Optional[LLMConfig]]:
    feature_map = config.llm.get_feature_provider_map()
    return {
        "query": config.llm.get_query_config() if "query" in feature_map else None,
        "tagging": config.llm.get_tagging_config() if "tagging" in feature_map else None,
        "ranking": config.llm.get_ranking_config() if "ranking" in feature_map else None,
    }


def format_providers(config: LSMConfig) -> str:
    """
    Format available LLM providers as a string.

    Args:
        config: LSM configuration

    Returns:
        Formatted provider list string
    """
    lines = [
        "",
        "=" * 60,
        "AVAILABLE LLM PROVIDERS",
        "=" * 60,
        "",
    ]

    providers = config.llm.get_provider_names()

    if not providers:
        lines.append("No providers configured.")
        lines.append("")
        return "\n".join(lines)

    lines.append("Selections:")
    feature_configs = _get_feature_configs(config)
    for feature, cfg in feature_configs.items():
        if cfg is None:
            continue
        label = format_feature_label(feature)
        provider = display_provider_name(cfg.provider)
        lines.append(f"  {label:7s} {provider}/{cfg.model}")
    lines.append("")

    lines.append(f"Providers ({len(providers)}):")
    lines.append("")

    seen_labels = set()
    for provider_name in providers:
        try:
            provider_config = config.llm.get_provider_by_name(provider_name)
            test_config = provider_config.resolve_first_available() if provider_config else None

            if test_config:
                provider = create_provider(test_config)
                is_available = "ok" if provider.is_available() else "- (API key not configured)"
            elif provider_config:
                is_available = "- (no feature config)"
            else:
                is_available = "- (not configured)"

            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"  {label:20s} {is_available:30s}")

        except Exception as e:
            logger.debug(f"Error checking provider {provider_name}: {e}")
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"  {label:20s} {'- (error)':30s}")

    lines.append("")
    lines.append("To switch providers, update your config.json:")
    lines.append('  "llms": [ { "provider_name": "provider_name", ... } ]')
    lines.append("")

    return "\n".join(lines)


def format_provider_status(config: LSMConfig) -> str:
    """
    Format provider health status and call statistics.

    Args:
        config: LSM configuration

    Returns:
        Formatted provider status string
    """
    lines = [
        "",
        "=" * 60,
        "PROVIDER HEALTH STATUS",
        "=" * 60,
        "",
    ]

    providers = config.llm.get_provider_names()
    if not providers:
        lines.append("No providers registered.")
        lines.append("")
        return "\n".join(lines)

    current_provider = config.llm.get_query_config().provider

    seen_labels = set()
    for provider_name in providers:
        try:
            provider_config = config.llm.get_provider_by_name(provider_name)
            test_config = provider_config.resolve_first_available() if provider_config else None
            if not test_config:
                status = "not_configured" if not provider_config else "missing_config"
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"{label:20s} status={status:12s}")
                continue

            provider = create_provider(test_config)
            health = provider.health_check()

            status = health.get("status", "unknown")
            stats = health.get("stats", {})
            success = stats.get("success_count", 0)
            failure = stats.get("failure_count", 0)
            last_error = stats.get("last_error")
            current_label = " (current)" if provider_name == current_provider else ""
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)

            lines.append(
                f"{label:20s} status={status:12s} success={success:4d} failure={failure:4d}{current_label}"
            )
            if last_error:
                lines.append(f"{'':20s} last_error={last_error}")
        except Exception as e:
            logger.debug(f"Error checking provider status {provider_name}: {e}")
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"{label:20s} status=error        error={e}")

    lines.append("")

    return "\n".join(lines)


def format_vectordb_providers(config: LSMConfig) -> str:
    """Format available vector DB providers as a string."""
    providers = list_available_providers()

    lines = [
        "",
        "=" * 60,
        "AVAILABLE VECTOR DB PROVIDERS",
        "=" * 60,
        "",
    ]

    if not providers:
        lines.append("No vector DB providers registered.")
        lines.append("")
        return "\n".join(lines)

    current_provider = config.vectordb.provider
    lines.append(f"Current Provider: {current_provider}")
    lines.append(f"Collection:       {config.vectordb.collection}")
    lines.append("")

    lines.append(f"Available Providers ({len(providers)}):")
    lines.append("")

    for provider_name in providers:
        is_current = "ACTIVE" if provider_name == current_provider else ""
        status = ""
        if provider_name == current_provider:
            try:
                provider = create_vectordb_provider(config.vectordb)
                status = "ok" if provider.is_available() else "unavailable"
            except Exception as e:
                status = f"error ({e})"
        lines.append(f"  {provider_name:20s} {status:20s} {is_current}")

    lines.append("")
    lines.append("To switch providers, update your config.json:")
    lines.append('  "vectordb": { "provider": "provider_name", ... }')
    lines.append("")

    return "\n".join(lines)


def format_vectordb_status(config: LSMConfig) -> str:
    """Format vector DB provider health and stats."""
    lines = [
        "",
        "=" * 60,
        "VECTOR DB STATUS",
        "=" * 60,
        "",
    ]

    try:
        provider = create_vectordb_provider(config.vectordb)
        health = provider.health_check()
        stats = provider.get_stats()

        lines.append(f"Provider: {health.get('provider', 'unknown')}")
        lines.append(f"Status:   {health.get('status', 'unknown')}")
        if health.get("error"):
            lines.append(f"Error:    {health.get('error')}")
        lines.append(f"Count:    {stats.get('count', 'n/a')}")
    except Exception as e:
        lines.append(f"Error: {e}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("PROVIDER HEALTH STATUS")
    lines.append("=" * 60)
    lines.append("")

    providers = config.llm.get_provider_names()
    if not providers:
        lines.append("No providers registered.")
        lines.append("")
        return "\n".join(lines)

    current_provider = config.llm.get_query_config().provider

    seen_labels = set()
    for provider_name in providers:
        try:
            provider_config = config.llm.get_provider_by_name(provider_name)
            test_config = provider_config.resolve_first_available() if provider_config else None
            if not test_config:
                status = "not_configured" if not provider_config else "missing_config"
                label = display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                lines.append(f"{label:20s} status={status:12s}")
                continue

            provider = create_provider(test_config)
            health = provider.health_check()

            status = health.get("status", "unknown")
            stats = health.get("stats", {})
            success = stats.get("success_count", 0)
            failure = stats.get("failure_count", 0)
            last_error = stats.get("last_error")
            current_label = " (current)" if provider_name == current_provider else ""
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)

            lines.append(
                f"{label:20s} status={status:12s} "
                f"success={success:4d} failure={failure:4d}{current_label}"
            )
            if last_error:
                lines.append(f"{'':20s} last_error={last_error}")
        except Exception as e:
            logger.debug(f"Error checking provider status {provider_name}: {e}")
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            lines.append(f"{label:20s} status=error        error={e}")

    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Remote Provider Management
# -----------------------------
def format_remote_providers(config: LSMConfig) -> str:
    """Format available remote source providers and their status."""
    lines = [
        "",
        "=" * 60,
        "REMOTE SOURCE PROVIDERS",
        "=" * 60,
        "",
    ]

    # Show registered provider types
    registered = get_registered_providers()
    lines.append(f"Registered Provider Types ({len(registered)}):")
    for provider_type, provider_class in sorted(registered.items()):
        lines.append(f"  {provider_type:20s} -> {provider_class.__name__}")
    lines.append("")

    # Show configured providers
    configured = config.remote_providers or []
    if not configured:
        lines.append("No remote providers configured.")
        lines.append("")
        lines.append("Add providers to your config.json:")
        lines.append('  "remote_providers": [{"name": "...", "type": "...", ...}]')
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Configured Providers ({len(configured)}):")
    lines.append("")
    lines.append(f"  {'NAME':<20s} {'TYPE':<20s} {'STATUS':<10s} {'WEIGHT':<8s} {'API KEY'}")
    lines.append("  " + "-" * 70)

    for provider_config in configured:
        name = provider_config.name
        ptype = provider_config.type
        status = "enabled" if provider_config.enabled else "disabled"
        weight = f"{provider_config.weight:.1f}"
        api_key = provider_config.api_key
        has_key = "set" if api_key and not api_key.startswith("INSERT") else "not set"
        # For providers that don't require API keys
        if ptype in {"wikipedia", "arxiv", "openalex", "crossref", "oai_pmh"}:
            has_key = "n/a"
        lines.append(f"  {name:<20s} {ptype:<20s} {status:<10s} {weight:<8s} {has_key}")

    lines.append("")

    # Show current mode's remote settings
    mode_config = config.get_mode_config()
    remote_policy = mode_config.source_policy.remote
    lines.append("Current Mode Remote Settings:")
    lines.append(f"  Enabled:       {remote_policy.enabled}")
    lines.append(f"  Max Results:   {remote_policy.max_results}")
    lines.append(f"  Rank Strategy: {remote_policy.rank_strategy}")
    if remote_policy.remote_providers:
        lines.append(f"  Mode Providers: {', '.join(str(p) for p in remote_policy.remote_providers)}")
    lines.append("")

    lines.append("Commands:")
    lines.append("  /remote-search <provider> <query>  Test a provider")
    lines.append("  /remote-search-all <query>         Search all enabled providers")
    lines.append("  /remote-provider enable <name>     Enable a provider")
    lines.append("  /remote-provider disable <name>    Disable a provider")
    lines.append("  /remote-provider weight <name> <value>  Set provider weight")
    lines.append("")

    return "\n".join(lines)


def run_remote_search(
    provider_name: str,
    query: str,
    config: LSMConfig,
    max_results: int = 5,
) -> str:
    """
    Run a test search on a specific remote provider.

    Args:
        provider_name: Name of the provider to search
        query: Search query
        config: LSM configuration
        max_results: Maximum results to return

    Returns:
        Formatted search results string
    """
    import time

    # Find provider config by name
    provider_config = None
    for pc in config.remote_providers or []:
        if pc.name.lower() == provider_name.lower():
            provider_config = pc
            break

    if not provider_config:
        return f"Provider not found: {provider_name}\nUse /remote-providers to see available providers.\n"

    lines = [
        "",
        f"Searching '{provider_name}' for: {query}",
        "-" * 60,
    ]

    try:
        start_time = time.time()

        # Create provider instance
        provider = create_remote_provider(
            provider_config.type,
            {
                "type": provider_config.type,
                "enabled": provider_config.enabled,
                "weight": provider_config.weight,
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

        # Execute search
        results = provider.search(query, max_results=max_results)
        elapsed = time.time() - start_time

        if not results:
            lines.append(f"No results found. ({elapsed:.2f}s)")
            lines.append("")
            return "\n".join(lines)

        lines.append(f"Found {len(results)} results in {elapsed:.2f}s:\n")

        for i, result in enumerate(results, 1):
            title = result.title or "(no title)"
            url = result.url or ""
            snippet = result.snippet or ""
            score = result.score

            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   URL: {url}")
            if score is not None:
                lines.append(f"   Score: {score:.3f}")
            if snippet:
                # Truncate long snippets
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                lines.append(f"   {snippet}")
            lines.append("")

    except Exception as e:
        logger.error(f"Remote search failed: {e}")
        lines.append(f"Search failed: {e}\n")

    return "\n".join(lines)


def run_remote_search_all(
    query: str,
    config: LSMConfig,
    state: SessionState,
) -> str:
    """
    Run a search across all enabled remote providers.

    Args:
        query: Search query
        config: LSM configuration
        state: Session state to store results

    Returns:
        Formatted search results string
    """
    import time

    active_providers = config.get_active_remote_providers()

    if not active_providers:
        return "No enabled remote providers configured.\nUse /remote-providers to see available providers.\n"

    lines = [
        "",
        f"Searching {len(active_providers)} providers for: {query}",
        "=" * 60,
    ]

    all_results = []
    start_time = time.time()

    for provider_config in active_providers:
        provider_name = provider_config.name
        lines.append(f"\n[{provider_name}]")

        try:
            # Create provider instance
            provider = create_remote_provider(
                provider_config.type,
                {
                    "type": provider_config.type,
                    "enabled": provider_config.enabled,
                    "weight": provider_config.weight,
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

            # Execute search
            max_results = provider_config.max_results or 5
            results = provider.search(query, max_results=max_results)

            if not results:
                lines.append("  No results")
                continue

            lines.append(f"  Found {len(results)} results")

            # Add to combined results with provider metadata
            for result in results:
                all_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "score": result.score,
                    "provider": provider_name,
                    "weight": provider_config.weight,
                    "weighted_score": (result.score or 0.5) * provider_config.weight,
                    "metadata": result.metadata,
                })

        except Exception as e:
            logger.error(f"Search failed for {provider_name}: {e}")
            lines.append(f"  Error: {e}")

    elapsed = time.time() - start_time

    # Sort by weighted score
    all_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique_results.append(result)

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total: {len(unique_results)} unique results in {elapsed:.2f}s")
    lines.append("=" * 60)

    # Display top results
    for i, result in enumerate(unique_results[:10], 1):
        title = result.get("title") or "(no title)"
        provider = result.get("provider", "")
        weighted_score = result.get("weighted_score", 0)
        url = result.get("url", "")

        lines.append(f"\n{i}. [{provider}] {title}")
        lines.append(f"   Weighted Score: {weighted_score:.3f}")
        if url:
            lines.append(f"   URL: {url}")

    lines.append("")

    # Store results in session state
    state.last_remote_sources = unique_results

    return "\n".join(lines)


def toggle_remote_provider(
    config: LSMConfig,
    provider_name: str,
    enabled: bool,
) -> bool:
    """
    Enable or disable a remote provider.

    Args:
        config: LSM configuration
        provider_name: Name of the provider
        enabled: True to enable, False to disable

    Returns:
        True if provider was found and updated, False otherwise
    """
    for provider_config in config.remote_providers or []:
        if provider_config.name.lower() == provider_name.lower():
            provider_config.enabled = enabled
            return True
    return False


def set_remote_provider_weight(
    config: LSMConfig,
    provider_name: str,
    weight: float,
) -> bool:
    """
    Set the weight for a remote provider.

    Args:
        config: LSM configuration
        provider_name: Name of the provider
        weight: New weight value (0.0-1.0)

    Returns:
        True if provider was found and updated, False otherwise
    """
    for provider_config in config.remote_providers or []:
        if provider_config.name.lower() == provider_name.lower():
            provider_config.weight = weight
            return True
    return False


# -----------------------------
# File Opening
# -----------------------------
def open_file(path: str) -> str:
    """
    Open file with system default application.

    Cross-platform: Windows, macOS, Linux.

    Args:
        path: Path to file to open

    Returns:
        Status message string
    """
    if not path or not os.path.exists(path):
        return f"File does not exist: {path}\n"

    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return f"Opened: {path}\n"
    except Exception as e:
        logger.error(f"Failed to open file: {e}")
        return f"Failed to open file: {e}\n"


# -----------------------------
# Cost Estimation
# -----------------------------
def estimate_synthesis_cost(
    provider,
    question: str,
    context: str,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    """Estimate cost for synthesis step."""
    input_tokens = estimate_tokens(f"{question}\n{context}")
    output_tokens = max_tokens or 0
    cost = provider.estimate_cost(input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def estimate_rerank_cost(
    provider,
    question: str,
    candidates: List[Candidate],
    k: int,
) -> Dict[str, Any]:
    """Estimate cost for reranking step."""
    combined = question + "\n" + "\n".join(c.text or "" for c in candidates[:k])
    input_tokens = estimate_tokens(combined)
    output_tokens = max(50, k * 30)
    cost = provider.estimate_cost(input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def estimate_query_cost(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> str:
    """
    Estimate costs for a query without invoking the LLM.

    Returns:
        Formatted cost estimate string
    """
    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local
    remote_policy = mode_config.source_policy.remote

    if not local_policy.enabled:
        if remote_policy.enabled:
            note = "Local sources disabled; cost estimate excludes remote retrieval.\n"
        else:
            note = ""
        query_config = config.llm.get_query_config()
        synthesis_provider = create_provider(query_config)
        synth_est = estimate_synthesis_cost(
            synthesis_provider,
            question,
            "",
            query_config.max_tokens,
        )
        lines = [note, "", "COST ESTIMATE"]
        if synth_est["cost"] is not None:
            lines.append(f"- Synthesis estimate: ${synth_est['cost']:.4f}")
        else:
            lines.append("- Synthesis estimate: unavailable for this provider")
        lines.append("")
        return "\n".join(lines)

    batch_size = config.batch_size
    k = local_policy.k
    k_rerank = local_policy.k_rerank
    no_rerank = config.query.no_rerank
    min_relevance = local_policy.min_relevance
    max_per_file = config.query.max_per_file
    local_pool = config.query.local_pool or max(k * 3, k_rerank * 4)

    query_vector = embed_text(embedder, question, batch_size=batch_size)

    filters_active = bool(state.path_contains) or bool(state.ext_allow) or bool(state.ext_deny)
    retrieve_k = config.query.retrieve_k or (max(k, k * 3) if filters_active else k)

    candidates = retrieve_candidates(collection, query_vector, retrieve_k)
    if not candidates:
        return "No results found in Chroma for this query.\n"

    filtered = filter_candidates(
        candidates,
        path_contains=state.path_contains,
        ext_allow=state.ext_allow,
        ext_deny=state.ext_deny,
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
        except Exception as e:
            logger.error(f"Failed to load pinned chunks for estimate: {e}")

    if not filtered:
        return "No results matched the configured filters.\n"

    rerank_strategy = config.query.rerank_strategy.lower()
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

    relevance = compute_relevance(filtered)
    if relevance < min_relevance:
        return "Relevance below threshold; no LLM call expected.\n"

    should_llm_rerank = rerank_strategy in ("llm", "hybrid") and not no_rerank
    chosen = filtered[: min(k_rerank, len(filtered))]

    total_estimated = 0.0
    cost_lines = []

    if should_llm_rerank:
        ranking_config = config.llm.get_ranking_config()
        rerank_provider = create_provider(ranking_config)
        rerank_est = estimate_rerank_cost(rerank_provider, question, chosen, k=min(k_rerank, len(chosen)))
        if rerank_est["cost"] is not None:
            total_estimated += rerank_est["cost"]
            cost_lines.append(f"Rerank estimate: ${rerank_est['cost']:.4f}")
        else:
            cost_lines.append("Rerank estimate: unavailable for this provider")

    context_block, _ = build_context_block(chosen)
    query_config = config.llm.get_query_config()
    synthesis_provider = create_provider(query_config)
    synth_est = estimate_synthesis_cost(
        synthesis_provider,
        question,
        context_block,
        query_config.max_tokens,
    )
    if synth_est["cost"] is not None:
        total_estimated += synth_est["cost"]
        cost_lines.append(f"Synthesis estimate: ${synth_est['cost']:.4f}")
    else:
        cost_lines.append("Synthesis estimate: unavailable for this provider")

    lines = ["", "COST ESTIMATE"]
    for line in cost_lines:
        lines.append(f"- {line}")
    if total_estimated > 0:
        lines.append(f"Total estimate: ${total_estimated:.4f}")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Command Result Types
# -----------------------------
class CommandResult:
    """Result from a command handler."""

    def __init__(
        self,
        output: str = "",
        handled: bool = True,
        should_exit: bool = False,
        action: Optional[str] = None,
        action_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize command result.

        Args:
            output: Text output to display
            handled: True if command was handled (not a query)
            should_exit: True if the user requested exit
            action: Optional action type (e.g., 'open_file', 'edit_note')
            action_data: Optional data for the action
        """
        self.output = output
        self.handled = handled
        self.should_exit = should_exit
        self.action = action
        self.action_data = action_data or {}


# -----------------------------
# Main Command Handler
# -----------------------------
def handle_command(
    line: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> CommandResult:
    """
    Handle query commands.

    Args:
        line: User input line
        state: Session state
        config: Global configuration
        embedder: SentenceTransformer model
        collection: ChromaDB collection

    Returns:
        CommandResult with output text and status flags
    """
    q = line.strip()
    ql = q.lower()

    # Exit
    if ql in {"/exit", "exit", "quit", "q"}:
        return CommandResult(output="", should_exit=True)

    # Help
    if ql in {"/help", "help", "?"}:
        return CommandResult(output=get_help())

    # Debug
    if ql == "/debug":
        return CommandResult(output=format_debug(state))

    # Cost summary / export
    if ql.startswith("/costs"):
        tracker = state.cost_tracker
        if not tracker:
            return CommandResult(output="Cost tracking is not initialized.\n")
        parts = q.split()
        if len(parts) == 1:
            return CommandResult(output=format_costs(state))
        if len(parts) >= 2 and parts[1].lower() == "export":
            export_path = None
            if len(parts) >= 3:
                export_path = Path(parts[2])
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                export_path = Path(f"costs-{timestamp}.csv")
            try:
                tracker.export_csv(export_path)
                return CommandResult(output=f"Cost data exported to: {export_path}\n")
            except Exception as e:
                return CommandResult(output=f"Failed to export costs: {e}\n")
        return CommandResult(output="Usage:\n  /costs\n  /costs export <path>\n")

    # Budget
    if ql.startswith("/budget"):
        tracker = state.cost_tracker
        if not tracker:
            return CommandResult(output="Cost tracking is not initialized.\n")
        parts = q.split()
        if len(parts) == 1:
            if tracker.budget_limit is None:
                return CommandResult(output="No budget set.\n")
            else:
                return CommandResult(output=f"Budget limit: ${tracker.budget_limit:.4f}\n")
        if len(parts) == 3 and parts[1].lower() == "set":
            try:
                tracker.budget_limit = float(parts[2])
                return CommandResult(output=f"Budget limit set to: ${tracker.budget_limit:.4f}\n")
            except ValueError:
                return CommandResult(output="Invalid budget amount. Use a numeric value.\n")
        return CommandResult(output="Usage:\n  /budget\n  /budget set <amount>\n")

    # Cost estimate (no LLM call)
    if ql.startswith("/cost-estimate"):
        parts = q.split(maxsplit=1)
        if len(parts) != 2:
            return CommandResult(output="Usage: /cost-estimate <query>\n")
        output = estimate_query_cost(parts[1].strip(), config, state, embedder, collection)
        return CommandResult(output=output)

    # Export citations
    if ql.startswith("/export-citations"):
        parts = q.split()
        fmt = "bibtex"
        note_path = None
        if len(parts) >= 2:
            fmt = parts[1].strip().lower()
        if len(parts) >= 3:
            note_path = parts[2].strip()

        if fmt not in {"bibtex", "zotero"}:
            return CommandResult(output="Format must be 'bibtex' or 'zotero'.\n")

        try:
            if note_path:
                output_path = export_citations_from_note(Path(note_path), fmt=fmt)
            else:
                if not state.last_label_to_candidate:
                    return CommandResult(output="No last query sources available to export.\n")
                sources = [
                    {
                        "source_path": c.source_path,
                        "source_name": c.source_name,
                        "chunk_index": c.chunk_index,
                        "ext": c.ext,
                        "label": label,
                        "title": (c.meta or {}).get("title"),
                        "author": (c.meta or {}).get("author"),
                        "mtime_ns": (c.meta or {}).get("mtime_ns"),
                        "ingested_at": (c.meta or {}).get("ingested_at"),
                    }
                    for label, c in state.last_label_to_candidate.items()
                ]
                output_path = export_citations_from_sources(sources, fmt=fmt)
            return CommandResult(output=f"Citations exported to: {output_path}\n")
        except Exception as e:
            return CommandResult(output=f"Failed to export citations: {e}\n")

    # List available models
    if ql.startswith("/models"):
        parts = q.split()
        provider_filter = parts[1].strip().lower() if len(parts) > 1 else None
        providers = config.llm.get_provider_names()
        if provider_filter:
            if provider_filter == "claude":
                providers = [p for p in providers if p in {"claude", "anthropic"}]
            else:
                providers = [p for p in providers if p.lower() == provider_filter]
            if not providers:
                return CommandResult(output=f"Provider not found in config: {provider_filter}\n")

        lines = []
        seen_labels = set()
        for provider_name in providers:
            label = display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            provider_config = config.llm.get_provider_by_name(provider_name)
            if not provider_config:
                continue
            try:
                test_config = provider_config.resolve_first_available()
                lines.append(f"{label}:")
                if not test_config:
                    lines.append("  (not configured for any feature)\n")
                    continue
                provider = create_provider(test_config)
                lines.append(format_models(state, provider))
                lines.append("")
            except Exception as e:
                logger.error(f"Failed to list models for {provider_name}: {e}")
                lines.append(f"  (failed to list models: {e})\n")
        return CommandResult(output="\n".join(lines))

    # List available providers
    if ql.strip() == "/providers":
        return CommandResult(output=format_providers(config))

    # Provider health status
    if ql.strip() == "/provider-status":
        return CommandResult(output=format_provider_status(config))

    if ql.strip() == "/vectordb-providers":
        return CommandResult(output=format_vectordb_providers(config))

    if ql.strip() == "/vectordb-status":
        return CommandResult(output=format_vectordb_status(config))

    # Remote providers listing
    if ql.strip() == "/remote-providers":
        return CommandResult(output=format_remote_providers(config))

    # Remote search on a specific provider
    if ql.startswith("/remote-search-all"):
        parts = q.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(output="Usage: /remote-search-all <query>\n")
        search_query = parts[1].strip()
        output = run_remote_search_all(search_query, config, state)
        return CommandResult(output=output)

    if ql.startswith("/remote-search"):
        parts = q.split(maxsplit=2)
        if len(parts) < 3:
            return CommandResult(output="Usage: /remote-search <provider> <query>\nExample: /remote-search wikipedia machine learning\n")
        provider_name = parts[1].strip()
        search_query = parts[2].strip()
        output = run_remote_search(provider_name, search_query, config)
        return CommandResult(output=output)

    # Remote provider enable/disable/weight
    if ql.startswith("/remote-provider"):
        parts = q.split()
        if len(parts) < 3:
            return CommandResult(output="Usage:\n  /remote-provider enable <name>\n  /remote-provider disable <name>\n  /remote-provider weight <name> <value>\n")

        action = parts[1].strip().lower()
        provider_name = parts[2].strip()

        if action == "enable":
            if toggle_remote_provider(config, provider_name, True):
                return CommandResult(output=f"Provider '{provider_name}' enabled.\n")
            else:
                return CommandResult(output=f"Provider not found: {provider_name}\n")

        if action == "disable":
            if toggle_remote_provider(config, provider_name, False):
                return CommandResult(output=f"Provider '{provider_name}' disabled.\n")
            else:
                return CommandResult(output=f"Provider not found: {provider_name}\n")

        if action == "weight":
            if len(parts) < 4:
                return CommandResult(output="Usage: /remote-provider weight <name> <value>\n")
            try:
                weight = float(parts[3].strip())
                if weight < 0.0:
                    return CommandResult(output="Weight must be non-negative.\n")
                if set_remote_provider_weight(config, provider_name, weight):
                    return CommandResult(output=f"Provider '{provider_name}' weight set to {weight:.2f}.\n")
                else:
                    return CommandResult(output=f"Provider not found: {provider_name}\n")
            except ValueError:
                return CommandResult(output="Invalid weight value. Use a numeric value.\n")

        return CommandResult(output="Unknown action. Use: enable, disable, weight\n")

    # Show/set current model
    if ql.startswith("/model"):
        parts = q.split()
        if len(parts) == 1:
            lines = []
            feature_configs = _get_feature_configs(config)
            for feature, cfg in feature_configs.items():
                if cfg is None:
                    continue
                label = format_feature_label(feature)
                provider = display_provider_name(cfg.provider)
                lines.append(f"{label}: {provider}/{cfg.model}")
            lines.append("")
            return CommandResult(output="\n".join(lines))

        if len(parts) != 4:
            return CommandResult(output="Usage:\n  /model                   (show current)\n  /model <task> <provider> <model>  (set model for a task)\n  /models [provider]       (list available models)\n")

        task = parts[1].strip().lower()
        provider_name = parts[2].strip()
        model_name = parts[3].strip()

        task_map = {
            "query": "query",
            "tag": "tagging",
            "tagging": "tagging",
            "rerank": "ranking",
            "ranking": "ranking",
        }
        feature = task_map.get(task)
        if not feature:
            return CommandResult(output="Unknown task. Use: query, tag, rerank\n")

        try:
            provider_names = config.llm.get_provider_names()
            normalized = provider_name
            if provider_name == "anthropic" and "claude" in provider_names:
                normalized = "claude"
            elif provider_name == "claude" and "anthropic" in provider_names:
                normalized = "anthropic"

            config.llm.set_feature_selection(feature, normalized, model_name)
            if feature == "query":
                state.model = model_name
            label = format_feature_label(feature)
            return CommandResult(output=f"Model set: {label} = {display_provider_name(normalized)}/{model_name}\n")
        except Exception as e:
            return CommandResult(output=f"Failed to set model: {e}\n")

    # Show/set current mode
    if ql.startswith("/mode"):
        parts = q.split()
        if len(parts) == 1:
            # Show current mode
            current_mode = config.query.mode
            mode_config = config.get_mode_config(current_mode)
            lines = [
                f"Current mode: {current_mode}",
                f"  Synthesis style: {mode_config.synthesis_style}",
                f"  Local sources: enabled (k={mode_config.source_policy.local.k})",
                f"  Remote sources: {'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}",
                f"  Model knowledge: {'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}",
                f"  Notes: {'enabled' if mode_config.notes.enabled else 'disabled'}",
                f"\nAvailable modes: {', '.join(config.modes.keys())}\n",
            ]
            return CommandResult(output="\n".join(lines))

        if len(parts) == 2 and parts[1].lower() == "set":
            return CommandResult(output="Usage:\n  /mode set <setting> <on|off>\nSettings: model_knowledge, remote, notes\n")

        if len(parts) >= 2 and parts[1].lower() == "set":
            if len(parts) != 4:
                return CommandResult(output="Usage:\n  /mode set <setting> <on|off>\nSettings: model_knowledge, remote, notes\n")

            setting = parts[2].strip().lower()
            value = parts[3].strip().lower()
            enabled = value in {"on", "true", "yes", "1"}

            mode_config = config.get_mode_config()
            if setting in {"model_knowledge", "model-knowledge"}:
                mode_config.source_policy.model_knowledge.enabled = enabled
            elif setting in {"remote", "remote_sources", "remote-sources"}:
                mode_config.source_policy.remote.enabled = enabled
            elif setting in {"notes"}:
                mode_config.notes.enabled = enabled
            else:
                return CommandResult(output=f"Unknown setting: {setting}\nSettings: model_knowledge, remote, notes\n")

            return CommandResult(output=f"Mode setting '{setting}' set to: {'on' if enabled else 'off'}\n")

        if len(parts) != 2:
            return CommandResult(output="Usage:\n  /mode           (show current)\n  /mode <name>    (switch to a different mode)\n")

        new_mode = parts[1].strip()

        # Validate mode exists
        if new_mode not in config.modes:
            return CommandResult(output=f"Mode not found: {new_mode}\nAvailable modes: {', '.join(config.modes.keys())}\n")

        # Switch mode
        config.query.mode = new_mode
        mode_config = config.get_mode_config(new_mode)
        lines = [
            f"Mode switched to: {new_mode}",
            f"  Synthesis style: {mode_config.synthesis_style}",
            f"  Remote sources: {'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}",
            f"  Model knowledge: {'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}\n",
        ]
        return CommandResult(output="\n".join(lines))

    # Save note from last query
    if ql.startswith("/note") or ql.startswith("/notes"):
        if not state.last_question:
            return CommandResult(output="No query to save. Run a query first.\n")

        try:
            # Get mode config for notes directory
            mode_config = config.get_mode_config()
            notes_config = mode_config.notes

            # Resolve notes directory
            if config.config_path:
                base_dir = config.config_path.parent
                notes_dir = base_dir / notes_config.dir
            else:
                notes_dir = Path(notes_config.dir)

            # Generate note content
            content = generate_note_content(
                query=state.last_question,
                answer=state.last_answer or "No answer generated",
                local_sources=state.last_local_sources_for_notes,
                remote_sources=state.last_remote_sources,
                mode=config.query.mode,
                use_wikilinks=notes_config.wikilinks,
                include_backlinks=notes_config.backlinks,
                include_tags=notes_config.include_tags,
            )

            # Determine filename
            parts_list = q.split(maxsplit=1)
            filename_override = parts_list[1].strip() if len(parts_list) > 1 else None
            if filename_override:
                filename = filename_override
                if not filename.lower().endswith(".md"):
                    filename += ".md"
                note_path = Path(filename)
                if not note_path.is_absolute():
                    note_path = notes_dir / note_path
            else:
                filename = get_note_filename(state.last_question, format=notes_config.filename_format)
                note_path = notes_dir / filename

            # Return action for UI to handle editor opening
            return CommandResult(
                output="\nOpening note in editor...\nEdit the note and save/close the editor to continue.\n",
                action="edit_note",
                action_data={
                    "content": content,
                    "note_path": str(note_path),
                    "notes_dir": str(notes_dir),
                },
            )

        except Exception as e:
            logger.error(f"Note save error: {e}")
            return CommandResult(output=f"Failed to save note: {e}\n")

    # Load document for context pinning
    if ql.startswith("/load"):
        parts = q.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(output="Usage: /load <file_path>\nExample: /load /docs/important.md\n\nThis pins a document for forced inclusion in next query context.\nUse /load clear to clear pinned chunks.\n")

        arg = parts[1].strip()

        # Handle /load clear
        if arg.lower() == "clear":
            state.pinned_chunks = []
            return CommandResult(output="Cleared all pinned chunks.\n")

        file_path = arg

        lines = [f"Loading chunks from: {file_path}", "Searching collection..."]

        try:
            chroma = require_chroma_collection(collection, "/load")
            # Search for chunks from this file
            results = chroma.get(
                where={"source_path": {"$eq": file_path}},
                include=["metadatas"],
            )

            if not results or not results.get("ids"):
                lines.append(f"\nNo chunks found for path: {file_path}")
                lines.append("Tip: Path must match exactly. Use /explore to find exact paths.\n")
                return CommandResult(output="\n".join(lines))

            chunk_ids = results["ids"]

            # Add to pinned chunks
            for chunk_id in chunk_ids:
                if chunk_id not in state.pinned_chunks:
                    state.pinned_chunks.append(chunk_id)

            lines.append(f"\nPinned {len(chunk_ids)} chunks from {file_path}")
            lines.append(f"Total pinned chunks: {len(state.pinned_chunks)}")
            lines.append("\nThese chunks will be forcibly included in your next query.")
            lines.append("Use /load clear to unpin all chunks.\n")

        except Exception as e:
            lines.append(f"Error loading chunks: {e}\n")
            logger.error(f"Load command error: {e}")

        return CommandResult(output="\n".join(lines))

    # Show / Expand
    if ql.startswith("/show") or ql.startswith("/expand"):
        parts = q.split()
        if len(parts) != 2:
            usage = (
                "/show S#   (e.g., /show S2)"
                if ql.startswith("/show")
                else "/expand S#   (e.g., /expand S2)"
            )
            return CommandResult(output=f"Usage: {usage}\n")

        label = parts[1].strip().upper()
        candidate = state.last_label_to_candidate.get(label)
        if not candidate:
            return CommandResult(output=f"No such label in last results: {label}\n")

        output = format_source_chunk(label, candidate, expanded=ql.startswith("/expand"))
        return CommandResult(output=output)

    # Open
    if ql.startswith("/open"):
        parts = q.split()
        if len(parts) != 2:
            return CommandResult(output="Usage: /open S#   (e.g., /open S2)\n")

        label = parts[1].strip().upper()
        candidate = state.last_label_to_candidate.get(label)
        if not candidate:
            return CommandResult(output=f"No such label in last results: {label}\n")

        path = (candidate.meta or {}).get("source_path")
        if not path:
            return CommandResult(output="No source_path available for this citation.\n")

        # Return action for UI to handle file opening
        return CommandResult(
            output="",
            action="open_file",
            action_data={"path": path},
        )

    # Set filters
    if ql.startswith("/set"):
        parts = q.split()
        if len(parts) < 3:
            return CommandResult(output="Usage:\n  /set path_contains <substring> [more...]\n  /set ext_allow .md .pdf\n  /set ext_deny .txt\n")

        key = parts[1]
        values = parts[2:]

        if key == "path_contains":
            state.path_contains = values if len(values) > 1 else values[0]
            return CommandResult(output=f"path_contains set to: {state.path_contains}\n")

        if key == "ext_allow":
            state.ext_allow = values
            return CommandResult(output=f"ext_allow set to: {state.ext_allow}\n")

        if key == "ext_deny":
            state.ext_deny = values
            return CommandResult(output=f"ext_deny set to: {state.ext_deny}\n")

        return CommandResult(output=f"Unknown filter key: {key}\n")

    # Clear filters
    if ql.startswith("/clear"):
        parts = q.split()
        if len(parts) != 2:
            return CommandResult(output="Usage: /clear path_contains|ext_allow|ext_deny\n")

        key = parts[1]
        if key == "path_contains":
            state.path_contains = None
            return CommandResult(output="path_contains cleared.\n")
        if key == "ext_allow":
            state.ext_allow = None
            return CommandResult(output="ext_allow cleared.\n")
        if key == "ext_deny":
            state.ext_deny = None
            return CommandResult(output="ext_deny cleared.\n")

        return CommandResult(output=f"Unknown filter key: {key}\n")

    if q.startswith("/"):
        return CommandResult(output=get_help())

    # Not a command - should be treated as a query
    return CommandResult(output="", handled=False)


# -----------------------------
# Backwards Compatibility
# -----------------------------
def print_models(state: SessionState, provider) -> None:
    """Print available models. DEPRECATED: Use format_models() instead."""
    print(format_models(state, provider))


def print_providers(config: LSMConfig) -> None:
    """Print available LLM providers. DEPRECATED: Use format_providers() instead."""
    print(format_providers(config))


def print_provider_status(config: LSMConfig) -> None:
    """Print provider health status. DEPRECATED: Use format_provider_status() instead."""
    print(format_provider_status(config))


def print_vectordb_providers(config: LSMConfig) -> None:
    """Print vector DB providers. DEPRECATED: Use format_vectordb_providers() instead."""
    print(format_vectordb_providers(config))


def print_vectordb_status(config: LSMConfig) -> None:
    """Print vector DB status. DEPRECATED: Use format_vectordb_status() instead."""
    print(format_vectordb_status(config))


def print_remote_providers(config: LSMConfig) -> None:
    """Print remote providers. DEPRECATED: Use format_remote_providers() instead."""
    print(format_remote_providers(config))
