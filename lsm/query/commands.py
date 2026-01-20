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
from typing import List, Dict, Any, Optional, Callable

from lsm.config.models import LSMConfig
from lsm.logging import get_logger
from lsm.providers import create_provider
from lsm.vectordb.utils import require_chroma_collection
from lsm.query.session import Candidate, SessionState
from lsm.query.retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from lsm.query.rerank import apply_local_reranking
from lsm.query.synthesis import build_context_block
from lsm.remote import create_remote_provider
from lsm.query.notes import get_note_filename, generate_note_content
from lsm.query.cost_tracking import estimate_tokens
from lsm.query.citations import export_citations_from_note, export_citations_from_sources

logger = get_logger(__name__)

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


def get_help() -> str:
    """
    Get the query mode help text.

    Returns:
        Help text string
    """
    lines = [
        "Enter a question to query your local knowledge base.",
        "Commands:",
        "  /exit           Quit",
        "  /help           Show this help",
        "  /show S#        Show the cited chunk (e.g., /show S2)",
        "  /expand S#      Show full chunk text (no truncation)",
        "  /open S#        Open the source file in default app",
        "  /models [provider]   List available models (optionally for one provider)",
        "  /model               Show current models for tasks",
        "  /model <task> <provider> <model>   Set model for a task",
        "  /providers      List available LLM providers",
        "  /provider-status Show provider health and recent stats",
        "  /vectordb-providers List available vector DB providers",
        "  /vectordb-status Show vector DB provider status",
        "  /remote-providers    List available remote source providers",
        "  /remote-search <provider> <query>  Test a remote provider",
        "  /remote-search-all <query>  Search all enabled providers",
        "  /remote-provider enable|disable|weight <name> [value]",
        "  /mode           Show current query mode",
        "  /mode <name>    Switch to a different query mode",
        "  /mode set <setting> <on|off>  Toggle mode settings (model_knowledge, remote, notes)",
        "  /note           Save last query as an editable note",
        "  /note <name>    Save last query note with custom filename",
        "  /notes          Alias for /note",
        "  /load <path>    Pin a document for forced context inclusion",
        "  /costs          Show session cost summary",
        "  /costs export <path>  Export cost data to CSV",
        "  /budget set <amount>  Set a session budget limit",
        "  /cost-estimate <query>  Estimate cost for a query without running it",
        "  /export-citations [format] [note_path]  Export citations (bibtex|zotero)",
        "  /debug          Print retrieval diagnostics for the last query",
        "  /set <filter>   Set session filters (path/ext)",
        "  /clear <filter> Clear session filters",
        "",
    ]
    return "\n".join(lines)


def display_provider_name(name: str) -> str:
    """
    Normalize provider name for display.

    Args:
        name: Raw provider name

    Returns:
        Display-friendly provider name
    """
    if name in {"anthropic", "claude"}:
        return "claude"
    return name


def format_feature_label(feature: str) -> str:
    """
    Format a feature name for display.

    Args:
        feature: Raw feature name

    Returns:
        Short label for display
    """
    return {
        "query": "query",
        "tagging": "tag",
        "ranking": "rerank",
    }.get(feature, feature)


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
# Command Handlers
# -----------------------------
def handle_exit_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if ql in {"/exit", "exit", "quit", "q"}:
        return CommandResult(output="", should_exit=True)
    return None


def handle_help_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if ql in {"/help", "help", "?"}:
        return CommandResult(output=get_help())
    return None


def handle_debug_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if ql == "/debug":
        return CommandResult(output=state.format_debug())
    return None


def handle_costs_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/costs"):
        return None
    tracker = state.cost_tracker
    if not tracker:
        return CommandResult(output="Cost tracking is not initialized.\n")
    parts = q.split()
    if len(parts) == 1:
        return CommandResult(output=state.format_costs())
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


def handle_budget_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/budget"):
        return None
    tracker = state.cost_tracker
    if not tracker:
        return CommandResult(output="Cost tracking is not initialized.\n")
    parts = q.split()
    if len(parts) == 1:
        if tracker.budget_limit is None:
            return CommandResult(output="No budget set.\n")
        return CommandResult(output=f"Budget limit: ${tracker.budget_limit:.4f}\n")
    if len(parts) == 3 and parts[1].lower() == "set":
        try:
            tracker.budget_limit = float(parts[2])
            return CommandResult(output=f"Budget limit set to: ${tracker.budget_limit:.4f}\n")
        except ValueError:
            return CommandResult(output="Invalid budget amount. Use a numeric value.\n")
    return CommandResult(output="Usage:\n  /budget\n  /budget set <amount>\n")


def handle_cost_estimate_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/cost-estimate"):
        return None
    parts = q.split(maxsplit=1)
    if len(parts) != 2:
        return CommandResult(output="Usage: /cost-estimate <query>\n")
    output = estimate_query_cost(parts[1].strip(), config, state, embedder, collection)
    return CommandResult(output=output)


def handle_export_citations_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/export-citations"):
        return None
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


def handle_remote_search_all_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/remote-search-all"):
        return None
    parts = q.split(maxsplit=1)
    if len(parts) < 2:
        return CommandResult(output="Usage: /remote-search-all <query>\n")
    search_query = parts[1].strip()
    output = run_remote_search_all(search_query, config, state)
    return CommandResult(output=output)


def handle_remote_search_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/remote-search"):
        return None
    parts = q.split(maxsplit=2)
    if len(parts) < 3:
        return CommandResult(
            output=(
                "Usage: /remote-search <provider> <query>\n"
                "Example: /remote-search wikipedia machine learning\n"
            )
        )
    provider_name = parts[1].strip()
    search_query = parts[2].strip()
    output = run_remote_search(provider_name, search_query, config)
    return CommandResult(output=output)


def handle_remote_provider_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/remote-provider"):
        return None
    parts = q.split()
    if len(parts) < 3:
        return CommandResult(
            output=(
                "Usage:\n"
                "  /remote-provider enable <name>\n"
                "  /remote-provider disable <name>\n"
                "  /remote-provider weight <name> <value>\n"
            )
        )

    action = parts[1].strip().lower()
    provider_name = parts[2].strip()

    if action == "enable":
        if toggle_remote_provider(config, provider_name, True):
            return CommandResult(output=f"Provider '{provider_name}' enabled.\n")
        return CommandResult(output=f"Provider not found: {provider_name}\n")

    if action == "disable":
        if toggle_remote_provider(config, provider_name, False):
            return CommandResult(output=f"Provider '{provider_name}' disabled.\n")
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
            return CommandResult(output=f"Provider not found: {provider_name}\n")
        except ValueError:
            return CommandResult(output="Invalid weight value. Use a numeric value.\n")

    return CommandResult(output="Unknown action. Use: enable, disable, weight\n")


def handle_model_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/model"):
        return None
    parts = q.split()
    if len(parts) == 1:
        return None

    if len(parts) != 4:
        return CommandResult(
            output=(
                "Usage:\n"
                "  /model                   (show current)\n"
                "  /model <task> <provider> <model>  (set model for a task)\n"
                "  /models [provider]       (list available models)\n"
            )
        )

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
        return CommandResult(
            output=f"Model set: {label} = {display_provider_name(normalized)}/{model_name}\n"
        )
    except Exception as e:
        return CommandResult(output=f"Failed to set model: {e}\n")


def handle_mode_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/mode"):
        return None
    parts = q.split()
    if len(parts) == 1:
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
        return CommandResult(
            output="Usage:\n  /mode set <setting> <on|off>\nSettings: model_knowledge, remote, notes\n"
        )

    if len(parts) >= 2 and parts[1].lower() == "set":
        if len(parts) != 4:
            return CommandResult(
                output="Usage:\n  /mode set <setting> <on|off>\nSettings: model_knowledge, remote, notes\n"
            )

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
            return CommandResult(
                output=f"Unknown setting: {setting}\nSettings: model_knowledge, remote, notes\n"
            )

        return CommandResult(output=f"Mode setting '{setting}' set to: {'on' if enabled else 'off'}\n")

    if len(parts) != 2:
        return CommandResult(output="Usage:\n  /mode           (show current)\n  /mode <name>    (switch to a different mode)\n")

    new_mode = parts[1].strip()
    if new_mode not in config.modes:
        return CommandResult(
            output=f"Mode not found: {new_mode}\nAvailable modes: {', '.join(config.modes.keys())}\n"
        )

    config.query.mode = new_mode
    mode_config = config.get_mode_config(new_mode)
    lines = [
        f"Mode switched to: {new_mode}",
        f"  Synthesis style: {mode_config.synthesis_style}",
        f"  Remote sources: {'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}",
        f"  Model knowledge: {'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}\n",
    ]
    return CommandResult(output="\n".join(lines))


def handle_note_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not (ql.startswith("/note") or ql.startswith("/notes")):
        return None
    if not state.last_question:
        return CommandResult(output="No query to save. Run a query first.\n")

    try:
        mode_config = config.get_mode_config()
        notes_config = mode_config.notes

        if config.config_path:
            base_dir = config.config_path.parent
            notes_dir = base_dir / notes_config.dir
        else:
            notes_dir = Path(notes_config.dir)

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


def handle_load_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/load"):
        return None
    parts = q.split(maxsplit=1)
    if len(parts) < 2:
        return CommandResult(
            output=(
                "Usage: /load <file_path>\n"
                "Example: /load /docs/important.md\n\n"
                "This pins a document for forced inclusion in next query context.\n"
                "Use /load clear to clear pinned chunks.\n"
            )
        )

    arg = parts[1].strip()
    if arg.lower() == "clear":
        state.pinned_chunks = []
        return CommandResult(output="Cleared all pinned chunks.\n")

    file_path = arg
    lines = [f"Loading chunks from: {file_path}", "Searching collection..."]

    try:
        chroma = require_chroma_collection(collection, "/load")
        results = chroma.get(
            where={"source_path": {"$eq": file_path}},
            include=["metadatas"],
        )

        if not results or not results.get("ids"):
            lines.append(f"\nNo chunks found for path: {file_path}")
            lines.append("Tip: Path must match exactly. Use /explore to find exact paths.\n")
            return CommandResult(output="\n".join(lines))

        chunk_ids = results["ids"]
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


def handle_show_expand_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not (ql.startswith("/show") or ql.startswith("/expand")):
        return None
    parts = q.split()
    if len(parts) != 2:
        usage = "/show S#   (e.g., /show S2)" if ql.startswith("/show") else "/expand S#   (e.g., /expand S2)"
        return CommandResult(output=f"Usage: {usage}\n")

    label = parts[1].strip().upper()
    candidate = state.last_label_to_candidate.get(label)
    if not candidate:
        return CommandResult(output=f"No such label in last results: {label}\n")

    output = candidate.format(label=label, expanded=ql.startswith("/expand"))
    return CommandResult(output=output)


def handle_open_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/open"):
        return None
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

    return CommandResult(
        output="",
        action="open_file",
        action_data={"path": path},
    )


def handle_set_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/set"):
        return None
    parts = q.split()
    if len(parts) < 3:
        return CommandResult(
            output="Usage:\n  /set path_contains <substring> [more...]\n  /set ext_allow .md .pdf\n  /set ext_deny .txt\n"
        )

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


def handle_clear_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if not ql.startswith("/clear"):
        return None
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


def handle_unknown_slash_command(
    q: str,
    ql: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> Optional[CommandResult]:
    if q.startswith("/"):
        return CommandResult(output=get_help())
    return None


def get_command_handlers() -> List[Callable[..., Optional[CommandResult]]]:
    return [
        handle_exit_command,
        handle_help_command,
        handle_debug_command,
        handle_costs_command,
        handle_budget_command,
        handle_cost_estimate_command,
        handle_export_citations_command,
        handle_remote_search_all_command,
        handle_remote_search_command,
        handle_remote_provider_command,
        handle_model_command,
        handle_mode_command,
        handle_note_command,
        handle_load_command,
        handle_show_expand_command,
        handle_open_command,
        handle_set_command,
        handle_clear_command,
        handle_unknown_slash_command,
    ]


