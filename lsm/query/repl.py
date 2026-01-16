"""
REPL (Read-Eval-Print Loop) for interactive query sessions.

Provides interactive commands, display utilities, and query orchestration.
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
from lsm.cli.logging import get_logger
from lsm.providers import create_provider
from lsm.vectordb.utils import require_chroma_collection
from lsm.vectordb import create_vectordb_provider, list_available_providers
from .session import Candidate, SessionState
from .retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from .rerank import apply_local_reranking
from .synthesis import build_context_block, fallback_answer, format_source_list
from .remote import create_remote_provider
from .notes import write_note, generate_note_content, edit_note_in_editor
from .cost_tracking import CostTracker, estimate_tokens, estimate_output_tokens
from .citations import export_citations_from_note, export_citations_from_sources

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


# -----------------------------
# Display Utilities
# -----------------------------
def print_banner() -> None:
    """Print REPL welcome banner."""
    print("Interactive query mode. Type your question and press Enter.")
    print("Commands: /exit, /help, /show S#, /expand S#, /open S#, /debug, /model, /models, /providers, /provider-status, /vectordb-providers, /vectordb-status, /mode, /note, /notes, /load, /set, /clear, /costs, /budget, /cost-estimate, /export-citations\n")


def print_help() -> None:
    """Print REPL help text."""
    print("Enter a question to query your local knowledge base.")
    print("Commands:")
    print("  /exit           Quit")
    print("  /help           Show this help")
    print("  /show S#        Show the cited chunk (e.g., /show S2)")
    print("  /expand S#      Show full chunk text (no truncation)")
    print("  /open S#        Open the source file in default app")
    print("  /models [provider]   List available models (optionally for one provider)")
    print("  /model               Show current models for tasks")
    print("  /model <task> <provider> <model>   Set model for a task")
    print("  /providers      List available LLM providers")
    print("  /provider-status Show provider health and recent stats")
    print("  /vectordb-providers List available vector DB providers")
    print("  /vectordb-status Show vector DB provider status")
    print("  /mode           Show current query mode")
    print("  /mode <name>    Switch to a different query mode")
    print("  /mode set <setting> <on|off>  Toggle mode settings (model_knowledge, remote, notes)")
    print("  /note           Save last query as an editable note")
    print("  /note <name>    Save last query note with custom filename")
    print("  /notes          Alias for /note")
    print("  /load <path>    Pin a document for forced context inclusion")
    print("  /costs          Show session cost summary")
    print("  /costs export <path>  Export cost data to CSV")
    print("  /budget set <amount>  Set a session budget limit")
    print("  /cost-estimate <query>  Estimate cost for a query without running it")
    print("  /export-citations [format] [note_path]  Export citations (bibtex|zotero)")
    print("  /debug          Print retrieval diagnostics for the last query")
    print("  /set …          Set session filters (path/ext)")
    print("  /clear …        Clear session filters\n")


def print_source_chunk(
    label: str,
    candidate: Candidate,
    expanded: bool = False,
) -> None:
    """
    Print a single source chunk.

    Args:
        label: Citation label (e.g., "S1")
        candidate: Candidate to display
        expanded: If True, show full text without truncation
    """
    meta = candidate.meta or {}
    source_path = meta.get("source_path", "unknown")
    chunk_index = meta.get("chunk_index", "NA")
    distance = candidate.distance

    if expanded:
        print(f"\n{label} — {source_path}")
        print(f"chunk_index={chunk_index}, distance={distance}")
        print("=" * 80)
        print((candidate.text or "").strip())
        print("=" * 80 + "\n")
    else:
        print(f"\n{label} — {source_path} (chunk_index={chunk_index}, distance={distance})")
        print("-" * 80)
        print((candidate.text or "").strip())
        print("-" * 80 + "\n")


def print_debug(state: SessionState) -> None:
    """
    Print debug information from last query.

    Args:
        state: Session state with debug artifacts
    """
    if not state.last_debug:
        print("No debug info yet. Ask a question first.\n")
        return

    print("\nDebug (last query):")
    for key, value in state.last_debug.items():
        print(f"- {key}: {value}")

    print("\nTop candidates (post-filter):")
    max_display = min(10, len(state.last_filtered_candidates))
    for i, c in enumerate(state.last_filtered_candidates[:max_display], start=1):
        meta = c.meta or {}
        source_path = meta.get("source_path", "unknown")
        source_name = meta.get("source_name") or Path(source_path).name
        chunk_index = meta.get("chunk_index", "NA")
        print(f"  {i:02d}. {source_name} (chunk_index={chunk_index}, distance={c.distance})")
    print()


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


def _display_provider_name(name: str) -> str:
    if name in {"anthropic", "claude"}:
        return "claude"
    return name


def _format_feature_label(feature: str) -> str:
    return {
        "query": "query",
        "tagging": "tag",
        "ranking": "rerank",
    }.get(feature, feature)


def print_models(state: SessionState, provider) -> None:
    """
    Print available models.

    Args:
        state: Session state (to cache model list)
        provider: LLM provider instance
    """
    try:
        ids = list_models(provider)
        state.available_models = ids

        if not ids:
            print("  (no models returned or listing unsupported)")
            return

        for model_id in ids:
            print(f"  - {model_id}")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        print(f"Failed to list models: {e}\n")


def _get_feature_configs(config: LSMConfig) -> dict[str, Optional[LLMConfig]]:
    feature_map = config.llm.get_feature_provider_map()
    return {
        "query": config.llm.get_query_config() if "query" in feature_map else None,
        "tagging": config.llm.get_tagging_config() if "tagging" in feature_map else None,
        "ranking": config.llm.get_ranking_config() if "ranking" in feature_map else None,
    }


def print_providers(config: LSMConfig) -> None:
    """
    Print available LLM providers.

    Args:
        config: LSM configuration
    """
    print()
    print("=" * 60)
    print("AVAILABLE LLM PROVIDERS")
    print("=" * 60)
    print()

    providers = config.llm.get_provider_names()

    if not providers:
        print("No providers configured.")
        print()
        return

    print("Selections:")
    feature_configs = _get_feature_configs(config)
    for feature, cfg in feature_configs.items():
        if cfg is None:
            continue
        label = _format_feature_label(feature)
        provider = _display_provider_name(cfg.provider)
        print(f"  {label:7s} {provider}/{cfg.model}")
    print()

    print(f"Providers ({len(providers)}):")
    print()

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

            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            print(f"  {label:20s} {is_available:30s}")

        except Exception as e:
            logger.debug(f"Error checking provider {provider_name}: {e}")
            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            print(f"  {label:20s} {'- (error)':30s}")

    print()
    print("To switch providers, update your config.json:")
    print('  "llms": [ { "provider_name": "provider_name", ... } ]')
    print()

# -----------------------------
# Cost Tracking Utilities
# -----------------------------
def print_costs(state: SessionState) -> None:
    """Print current session cost summary."""
    tracker = state.cost_tracker
    if not tracker:
        print("Cost tracking is not initialized.\n")
        return
    if not tracker.entries:
        print("No costs recorded for this session.\n")
        return
    print()
    print(tracker.format_summary())
    print()


def stream_output(chunks) -> str:
    """
    Stream chunks to stdout and return combined text.

    Prints a simple typing indicator before output.
    """
    print("\nTyping...")
    parts: List[str] = []
    for chunk in chunks:
        if chunk:
            parts.append(chunk)
            print(chunk, end="", flush=True)
    print()
    return "".join(parts).strip()


def estimate_synthesis_cost(
    provider,
    question: str,
    context: str,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
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
    combined = question + "\n" + "\n".join(c.text or "" for c in candidates[:k])
    input_tokens = estimate_tokens(combined)
    output_tokens = max(50, k * 30)
    cost = provider.estimate_cost(input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }

def print_provider_status(config: LSMConfig) -> None:
    """
    Print provider health status and call statistics.

    Args:
        config: LSM configuration
    """
    print()
    print("=" * 60)
    print("PROVIDER HEALTH STATUS")
    print("=" * 60)
    print()

    providers = config.llm.get_provider_names()
    if not providers:
        print("No providers registered.")
        print()
        return

    current_provider = config.llm.get_query_config().provider

    seen_labels = set()
    for provider_name in providers:
        try:
            provider_config = config.llm.get_provider_by_name(provider_name)
            test_config = provider_config.resolve_first_available() if provider_config else None
            if not test_config:
                status = "not_configured" if not provider_config else "missing_config"
                label = _display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                print(f"{label:20s} status={status:12s}")
                continue

            provider = create_provider(test_config)
            health = provider.health_check()

            status = health.get("status", "unknown")
            stats = health.get("stats", {})
            success = stats.get("success_count", 0)
            failure = stats.get("failure_count", 0)
            last_error = stats.get("last_error")
            current_label = " (current)" if provider_name == current_provider else ""
            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)

            print(
                f"{label:20s} status={status:12s} success={success:4d} failure={failure:4d}{current_label}"
            )
            if last_error:
                print(f"{'':20s} last_error={last_error}")
        except Exception as e:
            logger.debug(f"Error checking provider status {provider_name}: {e}")
            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            print(f"{label:20s} status=error        error={e}")

    print()


def print_vectordb_providers(config: LSMConfig) -> None:
    """Print available vector DB providers."""
    providers = list_available_providers()

    print()
    print("=" * 60)
    print("AVAILABLE VECTOR DB PROVIDERS")
    print("=" * 60)
    print()

    if not providers:
        print("No vector DB providers registered.")
        print()
        return

    current_provider = config.vectordb.provider
    print(f"Current Provider: {current_provider}")
    print(f"Collection:       {config.vectordb.collection}")
    print()

    print(f"Available Providers ({len(providers)}):")
    print()

    for provider_name in providers:
        is_current = "ACTIVE" if provider_name == current_provider else ""
        status = ""
        if provider_name == current_provider:
            try:
                provider = create_vectordb_provider(config.vectordb)
                status = "ok" if provider.is_available() else "unavailable"
            except Exception as e:
                status = f"error ({e})"
        print(f"  {provider_name:20s} {status:20s} {is_current}")

    print()
    print("To switch providers, update your config.json:")
    print('  "vectordb": { "provider": "provider_name", ... }')
    print()


def print_vectordb_status(config: LSMConfig) -> None:
    """Print vector DB provider health and stats."""
    print()
    print("=" * 60)
    print("VECTOR DB STATUS")
    print("=" * 60)
    print()

    try:
        provider = create_vectordb_provider(config.vectordb)
        health = provider.health_check()
        stats = provider.get_stats()

        print(f"Provider: {health.get('provider', 'unknown')}")
        print(f"Status:   {health.get('status', 'unknown')}")
        if health.get("error"):
            print(f"Error:    {health.get('error')}")
        print(f"Count:    {stats.get('count', 'n/a')}")
    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=" * 60)
    print("PROVIDER HEALTH STATUS")
    print("=" * 60)
    print()

    providers = config.llm.get_provider_names()
    if not providers:
        print("No providers registered.")
        print()
        return

    current_provider = config.llm.get_query_config().provider

    seen_labels = set()
    for provider_name in providers:
        try:
            provider_config = config.llm.get_provider_by_name(provider_name)
            test_config = provider_config.resolve_first_available() if provider_config else None
            if not test_config:
                status = "not_configured" if not provider_config else "missing_config"
                label = _display_provider_name(provider_name)
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                print(f"{label:20s} status={status:12s}")
                continue

            provider = create_provider(test_config)
            health = provider.health_check()

            status = health.get("status", "unknown")
            stats = health.get("stats", {})
            success = stats.get("success_count", 0)
            failure = stats.get("failure_count", 0)
            last_error = stats.get("last_error")
            current_label = " (current)" if provider_name == current_provider else ""
            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)

            print(
                f"{label:20s} status={status:12s} "
                f"success={success:4d} failure={failure:4d}{current_label}"
            )
            if last_error:
                print(f"{'':20s} last_error={last_error}")
        except Exception as e:
            logger.debug(f"Error checking provider status {provider_name}: {e}")
            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            print(f"{label:20s} status=error        error={e}")

    print()

# -----------------------------
# File Opening
# -----------------------------
def open_file(path: str) -> None:
    """
    Open file with system default application.

    Cross-platform: Windows, macOS, Linux.

    Args:
        path: Path to file to open
    """
    if not path or not os.path.exists(path):
        print(f"File does not exist: {path}\n")
        return

    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        logger.error(f"Failed to open file: {e}")
        print(f"Failed to open file: {e}\n")


# -----------------------------
# Command Handler
# -----------------------------
def handle_command(
    line: str,
    state: SessionState,
    config: LSMConfig,
    embedder,
    collection,
) -> bool:
    """
    Handle REPL commands.

    Args:
        line: User input line
        state: Session state
        config: Global configuration
        embedder: SentenceTransformer model
        collection: ChromaDB collection

    Returns:
        True if command was handled, False if input should be treated as question

    Raises:
        SystemExit: If user requests exit
    """
    q = line.strip()
    ql = q.lower()

    # Exit
    if ql in {"/exit", "exit", "quit", "q"}:
        raise SystemExit

    # Help
    if ql in {"/help", "help", "?"}:
        print_help()
        return True

    # Debug
    if ql == "/debug":
        print_debug(state)
        return True

    # Cost summary / export
    if ql.startswith("/costs"):
        tracker = state.cost_tracker
        if not tracker:
            print("Cost tracking is not initialized.\n")
            return True
        parts = q.split()
        if len(parts) == 1:
            print_costs(state)
            return True
        if len(parts) >= 2 and parts[1].lower() == "export":
            export_path = None
            if len(parts) >= 3:
                export_path = Path(parts[2])
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                export_path = Path(f"costs-{timestamp}.csv")
            try:
                tracker.export_csv(export_path)
                print(f"Cost data exported to: {export_path}\n")
            except Exception as e:
                print(f"Failed to export costs: {e}\n")
            return True
        print("Usage:")
        print("  /costs")
        print("  /costs export <path>\n")
        return True

    # Budget
    if ql.startswith("/budget"):
        tracker = state.cost_tracker
        if not tracker:
            print("Cost tracking is not initialized.\n")
            return True
        parts = q.split()
        if len(parts) == 1:
            if tracker.budget_limit is None:
                print("No budget set.\n")
            else:
                print(f"Budget limit: ${tracker.budget_limit:.4f}\n")
            return True
        if len(parts) == 3 and parts[1].lower() == "set":
            try:
                tracker.budget_limit = float(parts[2])
                print(f"Budget limit set to: ${tracker.budget_limit:.4f}\n")
            except ValueError:
                print("Invalid budget amount. Use a numeric value.\n")
            return True
        print("Usage:")
        print("  /budget")
        print("  /budget set <amount>\n")
        return True

    # Cost estimate (no LLM call)
    if ql.startswith("/cost-estimate"):
        parts = q.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: /cost-estimate <query>\n")
            return True
        estimate_query_cost(parts[1].strip(), config, state, embedder, collection)
        return True

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
            print("Format must be 'bibtex' or 'zotero'.\n")
            return True

        try:
            if note_path:
                output_path = export_citations_from_note(Path(note_path), fmt=fmt)
            else:
                if not state.last_label_to_candidate:
                    print("No last query sources available to export.\n")
                    return True
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
            print(f"Citations exported to: {output_path}\n")
        except Exception as e:
            print(f"Failed to export citations: {e}\n")
        return True

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
                print(f"Provider not found in config: {provider_filter}\n")
                return True

        seen_labels = set()
        for provider_name in providers:
            label = _display_provider_name(provider_name)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            provider_config = config.llm.get_provider_by_name(provider_name)
            if not provider_config:
                continue
            try:
                test_config = provider_config.resolve_first_available()
                print(f"{label}:")
                if not test_config:
                    print("  (not configured for any feature)\n")
                    continue
                provider = create_provider(test_config)
                print_models(state, provider)
                print()
            except Exception as e:
                logger.error(f"Failed to list models for {provider_name}: {e}")
                print(f"  (failed to list models: {e})\n")
        return True

    # List available providers
    if ql.strip() == "/providers":
        print_providers(config)
        return True

    # Provider health status
    if ql.strip() == "/provider-status":
        print_provider_status(config)
        return True

    if ql.strip() == "/vectordb-providers":
        print_vectordb_providers(config)
        return True

    if ql.strip() == "/vectordb-status":
        print_vectordb_status(config)
        return True

    # Show/set current model
    if ql.startswith("/model"):
        parts = q.split()
        if len(parts) == 1:
            feature_configs = _get_feature_configs(config)
            for feature, cfg in feature_configs.items():
                if cfg is None:
                    continue
                label = _format_feature_label(feature)
                provider = _display_provider_name(cfg.provider)
                print(f"{label}: {provider}/{cfg.model}")
            print()
            return True

        if len(parts) != 4:
            print("Usage:")
            print("  /model                   (show current)")
            print("  /model <task> <provider> <model>  (set model for a task)")
            print("  /models [provider]       (list available models)\n")
            return True

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
            print("Unknown task. Use: query, tag, rerank\n")
            return True

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
            label = _format_feature_label(feature)
            print(f"Model set: {label} = {_display_provider_name(normalized)}/{model_name}\n")
        except Exception as e:
            print(f"Failed to set model: {e}\n")
        return True

    # Show/set current mode
    if ql.startswith("/mode"):
        parts = q.split()
        if len(parts) == 1:
            # Show current mode
            current_mode = config.query.mode
            mode_config = config.get_mode_config(current_mode)
            print(f"Current mode: {current_mode}")
            print(f"  Synthesis style: {mode_config.synthesis_style}")
            print(f"  Local sources: enabled (k={mode_config.source_policy.local.k})")
            print(f"  Remote sources: {'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}")
            print(f"  Model knowledge: {'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}")
            print(f"  Notes: {'enabled' if mode_config.notes.enabled else 'disabled'}")
            print(f"\nAvailable modes: {', '.join(config.modes.keys())}\n")
            return True

        if len(parts) == 2 and parts[1].lower() == "set":
            print("Usage:")
            print("  /mode set <setting> <on|off>")
            print("Settings: model_knowledge, remote, notes\n")
            return True

        if len(parts) >= 2 and parts[1].lower() == "set":
            if len(parts) != 4:
                print("Usage:")
                print("  /mode set <setting> <on|off>")
                print("Settings: model_knowledge, remote, notes\n")
                return True

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
                print(f"Unknown setting: {setting}")
                print("Settings: model_knowledge, remote, notes\n")
                return True

            print(f"Mode setting '{setting}' set to: {'on' if enabled else 'off'}\n")
            return True

        if len(parts) != 2:
            print("Usage:")
            print("  /mode           (show current)")
            print("  /mode <name>    (switch to a different mode)\n")
            return True

        new_mode = parts[1].strip()

        # Validate mode exists
        if new_mode not in config.modes:
            print(f"Mode not found: {new_mode}")
            print(f"Available modes: {', '.join(config.modes.keys())}\n")
            return True

        # Switch mode
        config.query.mode = new_mode
        mode_config = config.get_mode_config(new_mode)
        print(f"Mode switched to: {new_mode}")
        print(f"  Synthesis style: {mode_config.synthesis_style}")
        print(f"  Remote sources: {'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}")
        print(f"  Model knowledge: {'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}\n")
        return True

    # Save note from last query
    if ql.startswith("/note") or ql.startswith("/notes"):
        if not state.last_question:
            print("No query to save. Run a query first.\n")
            return True

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

            print("\nOpening note in editor...")
            print("Edit the note and save/close the editor to continue.\n")

            # Open in editor for user to edit
            edited_content = edit_note_in_editor(content)

            if not edited_content or edited_content.strip() == "":
                print("Note was empty or cancelled. Not saving.\n")
                return True

            # Save the edited note
            from .notes import get_note_filename
            parts = q.split(maxsplit=1)
            filename_override = parts[1].strip() if len(parts) > 1 else None
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

            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(edited_content, encoding="utf-8")

            print(f"Note saved to: {note_path}\n")

        except Exception as e:
            print(f"Failed to save note: {e}\n")
            logger.error(f"Note save error: {e}")

        return True

    # Load document for context pinning
    if ql.startswith("/load"):
        parts = q.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /load <file_path>")
            print("Example: /load /docs/important.md")
            print("\nThis pins a document for forced inclusion in next query context.")
            print("Use /load clear to clear pinned chunks.\n")
            return True

        arg = parts[1].strip()

        # Handle /load clear
        if arg.lower() == "clear":
            state.pinned_chunks = []
            print("Cleared all pinned chunks.\n")
            return True

        file_path = arg

        print(f"Loading chunks from: {file_path}")
        print("Searching collection...")

        try:
            chroma = require_chroma_collection(collection, "/load")
            # Search for chunks from this file
            results = chroma.get(
                where={"source_path": {"$eq": file_path}},
                include=["metadatas"],
            )

            if not results or not results.get("ids"):
                print(f"\nNo chunks found for path: {file_path}")
                print("Tip: Path must match exactly. Use /explore to find exact paths.\n")
                return True

            chunk_ids = results["ids"]
            metadatas = results["metadatas"]

            # Add to pinned chunks
            for chunk_id in chunk_ids:
                if chunk_id not in state.pinned_chunks:
                    state.pinned_chunks.append(chunk_id)

            print(f"\nPinned {len(chunk_ids)} chunks from {file_path}")
            print(f"Total pinned chunks: {len(state.pinned_chunks)}")
            print("\nThese chunks will be forcibly included in your next query.")
            print("Use /load clear to unpin all chunks.\n")

        except Exception as e:
            print(f"Error loading chunks: {e}\n")
            logger.error(f"Load command error: {e}")

        return True

    # Show / Expand
    if ql.startswith("/show") or ql.startswith("/expand"):
        parts = q.split()
        if len(parts) != 2:
            usage = (
                "/show S#   (e.g., /show S2)"
                if ql.startswith("/show")
                else "/expand S#   (e.g., /expand S2)"
            )
            print(f"Usage: {usage}\n")
            return True

        label = parts[1].strip().upper()
        candidate = state.last_label_to_candidate.get(label)
        if not candidate:
            print(f"No such label in last results: {label}\n")
            return True

        print_source_chunk(label, candidate, expanded=ql.startswith("/expand"))
        return True

    # Open
    if ql.startswith("/open"):
        parts = q.split()
        if len(parts) != 2:
            print("Usage: /open S#   (e.g., /open S2)\n")
            return True

        label = parts[1].strip().upper()
        candidate = state.last_label_to_candidate.get(label)
        if not candidate:
            print(f"No such label in last results: {label}\n")
            return True

        path = (candidate.meta or {}).get("source_path")
        if not path:
            print("No source_path available for this citation.\n")
            return True

        open_file(path)
        return True

    # Set filters
    if ql.startswith("/set"):
        parts = q.split()
        if len(parts) < 3:
            print("Usage:")
            print("  /set path_contains <substring> [more...]")
            print("  /set ext_allow .md .pdf")
            print("  /set ext_deny .txt\n")
            return True

        key = parts[1]
        values = parts[2:]

        if key == "path_contains":
            state.path_contains = values if len(values) > 1 else values[0]
            print(f"path_contains set to: {state.path_contains}\n")
            return True

        if key == "ext_allow":
            state.ext_allow = values
            print(f"ext_allow set to: {state.ext_allow}\n")
            return True

        if key == "ext_deny":
            state.ext_deny = values
            print(f"ext_deny set to: {state.ext_deny}\n")
            return True

        print(f"Unknown filter key: {key}\n")
        return True

    # Clear filters
    if ql.startswith("/clear"):
        parts = q.split()
        if len(parts) != 2:
            print("Usage: /clear path_contains|ext_allow|ext_deny\n")
            return True

        key = parts[1]
        if key == "path_contains":
            state.path_contains = None
            print("path_contains cleared.\n")
            return True
        if key == "ext_allow":
            state.ext_allow = None
            print("ext_allow cleared.\n")
            return True
        if key == "ext_deny":
            state.ext_deny = None
            print("ext_deny cleared.\n")
            return True

        print(f"Unknown filter key: {key}\n")
        return True

    if q.startswith("/"):
        print_help()
        return True

    return False


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
        List of remote source dicts
    """
    remote_policy = mode_config.source_policy.remote

    if not remote_policy.enabled:
        return []

    # Get active remote providers
    active_providers = config.get_active_remote_providers()

    if not active_providers:
        logger.warning("Remote sources enabled but no providers configured")
        return []

    all_remote_results = []

    for provider_config in active_providers:
        provider_name = provider_config.name
        try:
            logger.info(f"Fetching from remote provider: {provider_name}")

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

            # Determine max_results
            max_results = provider_config.max_results or remote_policy.max_results

            # Fetch results
            results = provider.search(question, max_results=max_results)

            # Convert to dict format
            for result in results:
                all_remote_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "score": result.score,
                    "provider": provider_name,
                    "metadata": result.metadata,
                })

        except Exception as e:
            logger.error(f"Failed to fetch from {provider_name}: {e}")
            # Continue with other providers

    logger.info(f"Fetched {len(all_remote_results)} remote results")
    return all_remote_results


# -----------------------------
# Cost Estimation
# -----------------------------
def estimate_query_cost(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> None:
    """
    Estimate costs for a query without invoking the LLM.
    """
    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local

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
        print("No results found in Chroma for this query.\n")
        return

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
        print("No results matched the configured filters.\n")
        return

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
        print("Relevance below threshold; no LLM call expected.\n")
        return

    should_llm_rerank = rerank_strategy in ("llm", "hybrid") and not no_rerank
    chosen = filtered[: min(k_rerank, len(filtered))]

    total_estimated = 0.0
    lines = []

    if should_llm_rerank:
        ranking_config = config.llm.get_ranking_config()
        rerank_provider = create_provider(ranking_config)
        rerank_est = estimate_rerank_cost(rerank_provider, question, chosen, k=min(k_rerank, len(chosen)))
        if rerank_est["cost"] is not None:
            total_estimated += rerank_est["cost"]
            lines.append(f"Rerank estimate: ${rerank_est['cost']:.4f}")
        else:
            lines.append("Rerank estimate: unavailable for this provider")

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
        lines.append(f"Synthesis estimate: ${synth_est['cost']:.4f}")
    else:
        lines.append("Synthesis estimate: unavailable for this provider")

    print()
    print("COST ESTIMATE")
    for line in lines:
        print(f"- {line}")
    if total_estimated > 0:
        print(f"Total estimate: ${total_estimated:.4f}")
    print()

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

    # Get configuration values (use mode-specific or fallback to query config)
    batch_size = config.batch_size
    k = local_policy.k
    k_rerank = local_policy.k_rerank
    no_rerank = config.query.no_rerank
    min_relevance = local_policy.min_relevance
    max_per_file = config.query.max_per_file
    local_pool = config.query.local_pool or max(k * 3, k_rerank * 4)

    state.last_question = question

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

    if not candidates:
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

    if not filtered:
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
    relevance = compute_relevance(filtered)

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
    }

    # If relevance is too low, skip LLM and show fallback
    if relevance < min_relevance:
        chosen = filtered[: min(k_rerank, len(filtered))]
        state.last_chosen = chosen
        state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

        answer = fallback_answer(question, chosen)
        _, sources = build_context_block(chosen)

        print("\n" + answer)
        print(format_source_list(sources))
        print()
        return

    # Use ranking-specific LLM config if available
    ranking_config = config.llm.get_ranking_config()
    provider = create_provider(ranking_config)

    # Apply LLM reranking based on strategy
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
    state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

    # Fetch remote sources if enabled
    remote_sources = fetch_remote_sources(question, config, mode_config)

    # Generate answer with citations
    context_block, sources = build_context_block(chosen)

    # Use query-specific LLM config for synthesis
    query_config = config.llm.get_query_config()
    if state.model and state.model != query_config.model:
        query_config = replace(query_config, model=state.model)
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
