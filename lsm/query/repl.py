"""
REPL (Read-Eval-Print Loop) for interactive query sessions.

Provides interactive commands, display utilities, and query orchestration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

from lsm.config.models import LSMConfig
from lsm.cli.logging import get_logger
from .session import Candidate, SessionState
from .retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from .rerank import apply_local_reranking
from .synthesis import build_context_block, fallback_answer, format_source_list
from .providers import create_provider

logger = get_logger(__name__)


# -----------------------------
# Display Utilities
# -----------------------------
def print_banner() -> None:
    """Print REPL welcome banner."""
    print("Interactive query mode. Type your question and press Enter.")
    print("Commands: /exit, /help, /show S#, /expand S#, /open S#, /debug, /model, /models, /set, /clear\n")


def print_help() -> None:
    """Print REPL help text."""
    print("Enter a question to query your local knowledge base.")
    print("Commands:")
    print("  /exit           Quit")
    print("  /help           Show this help")
    print("  /show S#        Show the cited chunk (e.g., /show S2)")
    print("  /expand S#      Show full chunk text (no truncation)")
    print("  /open S#        Open the source file in default app")
    print("  /models         List models available to the API key")
    print("  /model          Show current model")
    print("  /model <name>   Set model for this session")
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
def list_models(client: OpenAI) -> List[str]:
    """
    List models available to the current API key.

    Uses the Models API: /v1/models

    Args:
        client: OpenAI client

    Returns:
        List of model IDs (sorted)
    """
    logger.debug("Fetching available models from OpenAI API")

    res = client.models.list()

    # Extract model IDs
    ids: List[str] = []
    for model in getattr(res, "data", []) or []:
        model_id = getattr(model, "id", None)
        if isinstance(model_id, str):
            ids.append(model_id)

    ids.sort()
    logger.info(f"Found {len(ids)} available models")

    return ids


def print_models(state: SessionState, client: OpenAI) -> None:
    """
    Print available models.

    Args:
        state: Session state (to cache model list)
        client: OpenAI client
    """
    try:
        ids = list_models(client)
        state.available_models = ids

        if not ids:
            print("No models returned by the API for this key/project.\n")
            return

        print("\nAvailable models (API key scope):")
        for model_id in ids:
            print(f"- {model_id}")
        print()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        print(f"Failed to list models: {e}\n")


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
    client: OpenAI,
) -> bool:
    """
    Handle REPL commands.

    Args:
        line: User input line
        state: Session state
        client: OpenAI client

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

    # List available models
    if ql.strip() == "/models":
        print_models(state, client)
        return True

    # Show/set current model
    if ql.startswith("/model"):
        parts = q.split()
        if len(parts) == 1:
            print(f"Current model: {state.model}\n")
            return True

        if len(parts) != 2:
            print("Usage:")
            print("  /model           (show current)")
            print("  /model <name>    (set model for this session)")
            print("  /models          (list available models)\n")
            return True

        new_model = parts[1].strip()

        # Optional validation if we've fetched /models at least once
        if state.available_models:
            if new_model not in state.available_models:
                print(f"Model not found in last /models list: {new_model}")
                print("Run /models to refresh the list or set anyway by clearing cache.\n")
                return True

        state.model = new_model
        print(f"Model set to: {state.model}\n")
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

    return False


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

    # Get configuration values
    batch_size = config.batch_size
    k = config.query.k
    k_rerank = config.query.k_rerank
    no_rerank = config.query.no_rerank
    min_relevance = config.query.min_relevance
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

    if not filtered:
        print("No results matched the configured filters.\n")
        return

    # Local quality passes: dedupe, lexical rerank, diversity
    local = apply_local_reranking(
        question,
        filtered,
        max_per_file=max_per_file,
        local_pool=local_pool,
    )

    # Trim to k for downstream steps
    filtered = local[: min(k, len(local))]

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

    # Create provider and rerank/synthesize
    provider = create_provider(config.llm)

    # Optional LLM rerank
    if no_rerank:
        chosen = filtered[: min(k_rerank, len(filtered))]
    else:
        # Convert candidates to provider format
        rerank_candidates = [
            {
                "text": c.text,
                "metadata": c.meta,
                "distance": c.distance,
            }
            for c in filtered
        ]

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

    state.last_chosen = chosen
    state.last_label_to_candidate = {f"S{i}": c for i, c in enumerate(chosen, start=1)}

    # Generate answer with citations
    context_block, sources = build_context_block(chosen)

    answer = provider.synthesize(
        question,
        context_block,
        mode=config.query.mode,
    )

    # Warn if no citations
    if "[S" not in answer:
        answer += (
            "\n\nNote: No inline citations were emitted. "
            "If this persists, tighten query.k / query.k_rerank or reduce chunk size."
        )

    print("\n" + answer)
    print(format_source_list(sources))
    print()


# -----------------------------
# Main REPL Loop
# -----------------------------
def run_repl(
    config: LSMConfig,
    embedder,
    collection,
    client: OpenAI,
) -> None:
    """
    Run the interactive REPL loop.

    Args:
        config: Global configuration
        embedder: SentenceTransformer model
        collection: ChromaDB collection
        client: OpenAI client
    """
    logger.info("Starting REPL session")

    # Initialize session state from config
    state = SessionState(
        path_contains=config.query.path_contains,
        ext_allow=config.query.ext_allow,
        ext_deny=config.query.ext_deny,
        model=config.llm.model,
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

        try:
            if handle_command(line, state, client):
                continue
        except SystemExit:
            print("Exiting.")
            return

        run_query_turn(line, config, state, embedder, collection)
