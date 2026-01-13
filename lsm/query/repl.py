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

from lsm.config.models import LSMConfig, LLMConfig
from lsm.cli.logging import get_logger
from lsm.providers import create_provider
from .session import Candidate, SessionState
from .retrieval import embed_text, retrieve_candidates, filter_candidates, compute_relevance
from .rerank import apply_local_reranking
from .synthesis import build_context_block, fallback_answer, format_source_list
from .remote import create_remote_provider
from .notes import write_note, generate_note_content, edit_note_in_editor

logger = get_logger(__name__)


# -----------------------------
# Display Utilities
# -----------------------------
def print_banner() -> None:
    """Print REPL welcome banner."""
    print("Interactive query mode. Type your question and press Enter.")
    print("Commands: /exit, /help, /show S#, /expand S#, /open S#, /debug, /model, /models, /providers, /provider-status, /mode, /note, /load, /set, /clear\n")


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
    print("  /providers      List available LLM providers")
    print("  /provider-status Show provider health and recent stats")
    print("  /mode           Show current query mode")
    print("  /mode <name>    Switch to a different query mode")
    print("  /note           Save last query as an editable note")
    print("  /load <path>    Pin a document for forced context inclusion")
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


def print_providers(config: LSMConfig) -> None:
    """
    Print available LLM providers.

    Args:
        config: LSM configuration
    """
    from lsm.providers import list_available_providers

    print()
    print("=" * 60)
    print("AVAILABLE LLM PROVIDERS")
    print("=" * 60)
    print()

    providers = list_available_providers()

    if not providers:
        print("No providers registered.")
        print()
        return

    # Show current provider
    current_provider = config.llm.provider
    print(f"Current Provider: {current_provider}")
    print(f"Current Model:    {config.llm.model}")
    print()

    # List all available providers
    print(f"Available Providers ({len(providers)}):")
    print()

    for provider_name in providers:
        # Try to create provider to check availability
        try:
            # Create a test config for this provider
            use_current = provider_name == current_provider
            test_config = LLMConfig(
                provider=provider_name,
                model=config.llm.model,
                api_key=config.llm.api_key if use_current else None,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                base_url=config.llm.base_url if use_current else None,
                endpoint=config.llm.endpoint if use_current else None,
                api_version=config.llm.api_version if use_current else None,
                deployment_name=config.llm.deployment_name if use_current else None,
            )
            provider = create_provider(test_config)

            # Check if available
            is_current = "✓ ACTIVE" if provider_name == current_provider else ""
            is_available = "✓" if provider.is_available() else "✗ (API key not configured)"

            print(f"  {provider_name:20s} {is_available:30s} {is_current}")

        except Exception as e:
            logger.debug(f"Error checking provider {provider_name}: {e}")
            is_current = "✓ ACTIVE" if provider_name == current_provider else ""
            print(f"  {provider_name:20s} {'✗ (Error)':30s} {is_current}")

    print()
    print("To switch providers, update your config.json:")
    print('  "llm": { "provider": "provider_name", ... }')
    print()
    print("See docs/api-reference/ADDING_PROVIDERS.md for adding new providers.")
    print()


def print_provider_status(config: LSMConfig) -> None:
    """
    Print provider health status and call statistics.

    Args:
        config: LSM configuration
    """
    from lsm.providers import list_available_providers

    print()
    print("=" * 60)
    print("PROVIDER HEALTH STATUS")
    print("=" * 60)
    print()

    providers = list_available_providers()
    if not providers:
        print("No providers registered.")
        print()
        return

    for provider_name in providers:
        try:
            use_current = provider_name == config.llm.provider
            test_config = LLMConfig(
                provider=provider_name,
                model=config.llm.model,
                api_key=config.llm.api_key if use_current else None,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                base_url=config.llm.base_url if use_current else None,
                endpoint=config.llm.endpoint if use_current else None,
                api_version=config.llm.api_version if use_current else None,
                deployment_name=config.llm.deployment_name if use_current else None,
            )
            provider = create_provider(test_config)
            health = provider.health_check()

            status = health.get("status", "unknown")
            stats = health.get("stats", {})
            success = stats.get("success_count", 0)
            failure = stats.get("failure_count", 0)
            last_error = stats.get("last_error")

            print(f"{provider_name:20s} status={status:12s} success={success:4d} failure={failure:4d}")
            if last_error:
                print(f"{'':20s} last_error={last_error}")
        except Exception as e:
            logger.debug(f"Error checking provider status {provider_name}: {e}")
            print(f"{provider_name:20s} status=error        error={e}")

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
    client: OpenAI,
    config: LSMConfig,
    collection,
) -> bool:
    """
    Handle REPL commands.

    Args:
        line: User input line
        state: Session state
        client: OpenAI client
        config: Global configuration
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

    # List available models
    if ql.strip() == "/models":
        print_models(state, client)
        return True

    # List available providers
    if ql.strip() == "/providers":
        print_providers(config)
        return True

    # Provider health status
    if ql.strip() == "/provider-status":
        print_provider_status(config)
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
    if ql.startswith("/note"):
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
            filename = get_note_filename(state.last_question, format=notes_config.filename_format)
            notes_dir.mkdir(parents=True, exist_ok=True)
            note_path = notes_dir / filename
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
            # Search for chunks from this file
            results = collection.get(
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

    for provider_name, provider_config in active_providers.items():
        try:
            logger.info(f"Fetching from remote provider: {provider_name}")

            # Create provider instance
            provider = create_remote_provider(
                provider_config.type,
                {
                    "api_key": provider_config.api_key,
                    "endpoint": provider_config.endpoint,
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
            pinned_results = collection.get(
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

    # Create provider and rerank/synthesize
    if state.model and state.model != config.llm.model:
        config.llm.model = state.model

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
    synthesis_provider = create_provider(query_config)

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
            if handle_command(line, state, client, config, collection):
                continue
        except SystemExit:
            print("Exiting.")
            return

        run_query_turn(line, config, state, embedder, collection)
