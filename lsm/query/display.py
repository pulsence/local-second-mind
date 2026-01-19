"""
Display utilities for the query REPL.

Contains result formatting, citation display, streaming output, and banner/help printing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from lsm.query.session import Candidate, SessionState


# -----------------------------
# Banner and Help
# -----------------------------
def print_banner() -> None:
    """Print REPL welcome banner."""
    print("Interactive query mode. Type your question and press Enter.")
    print("Commands: /exit, /help, /show S#, /expand S#, /open S#, /debug, /model, /models, /providers, /provider-status, /vectordb-providers, /vectordb-status, /remote-providers, /remote-search, /mode, /note, /notes, /load, /set, /clear, /costs, /budget, /cost-estimate, /export-citations\n")


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
    print("  /remote-providers    List available remote source providers")
    print("  /remote-search <provider> <query>  Test a remote provider")
    print("  /remote-search-all <query>  Search all enabled providers")
    print("  /remote-provider enable|disable|weight <name> [value]")
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


# -----------------------------
# Source Chunk Display
# -----------------------------
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
# Cost Display
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


# -----------------------------
# Streaming Output
# -----------------------------
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


# -----------------------------
# Model Display Helpers
# -----------------------------
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
