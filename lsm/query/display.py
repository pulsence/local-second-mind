"""
Display utilities for the query module.

Contains result formatting, citation display, and help text generation.
All functions return strings - the UI layer decides how to display them.
"""

from __future__ import annotations

from typing import List, Optional

from lsm.query.session import Candidate, SessionState


# -----------------------------
# Banner and Help
# -----------------------------
def get_banner() -> str:
    """
    Get the query mode welcome banner.

    Returns:
        Banner text string
    """
    lines = [
        "Interactive query mode. Type your question and press Enter.",
        "Commands: /exit, /help, /show S#, /expand S#, /open S#, /debug, /model, /models, /providers, /provider-status, /vectordb-providers, /vectordb-status, /remote-providers, /remote-search, /mode, /note, /notes, /load, /set, /clear, /costs, /budget, /cost-estimate, /export-citations",
        "",
    ]
    return "\n".join(lines)


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
        "  /set …          Set session filters (path/ext)",
        "  /clear …        Clear session filters",
        "",
    ]
    return "\n".join(lines)


# -----------------------------
# Source Chunk Display
# -----------------------------
def format_source_chunk(
    label: str,
    candidate: Candidate,
    expanded: bool = False,
) -> str:
    """
    Format a single source chunk for display.

    DEPRECATED: Use candidate.format(label, expanded) instead.

    Args:
        label: Citation label (e.g., "S1")
        candidate: Candidate to display
        expanded: If True, show full text without truncation

    Returns:
        Formatted chunk string
    """
    return candidate.format(label=label, expanded=expanded)


def format_debug(state: SessionState) -> str:
    """
    Format debug information from last query.

    DEPRECATED: Use state.format_debug() instead.

    Args:
        state: Session state with debug artifacts

    Returns:
        Formatted debug string
    """
    return state.format_debug()


# -----------------------------
# Cost Display
# -----------------------------
def format_costs(state: SessionState) -> str:
    """
    Format current session cost summary.

    DEPRECATED: Use state.format_costs() instead.

    Args:
        state: Session state with cost tracker

    Returns:
        Formatted cost summary string
    """
    return state.format_costs()


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


# -----------------------------
# Backwards compatibility aliases
# -----------------------------
def print_banner() -> None:
    """Print REPL welcome banner. DEPRECATED: Use get_banner() instead."""
    print(get_banner())


def print_help() -> None:
    """Print REPL help text. DEPRECATED: Use get_help() instead."""
    print(get_help())


def print_source_chunk(
    label: str,
    candidate: Candidate,
    expanded: bool = False,
) -> None:
    """Print a single source chunk. DEPRECATED: Use format_source_chunk() instead."""
    print(format_source_chunk(label, candidate, expanded))


def print_debug(state: SessionState) -> None:
    """Print debug information. DEPRECATED: Use format_debug() instead."""
    print(format_debug(state))


def print_costs(state: SessionState) -> None:
    """Print cost summary. DEPRECATED: Use format_costs() instead."""
    print(format_costs(state))


def stream_output(chunks) -> str:
    """
    Stream chunks to stdout and return combined text.

    DEPRECATED: This function prints directly. UI layers should handle
    streaming display themselves.
    """
    print("\nTyping...")
    parts: List[str] = []
    for chunk in chunks:
        if chunk:
            parts.append(chunk)
            print(chunk, end="", flush=True)
    print()
    return "".join(parts).strip()
