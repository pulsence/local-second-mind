"""Pure formatting helpers for agent log display and status rendering.

These functions extract rendering logic from AgentsScreen so formatting
can be tested independently of the TUI widget tree.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from rich.text import Text


def format_duration(duration_seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS or MM:SS.

    Args:
        duration_seconds: Elapsed seconds.

    Returns:
        Formatted duration string.
    """
    seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_interval_label(interval_seconds: float) -> str:
    """Format an interval in seconds as a label.

    Args:
        interval_seconds: Interval in seconds.

    Returns:
        Formatted interval string.
    """
    return f"{float(interval_seconds):.1f}s"


def stream_prefix(actor: str) -> tuple[str, str]:
    """Return (label, style) pair for an agent log actor prefix.

    Args:
        actor: Actor identifier (llm, tool, user, agent).

    Returns:
        Tuple of (display label, Rich style string).
    """
    normalized = str(actor or "").strip().lower()
    if normalized == "llm":
        return "LLM", "bright_cyan"
    if normalized == "tool":
        return "TOOL", "bright_magenta"
    if normalized == "user":
        return "USER", "bright_yellow"
    if normalized == "agent":
        return "AGENT", "bright_green"
    return "EVENT", "grey70"


def summarize_stream_args(value: Any) -> str:
    """Summarize tool action arguments for log display.

    Args:
        value: Raw arguments (dict, str, or None).

    Returns:
        Truncated string summary.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, dict) and not value:
        return ""
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        text = str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > 100:
        return text[:97].rstrip() + "..."
    return text


def summarize_stream_content(content: str) -> str:
    """Summarize log entry content for display.

    Args:
        content: Raw content string.

    Returns:
        Truncated string summary.
    """
    text = str(content or "").replace("\n", " ").strip()
    if not text:
        return "(empty)"
    if len(text) > 180:
        return text[:177].rstrip() + "..."
    return text


def format_stream_log_entry(row: dict[str, Any]) -> Text:
    """Format a single stream log entry as a Rich Text object.

    Args:
        row: Dict with keys ``actor``, ``content``, ``action``,
            ``action_arguments``.

    Returns:
        Formatted Rich Text line.
    """
    actor = str(row.get("actor", "")).strip().lower()
    prefix_label, prefix_style = stream_prefix(actor)
    content = str(row.get("content", "")).strip()
    action = str(row.get("action", "")).strip()
    args_summary = summarize_stream_args(row.get("action_arguments"))
    summary = summarize_stream_content(content)

    if actor == "tool":
        action_name = action or "tool"
        if args_summary:
            body = f"{action_name}({args_summary}) -> {summary}"
        else:
            body = f"{action_name} -> {summary}"
    elif actor == "llm":
        if action:
            body = f"{summary} (action={action})"
        else:
            body = summary
    else:
        body = summary

    line = Text(f"[{prefix_label}] ", style=prefix_style)
    line.append(body)
    return line


def format_stream_drop_notice(dropped_count: int) -> Text:
    """Format a stream drop notice as a Rich Text object.

    Args:
        dropped_count: Number of dropped entries.

    Returns:
        Formatted Rich Text line.
    """
    line = Text("[STREAM] ", style="bold yellow")
    line.append(
        f"Dropped {dropped_count} log entr{'y' if dropped_count == 1 else 'ies'} due to queue pressure.",
        style="yellow",
    )
    return line
