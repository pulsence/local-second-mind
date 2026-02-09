"""
Formatting and persistence helpers for agent logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .log_redactor import redact_secrets
from .models import AgentLogEntry


def format_agent_log(entries: List[AgentLogEntry]) -> str:
    """
    Format agent logs into a readable text transcript.

    Args:
        entries: Ordered log entries.

    Returns:
        Human-readable log text.
    """
    lines: list[str] = []
    for entry in entries:
        lines.append(f"Timestamp: {entry.timestamp.isoformat()}")
        lines.append(f"Actor: {entry.actor}")
        if entry.provider_name or entry.model_name:
            lines.append(
                f"LLM (Provider: {entry.provider_name or 'unknown'}: {entry.model_name or 'unknown'})"
            )
        lines.append(f"Agent: {redact_secrets(entry.content)}")
        if entry.action:
            lines.append(f"Requested Action: {entry.action}")
        if entry.action_arguments:
            lines.append(f"Arguments: {json.dumps(entry.action_arguments, sort_keys=True)}")
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def save_agent_log(entries: List[AgentLogEntry], path: Path) -> Path:
    """
    Save agent log entries to JSON.

    Args:
        entries: Log entries to persist.
        path: Output path.

    Returns:
        Written file path.
    """
    payload = [
        {
            "timestamp": entry.timestamp.isoformat(),
            "actor": entry.actor,
            "provider_name": entry.provider_name,
            "model_name": entry.model_name,
            "content": redact_secrets(entry.content),
            "action": entry.action,
            "action_arguments": entry.action_arguments,
        }
        for entry in entries
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_agent_log(path: Path) -> List[AgentLogEntry]:
    """
    Load agent log entries from JSON.

    Args:
        path: Path to JSON log file.

    Returns:
        Deserialized log entries.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[AgentLogEntry] = []
    for item in raw:
        entries.append(
            AgentLogEntry(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                actor=item["actor"],
                provider_name=item.get("provider_name"),
                model_name=item.get("model_name"),
                content=item.get("content", ""),
                action=item.get("action"),
                action_arguments=item.get("action_arguments"),
            )
        )
    return entries
