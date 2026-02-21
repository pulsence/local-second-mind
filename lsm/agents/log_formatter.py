"""
Formatting and persistence helpers for agent logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from lsm.utils.logger import LogVerbosity, normalize_verbosity
from lsm.utils.paths import safe_filename

from .log_redactor import redact_secrets
from .models import AgentLogEntry

_DEFAULT_LOG_EXTENSION = ".log"


def _escape_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return text


def _unescape_text(value: str) -> str:
    text = str(value or "")
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "n":
                result.append("\n")
                i += 2
                continue
            if nxt == "r":
                result.append("\r")
                i += 2
                continue
            if nxt == "t":
                result.append("\t")
                i += 2
                continue
            if nxt == "\\":
                result.append("\\")
                i += 2
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def resolve_agent_log_path(
    *,
    agents_folder: Path,
    agent_name: str,
    timestamp: datetime | None = None,
    extension: str = _DEFAULT_LOG_EXTENSION,
) -> Path:
    """
    Resolve the default log path for an agent run.
    """
    safe_agent = safe_filename(agent_name, default="agent")
    stamp = (timestamp or datetime.utcnow()).strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{safe_agent}_{stamp}{extension}"
    return Path(agents_folder) / safe_agent / "logs" / filename


def _format_entry(entry: AgentLogEntry, verbosity: LogVerbosity) -> str:
    content = _escape_text(redact_secrets(entry.content))
    parts = [entry.timestamp.isoformat(), str(entry.actor), content]

    if entry.action:
        parts.append(f"action={_escape_text(str(entry.action))}")

    if verbosity >= LogVerbosity.VERBOSE:
        if entry.provider_name:
            parts.append(f"provider={_escape_text(str(entry.provider_name))}")
        if entry.model_name:
            parts.append(f"model={_escape_text(str(entry.model_name))}")
        if entry.action_arguments:
            args = json.dumps(entry.action_arguments, sort_keys=True)
            parts.append(f"args={_escape_text(redact_secrets(args))}")

    if verbosity >= LogVerbosity.DEBUG:
        if entry.prompt:
            parts.append(f"prompt={_escape_text(redact_secrets(entry.prompt))}")
        if entry.raw_response:
            parts.append(f"raw_response={_escape_text(redact_secrets(entry.raw_response))}")

    return "\t".join(parts)


def format_agent_log(
    entries: List[AgentLogEntry],
    *,
    verbosity: str | int | LogVerbosity = LogVerbosity.NORMAL,
) -> str:
    """
    Format agent logs into a readable text transcript.

    Args:
        entries: Ordered log entries.
        verbosity: normal, verbose, or debug.

    Returns:
        Human-readable log text.
    """
    level = normalize_verbosity(verbosity, default=LogVerbosity.NORMAL)
    lines: list[str] = []
    for entry in entries:
        lines.append(_format_entry(entry, level))
    return "\n".join(lines).strip() + ("\n" if lines else "")


def save_agent_log(
    entries: List[AgentLogEntry],
    path: Path | None = None,
    *,
    verbosity: str | int | LogVerbosity = LogVerbosity.NORMAL,
    agent_name: str | None = None,
    agents_folder: Path | None = None,
) -> Path:
    """
    Save agent log entries to a plain-text log file.

    Args:
        entries: Log entries to persist.
        path: Optional output path. If omitted, agent_name and agents_folder
            are required to compute the default path.
        verbosity: normal, verbose, or debug.
        agent_name: Agent name used for default log paths.
        agents_folder: Root folder for agent workspaces.

    Returns:
        Written file path.
    """
    if path is None:
        if not agent_name or agents_folder is None:
            raise ValueError("agent_name and agents_folder are required when path is not provided")
        path = resolve_agent_log_path(agents_folder=agents_folder, agent_name=agent_name)

    payload = format_agent_log(entries, verbosity=verbosity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def load_agent_log(path: Path) -> List[AgentLogEntry]:
    """
    Load agent log entries from a plain-text log file.

    Args:
        path: Path to log file.

    Returns:
        Deserialized log entries.
    """
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            entries: list[AgentLogEntry] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                try:
                    timestamp = datetime.fromisoformat(item["timestamp"])
                except Exception:
                    continue
                entries.append(
                    AgentLogEntry(
                        timestamp=timestamp,
                        actor=str(item.get("actor", "")),
                        provider_name=item.get("provider_name"),
                        model_name=item.get("model_name"),
                        content=str(item.get("content", "")),
                        prompt=item.get("prompt"),
                        raw_response=item.get("raw_response"),
                        action=item.get("action"),
                        action_arguments=item.get("action_arguments"),
                    )
                )
            return entries
    entries: list[AgentLogEntry] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        timestamp = datetime.fromisoformat(parts[0])
        actor = parts[1]
        content = _unescape_text(parts[2])
        extras: dict[str, str] = {}
        for extra in parts[3:]:
            if "=" not in extra:
                continue
            key, value = extra.split("=", 1)
            extras[key] = _unescape_text(value)
        action = extras.get("action")
        action_arguments = None
        args_raw = extras.get("args")
        if args_raw:
            try:
                action_arguments = json.loads(args_raw)
            except json.JSONDecodeError:
                action_arguments = None
        entries.append(
            AgentLogEntry(
                timestamp=timestamp,
                actor=actor,
                provider_name=extras.get("provider"),
                model_name=extras.get("model"),
                content=content,
                prompt=extras.get("prompt"),
                raw_response=extras.get("raw_response"),
                action=action,
                action_arguments=action_arguments,
            )
        )
    return entries
