"""Shared parsing and formatting helpers for UI slash commands."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Sequence


QUERY_MODE_VALUES: tuple[str, ...] = ("grounded", "insight", "hybrid", "chat", "single")
QUERY_MODE_SETTINGS: tuple[str, ...] = ("model_knowledge", "remote", "notes", "llm_cache")
QUERY_CONTEXT_ACTIONS: tuple[str, ...] = ("doc", "chunk", "clear")
QUERY_EXPORT_FORMATS: tuple[str, ...] = ("bibtex", "zotero")
UI_DENSITY_VALUES: tuple[str, ...] = ("auto", "compact", "comfortable")

TRUE_VALUES: frozenset[str] = frozenset({"on", "true", "yes", "1"})
FALSE_VALUES: frozenset[str] = frozenset({"off", "false", "no", "0"})


class CommandParseError(ValueError):
    """Raised when command tokenization fails."""


@dataclass(frozen=True)
class ParsedCommand:
    """Normalized slash command representation."""

    text: str
    parts: tuple[str, ...]
    cmd: str


def normalize_argument(value: object) -> str:
    """Normalize a single argument to a trimmed string."""
    return str(value or "").strip()


def normalize_arguments(
    values: Sequence[object],
    *,
    lower: bool = False,
) -> list[str]:
    """Normalize a sequence of arguments into a list of trimmed strings."""
    normalized = [normalize_argument(value) for value in values]
    if lower:
        return [value.lower() for value in normalized]
    return normalized


def tokenize_command(command: str, *, use_shlex: bool = False) -> list[str]:
    """Split a command string into tokens."""
    text = normalize_argument(command)
    if not text:
        return []
    if not use_shlex:
        return text.split()
    try:
        return shlex.split(text)
    except ValueError as exc:  # pragma: no cover - exercised by shell command tests
        raise CommandParseError(str(exc)) from exc


def parse_slash_command(command: str) -> ParsedCommand:
    """Parse slash command text into tokens and command id."""
    text = normalize_argument(command)
    if not text.startswith("/"):
        return ParsedCommand(text=text, parts=(), cmd="")
    parts = tuple(tokenize_command(text))
    cmd = parts[0].lower() if parts else ""
    return ParsedCommand(text=text, parts=parts, cmd=cmd)


def parse_on_off_value(value: str) -> bool | None:
    """Parse common on/off-style values into a boolean."""
    normalized = normalize_argument(value).lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def format_command_error(message: str) -> str:
    """Format command/parser errors consistently with trailing newline."""
    return f"{normalize_argument(message)}\n"

