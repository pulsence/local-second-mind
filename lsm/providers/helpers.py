"""
Generic provider helper utilities.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Set


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences around content."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_json_payload(raw: str) -> Optional[Any]:
    """Parse JSON payload from plain text or markdown-wrapped text."""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

    return None


class UnsupportedParamTracker:
    """Track unsupported model parameters and detection logic."""

    def __init__(self) -> None:
        self._unsupported: Dict[str, Set[str]] = {}

    def mark_unsupported(self, model: str, param: str) -> None:
        self._unsupported.setdefault(model, set()).add(param)

    def should_send(self, model: str, param: str) -> bool:
        return param not in self._unsupported.get(model, set())

    def is_unsupported_error(self, error: Exception, param: str) -> bool:
        message = str(error)
        return (
            f"Unsupported parameter: '{param}'" in message
            or f"Unsupported parameter: \"{param}\"" in message
            or f"unexpected keyword argument '{param}'" in message
        )
