"""
Redaction helpers for secret-like content in agent logs.
"""

from __future__ import annotations

import re

_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*)\s*([:=])\s*([^\s,;]+)"
)
_SECRET_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bkey_[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)


def redact_secrets(text: str) -> str:
    """
    Redact secret-like values from text.

    Args:
        text: Input text.

    Returns:
        Redacted text.
    """
    value = str(text or "")
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1\2[REDACTED]", value)
    for pattern in _SECRET_TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
