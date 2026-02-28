"""
Fallback answer utilities for query flows.
"""

from __future__ import annotations


def generate_fallback_answer(
    question: str,
    context: str,
    provider_name: str,
    max_chars: int = 1200,
) -> str:
    """Generate fallback answer when synthesis provider calls fail."""
    snippet = context[:max_chars]
    if len(context) > max_chars:
        snippet += "\n...[truncated]..."

    return (
        f"[Offline mode: {provider_name} unavailable]\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{snippet}\n\n"
        "Note: Unable to generate synthesized answer. "
        "Please review the sources above directly."
    )
