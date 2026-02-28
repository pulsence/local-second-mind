"""Shared assertions for live LLM provider contract tests."""

from __future__ import annotations

from typing import Iterable


def _collect_stream_text(chunks: Iterable[str], max_chunks: int = 20) -> str:
    pieces: list[str] = []
    for idx, chunk in enumerate(chunks):
        if chunk:
            pieces.append(chunk)
        if idx >= max_chunks - 1:
            break
    return "".join(pieces).strip()


def exercise_live_provider_contract(
    provider,
    *,
    expect_response_id: bool,
) -> None:
    """Run a compact end-to-end check for BaseLLMProvider concrete methods."""
    assert provider.is_available() is True

    health = provider.health_check()
    assert health["provider"] == provider.name
    assert health["available"] is True

    direct = provider.send_message(
        instruction="You are concise.",
        input="Reply with a short acknowledgment for a live test.",
        temperature=0.0,
        max_tokens=80,
    )
    assert isinstance(direct, str)
    assert direct.strip()

    if expect_response_id:
        assert provider.last_response_id

    streamed = _collect_stream_text(
        provider.send_streaming_message(
            instruction="You are concise.",
            input="Stream a short acknowledgment for a live test.",
            temperature=0.0,
            max_tokens=80,
        )
    )
    assert streamed

