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

    direct = provider._send_message(
        system="You are concise.",
        user="Reply with a short acknowledgment for a live test.",
        temperature=0.0,
        max_tokens=80,
    )
    assert isinstance(direct, str)
    assert direct.strip()

    if expect_response_id:
        assert provider.last_response_id

    streamed = _collect_stream_text(
        provider._send_streaming_message(
            system="You are concise.",
            user="Stream a short acknowledgment for a live test.",
            temperature=0.0,
            max_tokens=80,
        )
    )
    assert streamed

    synthesized = provider.synthesize(
        question="What does this live test verify?",
        context="[S1] It verifies that the provider can answer and stream.",
        mode="grounded",
        temperature=0.1,
        max_tokens=160,
    )
    assert isinstance(synthesized, str)
    assert synthesized.strip()

    tags = provider.generate_tags(
        text="Local-first retrieval pipelines require traceable citations and stable embeddings.",
        num_tags=3,
        max_tokens=100,
    )
    assert isinstance(tags, list)
    assert len(tags) <= 3

    candidates = [
        {"text": "Citations improve trust in generated answers.", "metadata": {"idx": 0}},
        {"text": "Chunk overlap affects retrieval recall and context continuity.", "metadata": {"idx": 1}},
        {"text": "Weather forecasts describe atmospheric conditions.", "metadata": {"idx": 2}},
    ]
    reranked = provider.rerank(
        question="Why are citations important in grounded synthesis?",
        candidates=candidates,
        k=2,
    )
    assert len(reranked) == 2
    assert all("text" in item for item in reranked)
