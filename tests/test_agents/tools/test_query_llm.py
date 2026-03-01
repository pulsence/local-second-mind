"""
Tests for QueryLLMTool cache-aware chaining.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lsm.agents.tools.query_llm import QueryLLMTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Minimal stand-in for a BaseLLMProvider."""

    def __init__(self, response="LLM answer", response_id=None):
        self.last_kwargs = {}
        self._response = response
        self.last_response_id = response_id

    def send_message(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class FakeLLMRegistry:
    """Minimal stand-in for LLMRegistryConfig."""

    def __init__(self, provider):
        self._provider = provider

    def resolve_service(self, name):
        return SimpleNamespace(
            temperature=0.7,
            max_tokens=1024,
            model="test-model",
        )

    def resolve_tier(self, tier):
        return self.resolve_service(tier)


def _make_tool(provider=None, response_id=None):
    if provider is None:
        provider = FakeProvider(response_id=response_id)
    registry = FakeLLMRegistry(provider)
    tool = QueryLLMTool(registry)
    return tool, provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_prompt(monkeypatch):
    provider = FakeProvider(response="hello world")
    tool, _ = _make_tool(provider)
    monkeypatch.setattr(
        "lsm.agents.tools.query_llm.create_provider",
        lambda cfg: provider,
    )

    output = json.loads(tool.execute({"prompt": "say hello"}))
    assert output["answer"] == "hello world"


def test_cache_parameters_forwarded(monkeypatch):
    provider = FakeProvider()
    tool, _ = _make_tool(provider)
    monkeypatch.setattr(
        "lsm.agents.tools.query_llm.create_provider",
        lambda cfg: provider,
    )

    tool.execute({
        "prompt": "test",
        "previous_response_id": "resp-prev-001",
        "prompt_cache_key": "cache-key-abc",
        "prompt_cache_retention": 3600,
    })

    assert provider.last_kwargs["previous_response_id"] == "resp-prev-001"
    assert provider.last_kwargs["prompt_cache_key"] == "cache-key-abc"
    assert provider.last_kwargs["prompt_cache_retention"] == 3600


def test_response_id_in_output(monkeypatch):
    provider = FakeProvider(response_id="resp-new-001")
    tool, _ = _make_tool(provider)
    monkeypatch.setattr(
        "lsm.agents.tools.query_llm.create_provider",
        lambda cfg: provider,
    )

    output = json.loads(tool.execute({"prompt": "test"}))
    assert output["response_id"] == "resp-new-001"


def test_no_response_id_omitted(monkeypatch):
    provider = FakeProvider(response_id=None)
    tool, _ = _make_tool(provider)
    monkeypatch.setattr(
        "lsm.agents.tools.query_llm.create_provider",
        lambda cfg: provider,
    )

    output = json.loads(tool.execute({"prompt": "test"}))
    assert "response_id" not in output


def test_empty_prompt_raises(monkeypatch):
    tool, _ = _make_tool()
    monkeypatch.setattr(
        "lsm.agents.tools.query_llm.create_provider",
        lambda cfg: FakeProvider(),
    )

    with pytest.raises(ValueError, match="prompt is required"):
        tool.execute({"prompt": ""})


def test_context_mode_with_cache(monkeypatch):
    provider = FakeProvider()
    tool, _ = _make_tool(provider)
    monkeypatch.setattr(
        "lsm.agents.tools.query_llm.create_provider",
        lambda cfg: provider,
    )

    tool.execute({
        "prompt": "summarize",
        "context": "Python is a programming language.",
        "mode": "grounded",
        "previous_response_id": "resp-prev",
    })

    assert provider.last_kwargs["previous_response_id"] == "resp-prev"
    # Should have instruction set (grounded mode)
    assert provider.last_kwargs.get("instruction") is not None
