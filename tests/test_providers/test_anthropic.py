"""
Tests for Anthropic provider implementation.
"""

import json
from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.anthropic import AnthropicProvider


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
        temperature=0.7,
        max_tokens=2000,
    )


def _mock_anthropic_response(payload: str) -> Mock:
    part = Mock()
    part.text = payload
    response = Mock()
    response.content = [part]
    return response


def test_is_available(llm_config):
    provider = AnthropicProvider(llm_config)
    assert provider.is_available() is True


def test_rerank_success(llm_config):
    candidates = [
        {"text": "First", "metadata": {}, "distance": 0.1},
        {"text": "Second", "metadata": {}, "distance": 0.2},
        {"text": "Third", "metadata": {}, "distance": 0.3},
    ]

    ranking = json.dumps(
        {"ranking": [{"index": 2, "reason": "Best"}, {"index": 0, "reason": "Good"}]}
    )

    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = _mock_anthropic_response(ranking)
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        result = provider.rerank("Question?", candidates, k=2)

        assert len(result) == 2
        assert result[0]["text"] == "Third"
        assert result[1]["text"] == "First"

        health = provider.health_check()
        assert health["stats"]["success_count"] == 1


def test_synthesize_fallback_on_error(llm_config):
    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        answer = provider.synthesize("Question?", "[S1] Context", mode="grounded")

        assert "Offline mode" in answer
        health = provider.health_check()
        assert health["stats"]["failure_count"] == 1


def test_generate_tags_success(llm_config):
    tags_payload = json.dumps({"tags": ["python", "analysis"]})

    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = _mock_anthropic_response(tags_payload)
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        tags = provider.generate_tags("Python analysis", num_tags=2)

        assert tags == ["python", "analysis"]


def test_generate_tags_code_fence(llm_config):
    tags_payload = "```json\n{\"tags\": [\"alpha\", \"beta\"]}\n```"

    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = _mock_anthropic_response(tags_payload)
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        tags = provider.generate_tags("Alpha beta", num_tags=2)

        assert tags == ["alpha", "beta"]


def test_list_models(llm_config):
    model_items = [Mock(id="claude-3-haiku-20240307"), Mock(id="claude-3-5-sonnet-20241022")]
    response = Mock()
    response.data = model_items

    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.models.list.return_value = response
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        models = provider.list_models()

        assert "claude-3-haiku-20240307" in models
        assert "claude-3-5-sonnet-20241022" in models


def test_estimate_cost(llm_config):
    provider = AnthropicProvider(llm_config)
    cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost > 0
