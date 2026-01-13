"""
Tests for Gemini provider implementation.
"""

import json
from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.base import BaseLLMProvider
from lsm.providers.gemini import GeminiProvider


@pytest.fixture(autouse=True)
def reset_health_stats():
    BaseLLMProvider._GLOBAL_HEALTH_STATS = {}


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="gemini",
        model="gemini-1.5-pro",
        api_key="test-key",
        temperature=0.7,
        max_tokens=2000,
    )


def _setup_genai_mock(mock_genai: Mock, payload: str) -> None:
    mock_response = Mock()
    mock_response.text = payload
    mock_model = Mock()
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model


def test_rerank_success(llm_config):
    candidates = [
        {"text": "First", "metadata": {}, "distance": 0.1},
        {"text": "Second", "metadata": {}, "distance": 0.2},
        {"text": "Third", "metadata": {}, "distance": 0.3},
    ]

    ranking = json.dumps(
        {"ranking": [{"index": 0, "reason": "Best"}, {"index": 2, "reason": "Good"}]}
    )

    with patch("lsm.providers.gemini.genai") as mock_genai:
        _setup_genai_mock(mock_genai, ranking)

        provider = GeminiProvider(llm_config)
        result = provider.rerank("Question?", candidates, k=2)

        assert len(result) == 2
        assert result[0]["text"] == "First"
        assert result[1]["text"] == "Third"

        health = provider.health_check()
        assert health["stats"]["success_count"] == 1


def test_synthesize_fallback_on_error(llm_config):
    with patch("lsm.providers.gemini.genai") as mock_genai:
        mock_genai.GenerativeModel.side_effect = Exception("API error")

        provider = GeminiProvider(llm_config)
        answer = provider.synthesize("Question?", "[S1] Context", mode="grounded")

        assert "Offline mode" in answer
        health = provider.health_check()
        assert health["stats"]["failure_count"] == 1


def test_generate_tags_success(llm_config):
    tags_payload = json.dumps({"tags": ["gemini", "provider"]})

    with patch("lsm.providers.gemini.genai") as mock_genai:
        _setup_genai_mock(mock_genai, tags_payload)

        provider = GeminiProvider(llm_config)
        tags = provider.generate_tags("Gemini provider", num_tags=2)

        assert tags == ["gemini", "provider"]


def test_estimate_cost(llm_config):
    provider = GeminiProvider(llm_config)
    cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost > 0
