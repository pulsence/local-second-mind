"""
Tests for Azure OpenAI provider implementation.
"""

import json
from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.azure_openai import AzureOpenAIProvider


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="azure_openai",
        model="gpt-35-turbo",
        api_key="test-key",
        endpoint="https://example.openai.azure.com/",
        api_version="2023-05-15",
        deployment_name="gpt-35-turbo",
        temperature=0.7,
        max_tokens=2000,
    )


def test_rerank_success(llm_config):
    candidates = [
        {"text": "First", "metadata": {}, "distance": 0.1},
        {"text": "Second", "metadata": {}, "distance": 0.2},
        {"text": "Third", "metadata": {}, "distance": 0.3},
    ]

    ranking = json.dumps(
        {"ranking": [{"index": 2, "reason": "Best"}, {"index": 1, "reason": "Good"}]}
    )

    with patch("lsm.providers.azure_openai.AzureOpenAI") as mock_azure:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = ranking
        mock_client.responses.create.return_value = mock_response
        mock_azure.return_value = mock_client

        provider = AzureOpenAIProvider(llm_config)
        result = provider.rerank("Question?", candidates, k=2)

        assert len(result) == 2
        assert result[0]["text"] == "Third"
        assert result[1]["text"] == "Second"

        health = provider.health_check()
        assert health["stats"]["success_count"] == 1


def test_synthesize_fallback_on_error(llm_config):
    with patch("lsm.providers.azure_openai.AzureOpenAI") as mock_azure:
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("API error")
        mock_azure.return_value = mock_client

        provider = AzureOpenAIProvider(llm_config)
        answer = provider.synthesize("Question?", "[S1] Context", mode="grounded")

        assert "Offline mode" in answer
        health = provider.health_check()
        assert health["stats"]["failure_count"] == 1


def test_generate_tags_success(llm_config):
    tags_payload = json.dumps({"tags": ["azure", "openai"]})

    with patch("lsm.providers.azure_openai.AzureOpenAI") as mock_azure:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = tags_payload
        mock_client.responses.create.return_value = mock_response
        mock_azure.return_value = mock_client

        provider = AzureOpenAIProvider(llm_config)
        tags = provider.generate_tags("Azure OpenAI", num_tags=2)

        assert tags == ["azure", "openai"]
