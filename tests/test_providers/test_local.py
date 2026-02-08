"""
Tests for Local (Ollama) provider implementation.
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from lsm.config.models import LLMConfig
from lsm.providers.local import LocalProvider


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="local",
        model="llama2",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2000,
    )


def _mock_response(payload: str) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": payload}}
    return response


def test_rerank_success(llm_config):
    candidates = [
        {"text": "First", "metadata": {}, "distance": 0.1},
        {"text": "Second", "metadata": {}, "distance": 0.2},
        {"text": "Third", "metadata": {}, "distance": 0.3},
    ]

    ranking = json.dumps(
        {"ranking": [{"index": 1, "reason": "Best"}, {"index": 0, "reason": "Good"}]}
    )

    with patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.return_value = _mock_response(ranking)

        provider = LocalProvider(llm_config)
        result = provider.rerank("Question?", candidates, k=2)

        assert len(result) == 2
        assert result[0]["text"] == "Second"
        assert result[1]["text"] == "First"

        health = provider.health_check()
        assert health["stats"]["success_count"] == 1


def test_synthesize_fallback_on_error(llm_config):
    with patch("lsm.providers.base.time.sleep"), patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")

        provider = LocalProvider(llm_config)
        answer = provider.synthesize("Question?", "[S1] Context", mode="grounded")

        assert "Offline mode" in answer
        health = provider.health_check()
        assert health["stats"]["failure_count"] == 1


def test_synthesize_sets_last_response_id_none(llm_config):
    with patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.return_value = _mock_response("Answer [S1]")

        provider = LocalProvider(llm_config)
        answer = provider.synthesize("Question?", "[S1] Context", mode="grounded", enable_server_cache=True)

        assert "Answer" in answer
        assert provider.last_response_id is None


def test_generate_tags_success(llm_config):
    tags_payload = json.dumps({"tags": ["local", "model"]})

    with patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.return_value = _mock_response(tags_payload)

        provider = LocalProvider(llm_config)
        tags = provider.generate_tags("Local model", num_tags=2)

        assert tags == ["local", "model"]
