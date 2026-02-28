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
    response.id = "anthropic_resp_1"
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


def test_synthesize_tracks_last_response_id(llm_config):
    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = _mock_anthropic_response("Answer [S1]")
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        answer = provider.synthesize("Question?", "[S1] Context", mode="grounded", enable_server_cache=True)

        assert "Answer" in answer
        assert provider.last_response_id == "anthropic_resp_1"


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


def test_send_message_maps_instruction_and_cache_key(llm_config):
    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = _mock_anthropic_response("Answer")
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        provider.send_message(
            input="Question",
            instruction="System rule",
            prompt_cache_key="anthropic-cache",
            max_tokens=64,
        )

        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["messages"][0]["content"] == "Question"
        assert kwargs["system"][0]["text"] == "System rule"
        assert kwargs["system"][0]["cache_control"]["type"] == "ephemeral"
        assert kwargs["extra_headers"]["x-prompt-cache-key"] == "anthropic-cache"


def test_send_message_logs_unsupported_params(llm_config, caplog):
    import lsm.providers.anthropic as anthropic_module

    anthropic_module._UNSUPPORTED_PARAM_TRACKER._unsupported.clear()
    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = _mock_anthropic_response("Answer")
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        with caplog.at_level("DEBUG", logger="lsm.providers.anthropic"):
            provider.send_message(
                input="Question",
                instruction="System rule",
                previous_response_id="prev-id",
                prompt="Prefix",
                prompt_cache_retention=3600,
                max_tokens=64,
            )

        assert "does not support 'previous_response_id'" in caplog.text
        assert "does not support 'prompt'" in caplog.text
        assert "does not support 'prompt_cache_retention'" in caplog.text


def test_streaming_maps_cache_key(llm_config):
    stream_event = {"type": "content_block_delta", "delta": {"text": "chunk"}}
    with patch("lsm.providers.anthropic.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.stream.side_effect = AttributeError()
        mock_client.messages.create.return_value = [stream_event]
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(llm_config)
        chunks = list(
            provider.send_streaming_message(
                input="Question",
                instruction="System rule",
                prompt_cache_key="anthropic-cache",
                max_tokens=64,
            )
        )

        assert "".join(chunks) == "chunk"
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["system"][0]["cache_control"]["type"] == "ephemeral"
        assert kwargs["extra_headers"]["x-prompt-cache-key"] == "anthropic-cache"
