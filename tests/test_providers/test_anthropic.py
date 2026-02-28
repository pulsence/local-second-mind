"""Tests for Anthropic provider implementation."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.anthropic import AnthropicProvider


@pytest.fixture
def llm_config() -> LLMConfig:
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


def test_is_available(llm_config: LLMConfig) -> None:
    provider = AnthropicProvider(llm_config)
    assert provider.is_available() is True


def test_list_models(llm_config: LLMConfig) -> None:
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


def test_estimate_cost(llm_config: LLMConfig) -> None:
    provider = AnthropicProvider(llm_config)
    cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost > 0


def test_send_message_maps_instruction_and_cache_key(llm_config: LLMConfig) -> None:
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


def test_send_message_logs_unsupported_params(llm_config: LLMConfig, caplog) -> None:
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


def test_streaming_maps_cache_key(llm_config: LLMConfig) -> None:
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
