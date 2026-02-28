"""Tests for Gemini provider implementation."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.gemini import GeminiProvider


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        provider="gemini",
        model="gemini-1.5-pro",
        api_key="test-key",
        temperature=0.7,
        max_tokens=2000,
    )


def _setup_genai_mock(mock_genai: Mock, payload: str) -> Mock:
    mock_response = Mock()
    mock_response.text = payload
    mock_response.response_id = "gemini_resp_1"
    mock_client = Mock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client
    return mock_client


def test_list_models(llm_config: LLMConfig) -> None:
    model_items = [Mock(), Mock()]
    model_items[0].name = "models/gemini-1.5-pro"
    model_items[1].name = "gemini-1.5-flash"
    response = Mock()
    response.page = model_items

    with patch("lsm.providers.gemini.genai") as mock_genai:
        mock_client = Mock()
        mock_client.models.list.return_value = response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(llm_config)
        models = provider.list_models()

        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models


def test_estimate_cost(llm_config: LLMConfig) -> None:
    provider = GeminiProvider(llm_config)
    cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost > 0


def test_send_message_uses_system_instruction_config(llm_config: LLMConfig) -> None:
    with patch("lsm.providers.gemini.genai") as mock_genai:
        mock_client = _setup_genai_mock(mock_genai, "ok")

        provider = GeminiProvider(llm_config)
        provider.send_message(input="Question", instruction="System rule", max_tokens=32)

        kwargs = mock_client.models.generate_content.call_args.kwargs
        assert kwargs["contents"] == "Question"
        assert kwargs["config"].system_instruction == "System rule"


def test_send_message_instruction_optional(llm_config: LLMConfig) -> None:
    with patch("lsm.providers.gemini.genai") as mock_genai:
        mock_client = _setup_genai_mock(mock_genai, "ok")

        provider = GeminiProvider(llm_config)
        provider.send_message(input="Question", instruction=None, max_tokens=32)

        kwargs = mock_client.models.generate_content.call_args.kwargs
        assert kwargs["contents"] == "Question"
        assert kwargs["config"].system_instruction is None


def test_send_message_logs_unsupported_cache_params(llm_config: LLMConfig, caplog) -> None:
    import lsm.providers.gemini as gemini_module

    gemini_module._UNSUPPORTED_PARAM_TRACKER._unsupported.clear()
    with patch("lsm.providers.gemini.genai") as mock_genai:
        _setup_genai_mock(mock_genai, "ok")

        provider = GeminiProvider(llm_config)
        with caplog.at_level("DEBUG", logger="lsm.providers.gemini"):
            provider.send_message(
                input="Question",
                previous_response_id="resp-prev",
                prompt_cache_key="cache-key",
                prompt_cache_retention=300,
                max_tokens=32,
            )

        assert "does not support 'previous_response_id'" in caplog.text
        assert "does not support 'prompt_cache_key'" in caplog.text
        assert "does not support 'prompt_cache_retention'" in caplog.text


def test_streaming_uses_system_instruction_config(llm_config: LLMConfig) -> None:
    stream_chunk = Mock()
    stream_chunk.text = "hello"

    with patch("lsm.providers.gemini.genai") as mock_genai:
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = [stream_chunk]
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(llm_config)
        chunks = list(
            provider.send_streaming_message(
                input="Question",
                instruction="System stream rule",
                max_tokens=32,
            )
        )

        assert "".join(chunks) == "hello"
        kwargs = mock_client.models.generate_content_stream.call_args.kwargs
        assert kwargs["contents"] == "Question"
        assert kwargs["config"].system_instruction == "System stream rule"
