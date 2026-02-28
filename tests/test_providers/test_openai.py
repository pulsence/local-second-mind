"""Tests for OpenAI provider implementation."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.openai import OpenAIProvider


class TestOpenAIProvider:
    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        return LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=2000,
        )

    def test_provider_initialization(self, llm_config: LLMConfig) -> None:
        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(llm_config)
            assert provider.name == "openai"
            assert provider.model == "gpt-5.2"
            mock_openai.assert_called_once_with(api_key="test-api-key")

    def test_provider_initialization_without_api_key(self) -> None:
        config = LLMConfig(provider="openai", model="gpt-5.2")
        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(config)
            assert provider.name == "openai"
            mock_openai.assert_called_once_with()

    def test_is_available_with_api_key(self, llm_config: LLMConfig) -> None:
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            assert provider.is_available() is True

    def test_is_available_without_api_key(self) -> None:
        config = LLMConfig(provider="openai", model="gpt-5.2", api_key=None)
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
            assert provider.is_available() is False

    def test_send_message_uses_plain_string_input_and_optional_instruction(
        self,
        llm_config: LLMConfig,
    ) -> None:
        mock_response = Mock()
        mock_response.output_text = "ok"
        mock_response.id = "resp_plain"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            provider.send_message(input="hello", instruction=None, prompt="preface", max_tokens=20)

            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["input"] == "preface\n\nhello"
            assert "instructions" not in kwargs

    def test_send_message_forwards_cache_params_without_enable_gate(
        self,
        llm_config: LLMConfig,
    ) -> None:
        mock_response = Mock()
        mock_response.output_text = "cached"
        mock_response.id = "resp_new"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            provider.send_message(
                input="follow-up",
                instruction="system",
                previous_response_id="resp_prev",
                prompt_cache_key="key-1",
                prompt_cache_retention=7200,
                max_tokens=40,
            )

            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["previous_response_id"] == "resp_prev"
            assert kwargs["prompt_cache_key"] == "key-1"
            assert kwargs["prompt_cache_retention"] == 7200
            assert provider.last_response_id == "resp_new"

    def test_streaming_supports_cache_prompt_tools_and_json_schema(
        self,
        llm_config: LLMConfig,
    ) -> None:
        stream_events = [
            {"type": "response.output_text.delta", "delta": "hello"},
            {"type": "response.completed", "response": {"id": "resp_stream"}},
        ]

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = iter(stream_events)
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            chunks = list(
                provider.send_streaming_message(
                    input="payload",
                    instruction="rules",
                    prompt="prefix",
                    previous_response_id="resp_prev",
                    prompt_cache_key="cache-key",
                    prompt_cache_retention=600,
                    json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    tools=[
                        {
                            "name": "ping",
                            "description": "Ping tool",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    tool_choice="auto",
                    max_tokens=50,
                )
            )

            assert "".join(chunks) == "hello"
            assert provider.last_response_id == "resp_stream"

            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["input"] == "prefix\n\npayload"
            assert kwargs["previous_response_id"] == "resp_prev"
            assert kwargs["prompt_cache_key"] == "cache-key"
            assert kwargs["prompt_cache_retention"] == 600
            assert kwargs["tools"][0]["function"]["name"] == "ping"
            assert kwargs["tool_choice"] == "auto"
            assert kwargs["text"]["format"]["type"] == "json_schema"
            assert kwargs["stream"] is True

    def test_estimate_cost_gpt5(self, llm_config: LLMConfig) -> None:
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
            assert cost is not None
            assert cost > 0

    def test_estimate_cost_gpt4(self) -> None:
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
            assert cost is not None
            assert cost > 0
