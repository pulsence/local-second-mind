"""Provider caching parameter integration checks."""

from __future__ import annotations

from unittest.mock import Mock, patch

from lsm.config.models import LLMConfig
from lsm.providers.openai import OpenAIProvider
from lsm.providers.openrouter import OpenRouterProvider


def test_openai_previous_response_id_round_trip_tracks_response_id() -> None:
    config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test-key")
    response = Mock()
    response.output_text = "ok"
    response.id = "resp_new"

    with patch("lsm.providers.openai.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_client.responses.create.return_value = response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(config)
        provider.send_message(
            input="follow-up",
            instruction="system",
            previous_response_id="resp_prev",
            max_tokens=64,
        )

        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["previous_response_id"] == "resp_prev"
        assert provider.last_response_id == "resp_new"


def test_openrouter_prompt_cache_key_forwarded_to_headers() -> None:
    config = LLMConfig(provider="openrouter", model="openai/gpt-4o", api_key="test-key")
    response = Mock()
    response.choices = []
    response.model = "openai/gpt-4o"
    response.usage = {"total_tokens": 1}
    response.id = "resp_or"

    with patch("lsm.providers.openrouter.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = response
        mock_openai.return_value = mock_client

        provider = OpenRouterProvider(config)
        provider.send_message(
            input="prompt",
            instruction="rules",
            prompt_cache_key="or-cache-key",
            max_tokens=32,
        )

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_headers"]["x-prompt-cache-key"] == "or-cache-key"
        assert isinstance(kwargs["messages"][0]["content"], list)


def test_openai_streaming_and_non_streaming_cache_param_parity() -> None:
    config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test-key")
    response = Mock()
    response.output_text = "ok"
    response.id = "resp_sync"
    stream_events = [
        {"type": "response.output_text.delta", "delta": "part"},
        {"type": "response.completed", "response": {"id": "resp_stream"}},
    ]

    with patch("lsm.providers.openai.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_client.responses.create.side_effect = [response, iter(stream_events)]
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(config)
        provider.send_message(
            input="payload",
            instruction="sys",
            previous_response_id="prev-1",
            prompt_cache_key="cache-1",
            prompt_cache_retention=120,
            max_tokens=64,
        )
        list(
            provider.send_streaming_message(
                input="payload",
                instruction="sys",
                previous_response_id="prev-1",
                prompt_cache_key="cache-1",
                prompt_cache_retention=120,
                max_tokens=64,
            )
        )

        sync_kwargs = mock_client.responses.create.call_args_list[0].kwargs
        stream_kwargs = mock_client.responses.create.call_args_list[1].kwargs
        for field in (
            "previous_response_id",
            "prompt_cache_key",
            "prompt_cache_retention",
        ):
            assert sync_kwargs[field] == stream_kwargs[field]
        assert stream_kwargs["stream"] is True
