"""
Tests for OpenRouter provider implementation.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig, LLMProviderConfig, LLMRegistryConfig, LLMServiceConfig
from lsm.providers import create_provider, list_available_providers
from lsm.providers.openrouter import OpenRouterProvider


def _mock_chat_response(
    *,
    content: str = "ok",
    model: str | None = None,
    usage: dict | None = None,
    tool_calls: list | None = None,
):
    message = Mock()
    message.content = content
    message.tool_calls = tool_calls

    choice = Mock()
    choice.message = message

    response = Mock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    response.id = "resp_test"
    return response


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        return LLMConfig(
            provider="openrouter",
            model="openai/gpt-4o",
            api_key="test-key",
            temperature=0.2,
            max_tokens=120,
        )

    def test_provider_initialization_defaults(self, llm_config: LLMConfig) -> None:
        with patch("lsm.providers.openrouter.OpenAI") as mock_openai:
            provider = OpenRouterProvider(llm_config)

            assert provider.name == "openrouter"
            assert provider.model == "openai/gpt-4o"
            _, kwargs = mock_openai.call_args
            assert kwargs.get("base_url") == "https://openrouter.ai/api/v1"

    def test_is_available_requires_api_key(self) -> None:
        config = LLMConfig(provider="openrouter", model="openai/gpt-4o")
        with patch("lsm.providers.openrouter.OpenAI"):
            provider = OpenRouterProvider(config)
            assert provider.is_available() is False

    def test_model_selection_via_registry(self) -> None:
        registry = LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openrouter", api_key="test-key")],
            services={
                "query": LLMServiceConfig(provider="openrouter", model="openai/gpt-4o"),
            },
        )

        cfg = registry.get_query_config()
        assert cfg.provider == "openrouter"
        assert cfg.model == "openai/gpt-4o"

    def test_fallback_models_forwarded(self, llm_config: LLMConfig) -> None:
        llm_config.fallback_models = ["anthropic/claude-3.5-sonnet"]
        response = _mock_chat_response()

        with patch("lsm.providers.openrouter.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            provider = OpenRouterProvider(llm_config)
            provider._send_message("sys", "user", temperature=0.2, max_tokens=12)

            _, kwargs = mock_client.chat.completions.create.call_args
            extra_body = kwargs.get("extra_body") or {}
            assert extra_body.get("route") == "fallback"
            assert extra_body.get("models") == [
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
            ]

    def test_prompt_cache_control_enabled(self, llm_config: LLMConfig) -> None:
        response = _mock_chat_response()

        with patch("lsm.providers.openrouter.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            provider = OpenRouterProvider(llm_config)
            provider._send_message(
                "system",
                "user",
                temperature=0.2,
                max_tokens=12,
                enable_server_cache=True,
            )

            _, kwargs = mock_client.chat.completions.create.call_args
            messages = kwargs.get("messages") or []
            assert messages
            assert isinstance(messages[0].get("content"), list)
            assert messages[0]["content"][0].get("cache_control") is not None

    def test_usage_tracking_populated(self, llm_config: LLMConfig) -> None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        response = _mock_chat_response(model="openai/gpt-4o", usage=usage)

        with patch("lsm.providers.openrouter.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            provider = OpenRouterProvider(llm_config)
            provider._send_message("sys", "user", temperature=0.2, max_tokens=12)

            assert provider.last_usage is not None
            assert provider.last_usage.get("total_tokens") == 15
            assert provider.last_response_model == "openai/gpt-4o"

    def test_provider_registry_includes_openrouter(self) -> None:
        assert "openrouter" in list_available_providers()

        config = LLMConfig(provider="openrouter", model="openai/gpt-4o", api_key="test")
        with patch("lsm.providers.openrouter.OpenAI"):
            provider = create_provider(config)
            assert isinstance(provider, OpenRouterProvider)
