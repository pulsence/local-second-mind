"""
Tests for provider factory.
"""

import pytest
from unittest.mock import patch

from lsm.config.models import LLMConfig
from lsm.providers import create_provider, list_available_providers
from lsm.providers.base import BaseLLMProvider
from lsm.providers.openai import OpenAIProvider
from lsm.providers.factory import register_provider


class TestProviderFactory:
    """Tests for provider factory functions."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test")

        with patch("lsm.providers.openai.OpenAI"):
            provider = create_provider(config)

            assert isinstance(provider, OpenAIProvider)
            assert provider.name == "openai"

    def test_create_provider_case_insensitive(self):
        """Test provider name is case-insensitive."""
        config = LLMConfig(provider="OpenAI", model="gpt-5.2", api_key="test")

        with patch("lsm.providers.openai.OpenAI"):
            provider = create_provider(config)

            assert isinstance(provider, OpenAIProvider)

    def test_create_claude_alias(self):
        """Test that claude aliases to anthropic provider."""
        config = LLMConfig(provider="claude", model="claude-3-5-sonnet-20241022", api_key="test")

        with patch("lsm.providers.anthropic.Anthropic"):
            provider = create_provider(config)

            assert provider.name in {"anthropic", "claude"}

    def test_create_unknown_provider_raises_error(self):
        """Test creating unknown provider raises ValueError."""
        config = LLMConfig(provider="unknown", model="test", api_key="test")

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_provider(config)

    def test_list_available_providers(self):
        """Test listing available providers."""
        providers = list_available_providers()

        assert isinstance(providers, list)
        assert "openai" in providers

    def test_register_custom_provider(self):
        """Test registering a custom provider."""

        class CustomProvider(BaseLLMProvider):
            def __init__(self, config):
                self.config = config

            @property
            def name(self) -> str:
                return "custom"

            @property
            def model(self) -> str:
                return "custom-model"

            def is_available(self) -> bool:
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return candidates[:k]

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return "Custom answer"

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield "Custom answer"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["custom", "tag"]

        register_provider("custom", CustomProvider)

        providers = list_available_providers()
        assert "custom" in providers

        config = LLMConfig(provider="custom", model="custom-model")
        provider = create_provider(config)
        assert isinstance(provider, CustomProvider)

    def test_register_invalid_provider_raises_error(self):
        """Test registering non-BaseLLMProvider raises TypeError."""

        class NotAProvider:
            pass

        with pytest.raises(TypeError, match="must inherit from BaseLLMProvider"):
            register_provider("invalid", NotAProvider)  # type: ignore

    def test_register_duplicate_provider_overwrites(self):
        """Test registering duplicate provider name overwrites previous."""

        class Provider1(BaseLLMProvider):
            def __init__(self, config):
                self.config = config
                self.version = 1

            @property
            def name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self) -> bool:
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return []

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return ""

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield ""

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return []

        class Provider2(BaseLLMProvider):
            def __init__(self, config):
                self.config = config
                self.version = 2

            @property
            def name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self) -> bool:
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return []

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return ""

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield ""

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return []

        register_provider("test_dup", Provider1)
        register_provider("test_dup", Provider2)

        config = LLMConfig(provider="test_dup", model="test-model")
        provider = create_provider(config)

        # Should get Provider2 (version 2)
        assert provider.version == 2
