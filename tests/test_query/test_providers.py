"""
Tests for query provider module.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from lsm.config.models import LLMConfig
from lsm.query.providers import create_provider, list_available_providers
from lsm.query.providers.base import BaseLLMProvider
from lsm.query.providers.openai import OpenAIProvider
from lsm.query.providers.factory import register_provider


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore


class TestOpenAIProvider:
    """Tests for OpenAIProvider implementation."""

    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=2000,
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock_client = Mock()
        mock_client.responses = Mock()
        mock_client.models = Mock()
        return mock_client

    def test_provider_initialization(self, llm_config):
        """Test OpenAI provider initialization."""
        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(llm_config)

            assert provider.name == "openai"
            assert provider.model == "gpt-5.2"
            mock_openai.assert_called_once_with(api_key="test-api-key")

    def test_provider_initialization_without_api_key(self):
        """Test OpenAI provider uses env var when no API key in config."""
        config = LLMConfig(provider="openai", model="gpt-5.2")

        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(config)

            assert provider.name == "openai"
            mock_openai.assert_called_once_with()  # No api_key parameter

    def test_is_available_with_api_key(self, llm_config):
        """Test is_available returns True when API key is set."""
        with patch("lsm.query.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            assert provider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test is_available returns False when no API key."""
        config = LLMConfig(provider="openai", model="gpt-5.2", api_key=None)

        with patch("lsm.query.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
            assert provider.is_available() is False

    def test_rerank_with_valid_candidates(self, llm_config):
        """Test rerank with valid candidates."""
        candidates = [
            {
                "text": "Python is a programming language",
                "metadata": {"source_path": "/docs/python.md", "chunk_index": 0},
                "distance": 0.2,
            },
            {
                "text": "Java is also a programming language",
                "metadata": {"source_path": "/docs/java.md", "chunk_index": 0},
                "distance": 0.3,
            },
            {
                "text": "Python has great libraries",
                "metadata": {"source_path": "/docs/python.md", "chunk_index": 1},
                "distance": 0.25,
            },
        ]

        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "ranking": [
                {"index": 0, "reason": "Most relevant"},
                {"index": 2, "reason": "Also relevant"},
            ]
        })

        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            result = provider.rerank("What is Python?", candidates, k=2)

            assert len(result) == 2
            assert result[0]["text"] == "Python is a programming language"
            assert result[1]["text"] == "Python has great libraries"

    def test_rerank_with_empty_candidates(self, llm_config):
        """Test rerank with empty candidate list."""
        with patch("lsm.query.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            result = provider.rerank("What is Python?", [], k=5)

            assert result == []

    def test_rerank_fallback_on_api_error(self, llm_config):
        """Test rerank falls back to original order on API error."""
        candidates = [
            {"text": "First", "metadata": {}, "distance": 0.1},
            {"text": "Second", "metadata": {}, "distance": 0.2},
            {"text": "Third", "metadata": {}, "distance": 0.3},
        ]

        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            result = provider.rerank("Question?", candidates, k=2)

            # Should return first k candidates
            assert len(result) == 2
            assert result[0]["text"] == "First"
            assert result[1]["text"] == "Second"

    def test_rerank_fallback_on_invalid_json(self, llm_config):
        """Test rerank falls back on invalid JSON response."""
        candidates = [
            {"text": "First", "metadata": {}, "distance": 0.1},
            {"text": "Second", "metadata": {}, "distance": 0.2},
        ]

        mock_response = Mock()
        mock_response.output_text = "Not valid JSON"

        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            result = provider.rerank("Question?", candidates, k=2)

            assert len(result) == 2
            assert result[0]["text"] == "First"

    def test_synthesize_grounded_mode(self, llm_config):
        """Test synthesize in grounded mode."""
        mock_response = Mock()
        mock_response.output_text = "Python is a programming language [S1]."

        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "What is Python?",
                "[S1] Python is a high-level language",
                mode="grounded",
            )

            assert "[S1]" in answer
            mock_client.responses.create.assert_called_once()

    def test_synthesize_insight_mode(self, llm_config):
        """Test synthesize in insight mode."""
        mock_response = Mock()
        mock_response.output_text = "Analysis of Python ecosystem [S1] [S2]."

        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "Analyze Python",
                "[S1] Python docs\n[S2] Python guide",
                mode="insight",
            )

            assert "Analysis" in answer
            mock_client.responses.create.assert_called_once()

    def test_synthesize_fallback_on_error(self, llm_config):
        """Test synthesize returns fallback on API error."""
        with patch("lsm.query.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "What is Python?",
                "[S1] Context here",
                mode="grounded",
            )

            assert "Offline mode" in answer
            assert "What is Python?" in answer

    def test_estimate_cost_gpt5(self, llm_config):
        """Test cost estimation for GPT-5 model."""
        with patch("lsm.query.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

            assert cost is not None
            assert cost > 0
            assert isinstance(cost, float)

    def test_estimate_cost_gpt4(self):
        """Test cost estimation for GPT-4 model."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")

        with patch("lsm.query.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

            assert cost is not None
            assert cost > 0


class TestProviderFactory:
    """Tests for provider factory."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test")

        with patch("lsm.query.providers.openai.OpenAI"):
            provider = create_provider(config)

            assert isinstance(provider, OpenAIProvider)
            assert provider.name == "openai"

    def test_create_provider_case_insensitive(self):
        """Test provider name is case-insensitive."""
        config = LLMConfig(provider="OpenAI", model="gpt-5.2", api_key="test")

        with patch("lsm.query.providers.openai.OpenAI"):
            provider = create_provider(config)

            assert isinstance(provider, OpenAIProvider)

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
