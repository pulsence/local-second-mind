"""
Tests for per-provider pricing and cost estimation.

Verifies that each LLM provider has correct MODEL_PRICING dicts
and that estimate_cost returns accurate cost calculations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.base import BaseLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_config(**overrides) -> LLMConfig:
    """Create a minimal LLMConfig for testing."""
    defaults = {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "test-key",
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ===================================================================
# BaseLLMProvider tests
# ===================================================================

class TestBaseLLMProviderPricing:
    """Tests for base class pricing interface."""

    def test_default_estimate_cost_returns_none(self):
        """Base implementation returns None when no pricing is set."""

        class MinimalProvider(BaseLLMProvider):
            def __init__(self):
                pass

            @property
            def name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self):
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return []

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return ""

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield ""

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return []

            def send_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
                return ""

            def send_streaming_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
                yield ""

        provider = MinimalProvider()
        assert provider.estimate_cost(1000, 500) is None

    def test_default_get_model_pricing_returns_none(self):
        """Base implementation get_model_pricing returns None."""

        class MinimalProvider(BaseLLMProvider):
            def __init__(self):
                pass

            @property
            def name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self):
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return []

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return ""

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield ""

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return []

            def send_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
                return ""

            def send_streaming_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
                yield ""

        provider = MinimalProvider()
        assert provider.get_model_pricing() is None


# ===================================================================
# OpenAI Provider pricing tests
# ===================================================================

class TestOpenAIPricing:
    """Tests for OpenAI provider pricing."""

    @pytest.fixture
    def provider(self):
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-4o")
        with patch("lsm.providers.openai.OpenAI"):
            return OpenAIProvider(config)

    def test_model_pricing_dict_exists(self, provider):
        """OpenAI provider should have a MODEL_PRICING dict."""
        from lsm.providers.openai import OpenAIProvider
        assert hasattr(OpenAIProvider, "MODEL_PRICING")
        assert isinstance(OpenAIProvider.MODEL_PRICING, dict)
        assert len(OpenAIProvider.MODEL_PRICING) > 0

    def test_model_pricing_contains_key_models(self, provider):
        """MODEL_PRICING should contain all major OpenAI models."""
        from lsm.providers.openai import OpenAIProvider
        expected_models = [
            "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o3", "o3-mini", "o4-mini",
        ]
        for model in expected_models:
            assert model in OpenAIProvider.MODEL_PRICING, (
                f"{model} missing from OpenAI MODEL_PRICING"
            )

    def test_pricing_dict_structure(self, provider):
        """Each pricing entry should have 'input' and 'output' keys."""
        from lsm.providers.openai import OpenAIProvider
        for model_name, pricing in OpenAIProvider.MODEL_PRICING.items():
            assert "input" in pricing, f"{model_name} missing 'input' key"
            assert "output" in pricing, f"{model_name} missing 'output' key"
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_estimate_cost_known_model(self):
        """estimate_cost should return a float for known models."""
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-4o")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_unknown_model(self):
        """estimate_cost should return None for unknown models."""
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-99-nonexistent")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
        cost = provider.estimate_cost(1000, 500)
        assert cost is None

    def test_estimate_cost_calculation_accuracy(self):
        """estimate_cost math should be correct: (tokens / 1M) * rate."""
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-4o")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
        pricing = OpenAIProvider.MODEL_PRICING["gpt-4o"]
        input_tokens = 500_000
        output_tokens = 100_000
        expected = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )
        actual = provider.estimate_cost(input_tokens, output_tokens)
        assert actual == pytest.approx(expected)

    def test_estimate_cost_zero_tokens(self):
        """estimate_cost with 0 tokens should return 0."""
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-4o")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
        cost = provider.estimate_cost(0, 0)
        assert cost == 0.0

    def test_get_model_pricing_known_model(self):
        """get_model_pricing should return pricing for current model."""
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-4o")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
        pricing = provider.get_model_pricing()
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing

    def test_get_model_pricing_unknown_model(self):
        """get_model_pricing should return None for unknown model."""
        from lsm.providers.openai import OpenAIProvider
        config = _make_llm_config(provider="openai", model="gpt-99-nonexistent")
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
        pricing = provider.get_model_pricing()
        assert pricing is None


# ===================================================================
# Anthropic Provider pricing tests
# ===================================================================

class TestAnthropicPricing:
    """Tests for Anthropic provider pricing."""

    @pytest.fixture
    def provider(self):
        from lsm.providers.anthropic import AnthropicProvider
        config = _make_llm_config(
            provider="anthropic", model="claude-sonnet-4.5"
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return AnthropicProvider(config)

    def test_model_pricing_dict_exists(self, provider):
        """Anthropic provider should have a MODEL_PRICING dict."""
        from lsm.providers.anthropic import AnthropicProvider
        assert hasattr(AnthropicProvider, "MODEL_PRICING")
        assert isinstance(AnthropicProvider.MODEL_PRICING, dict)
        assert len(AnthropicProvider.MODEL_PRICING) > 0

    def test_model_pricing_contains_key_models(self, provider):
        """MODEL_PRICING should contain all major Anthropic models."""
        from lsm.providers.anthropic import AnthropicProvider
        expected_models = [
            "claude-opus-4", "claude-sonnet-4",
            "claude-sonnet-4.5", "claude-haiku-4.5",
        ]
        for model in expected_models:
            assert model in AnthropicProvider.MODEL_PRICING, (
                f"{model} missing from Anthropic MODEL_PRICING"
            )

    def test_pricing_dict_structure(self, provider):
        """Each pricing entry should have 'input' and 'output' keys."""
        from lsm.providers.anthropic import AnthropicProvider
        for model_name, pricing in AnthropicProvider.MODEL_PRICING.items():
            assert "input" in pricing, f"{model_name} missing 'input' key"
            assert "output" in pricing, f"{model_name} missing 'output' key"
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_estimate_cost_known_model(self):
        """estimate_cost should return a float for known models."""
        from lsm.providers.anthropic import AnthropicProvider
        config = _make_llm_config(
            provider="anthropic", model="claude-sonnet-4.5"
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(config)
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_unknown_model(self):
        """estimate_cost should return None for unknown models."""
        from lsm.providers.anthropic import AnthropicProvider
        config = _make_llm_config(
            provider="anthropic", model="claude-99-nonexistent"
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(config)
        cost = provider.estimate_cost(1000, 500)
        assert cost is None

    def test_estimate_cost_calculation_accuracy(self):
        """estimate_cost math should be correct."""
        from lsm.providers.anthropic import AnthropicProvider
        config = _make_llm_config(
            provider="anthropic", model="claude-sonnet-4.5"
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(config)
        pricing = AnthropicProvider.MODEL_PRICING["claude-sonnet-4.5"]
        input_tokens = 500_000
        output_tokens = 100_000
        expected = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )
        actual = provider.estimate_cost(input_tokens, output_tokens)
        assert actual == pytest.approx(expected)

    def test_get_model_pricing_known_model(self):
        """get_model_pricing should return pricing for current model."""
        from lsm.providers.anthropic import AnthropicProvider
        config = _make_llm_config(
            provider="anthropic", model="claude-sonnet-4.5"
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider(config)
        pricing = provider.get_model_pricing()
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing


# ===================================================================
# Gemini Provider pricing tests
# ===================================================================

class TestGeminiPricing:
    """Tests for Gemini provider pricing."""

    @pytest.fixture
    def provider(self):
        from lsm.providers.gemini import GeminiProvider
        config = _make_llm_config(
            provider="gemini", model="gemini-2.5-flash"
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            return GeminiProvider(config)

    def test_model_pricing_dict_exists(self, provider):
        """Gemini provider should have a MODEL_PRICING dict."""
        from lsm.providers.gemini import GeminiProvider
        assert hasattr(GeminiProvider, "MODEL_PRICING")
        assert isinstance(GeminiProvider.MODEL_PRICING, dict)
        assert len(GeminiProvider.MODEL_PRICING) > 0

    def test_model_pricing_contains_key_models(self, provider):
        """MODEL_PRICING should contain all major Gemini models."""
        from lsm.providers.gemini import GeminiProvider
        expected_models = [
            "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]
        for model in expected_models:
            assert model in GeminiProvider.MODEL_PRICING, (
                f"{model} missing from Gemini MODEL_PRICING"
            )

    def test_pricing_dict_structure(self, provider):
        """Each pricing entry should have 'input' and 'output' keys."""
        from lsm.providers.gemini import GeminiProvider
        for model_name, pricing in GeminiProvider.MODEL_PRICING.items():
            assert "input" in pricing, f"{model_name} missing 'input' key"
            assert "output" in pricing, f"{model_name} missing 'output' key"
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_estimate_cost_known_model(self):
        """estimate_cost should return a float for known models."""
        from lsm.providers.gemini import GeminiProvider
        config = _make_llm_config(
            provider="gemini", model="gemini-2.5-flash"
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider(config)
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0

    def test_estimate_cost_unknown_model(self):
        """estimate_cost should return None for unknown models."""
        from lsm.providers.gemini import GeminiProvider
        config = _make_llm_config(
            provider="gemini", model="gemini-99-nonexistent"
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider(config)
        cost = provider.estimate_cost(1000, 500)
        assert cost is None

    def test_estimate_cost_calculation_accuracy(self):
        """estimate_cost math should be correct."""
        from lsm.providers.gemini import GeminiProvider
        config = _make_llm_config(
            provider="gemini", model="gemini-2.5-flash"
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider(config)
        pricing = GeminiProvider.MODEL_PRICING["gemini-2.5-flash"]
        input_tokens = 500_000
        output_tokens = 100_000
        expected = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )
        actual = provider.estimate_cost(input_tokens, output_tokens)
        assert actual == pytest.approx(expected)

    def test_get_model_pricing_known_model(self):
        """get_model_pricing should return pricing for current model."""
        from lsm.providers.gemini import GeminiProvider
        config = _make_llm_config(
            provider="gemini", model="gemini-2.5-flash"
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiProvider(config)
        pricing = provider.get_model_pricing()
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing


# ===================================================================
# Local Provider pricing tests
# ===================================================================

class TestLocalProviderPricing:
    """Tests for Local/Ollama provider pricing."""

    @pytest.fixture
    def provider(self):
        from lsm.providers.local import LocalProvider
        config = _make_llm_config(
            provider="local", model="llama3"
        )
        return LocalProvider(config)

    def test_estimate_cost_always_zero(self, provider):
        """Local provider should always return 0.0 (free)."""
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        assert cost == 0.0

    def test_estimate_cost_any_model_returns_zero(self):
        """Any local model should return 0.0."""
        from lsm.providers.local import LocalProvider
        for model_name in ["llama3", "mistral", "phi-3", "custom-model"]:
            config = _make_llm_config(provider="local", model=model_name)
            provider = LocalProvider(config)
            assert provider.estimate_cost(1000, 500) == 0.0

    def test_get_model_pricing_returns_zero_dict(self, provider):
        """Local provider get_model_pricing returns zero-cost dict."""
        pricing = provider.get_model_pricing()
        assert pricing is not None
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0


# ===================================================================
# Cross-provider consistency tests
# ===================================================================

class TestPricingConsistency:
    """Cross-provider pricing consistency checks."""

    def test_all_providers_have_model_pricing(self):
        """All provider classes should have MODEL_PRICING class attribute."""
        from lsm.providers.openai import OpenAIProvider
        from lsm.providers.anthropic import AnthropicProvider
        from lsm.providers.gemini import GeminiProvider

        for cls in [OpenAIProvider, AnthropicProvider, GeminiProvider]:
            assert hasattr(cls, "MODEL_PRICING"), (
                f"{cls.__name__} missing MODEL_PRICING"
            )

    def test_pricing_values_reasonable(self):
        """All pricing values should be in reasonable range (0 to 200)."""
        from lsm.providers.openai import OpenAIProvider
        from lsm.providers.anthropic import AnthropicProvider
        from lsm.providers.gemini import GeminiProvider

        for cls in [OpenAIProvider, AnthropicProvider, GeminiProvider]:
            for model_name, pricing in cls.MODEL_PRICING.items():
                assert 0 <= pricing["input"] <= 200, (
                    f"{cls.__name__} {model_name} input price {pricing['input']} out of range"
                )
                assert 0 <= pricing["output"] <= 1000, (
                    f"{cls.__name__} {model_name} output price {pricing['output']} out of range"
                )

    def test_output_price_at_least_input_price(self):
        """Output price should be >= input price for all models."""
        from lsm.providers.openai import OpenAIProvider
        from lsm.providers.anthropic import AnthropicProvider
        from lsm.providers.gemini import GeminiProvider

        for cls in [OpenAIProvider, AnthropicProvider, GeminiProvider]:
            for model_name, pricing in cls.MODEL_PRICING.items():
                assert pricing["output"] >= pricing["input"], (
                    f"{cls.__name__} {model_name}: output (${pricing['output']}) "
                    f"< input (${pricing['input']})"
                )
