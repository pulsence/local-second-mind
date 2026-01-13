"""
Tests for BaseLLMProvider abstract interface.
"""

import pytest

from lsm.providers.base import BaseLLMProvider


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore

    def test_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""

        class IncompleteProvider(BaseLLMProvider):
            """Provider missing required methods."""
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore

    def test_string_representation(self):
        """Test __str__ and __repr__ methods."""

        class CompleteProvider(BaseLLMProvider):
            def __init__(self, config):
                self.config = config

            @property
            def name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self) -> bool:
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return candidates[:k]

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return "test answer"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["tag1", "tag2"]

        provider = CompleteProvider({})

        assert str(provider) == "test/test-model"
        assert repr(provider) == "CompleteProvider(model='test-model')"

    def test_estimate_cost_default_implementation(self):
        """Test that estimate_cost returns None by default."""

        class CompleteProvider(BaseLLMProvider):
            def __init__(self, config):
                self.config = config

            @property
            def name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self) -> bool:
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return candidates[:k]

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return "test answer"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["tag1", "tag2"]

        provider = CompleteProvider({})

        # Default implementation should return None
        assert provider.estimate_cost(1000, 500) is None
