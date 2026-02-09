"""
Tests for BaseLLMProvider abstract interface.
"""

from types import SimpleNamespace

import pytest

from lsm.providers.base import BaseLLMProvider


class ConcreteProvider(BaseLLMProvider):
    def __init__(self, *, send_response="ok", stream_chunks=None, fail_send=False, config=None):
        self._send_response = send_response
        self._stream_chunks = stream_chunks if stream_chunks is not None else ["ok"]
        self._fail_send = fail_send
        self._config = config or SimpleNamespace(temperature=0.3, max_tokens=128)
        self.last_send_args = None
        super().__init__()

    @property
    def config(self):
        return self._config

    @property
    def name(self) -> str:
        return "test"

    @property
    def model(self) -> str:
        return "test-model"

    def is_available(self) -> bool:
        return True

    def _send_message(self, system, user, temperature, max_tokens, **kwargs):
        if self._fail_send:
            raise RuntimeError("send failed")
        self.last_send_args = {
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        }
        return self._send_response

    def _send_streaming_message(self, system, user, temperature, max_tokens, **kwargs):
        for chunk in self._stream_chunks:
            yield chunk


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

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield "test answer"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["tag1", "tag2"]

            def _send_message(self, system, user, temperature, max_tokens, **kwargs):
                return "ok"

            def _send_streaming_message(self, system, user, temperature, max_tokens, **kwargs):
                yield "ok"

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

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield "test answer"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["tag1", "tag2"]

            def _send_message(self, system, user, temperature, max_tokens, **kwargs):
                return "ok"

            def _send_streaming_message(self, system, user, temperature, max_tokens, **kwargs):
                yield "ok"

        provider = CompleteProvider({})

        # Default implementation should return None
        assert provider.estimate_cost(1000, 500) is None

    def test_error_categorization_and_circuit_breaker(self):
        """Retry classifier and circuit breaker should be available on base class."""

        class CompleteProvider(BaseLLMProvider):
            def __init__(self, config):
                self.config = config
                super().__init__()

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
                return "ok"

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield "ok"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["tag1"]

            def _send_message(self, system, user, temperature, max_tokens, **kwargs):
                return "ok"

            def _send_streaming_message(self, system, user, temperature, max_tokens, **kwargs):
                yield "ok"

        provider = CompleteProvider({})
        provider.CIRCUIT_BREAKER_THRESHOLD = 2

        assert provider.is_retryable_error(TimeoutError("timed out")) is True
        assert provider.is_retryable_error(ValueError("bad input")) is False

        provider._record_failure(RuntimeError("temporary failure"), "test")
        provider._record_failure(RuntimeError("temporary failure"), "test")

        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            provider._with_retry(lambda: "ok", "test", max_attempts=1)

    def test_base_fallback_uses_provider_name(self):
        """Base fallback answer should be available and include provider name."""

        class CompleteProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__()

            @property
            def name(self) -> str:
                return "test-provider"

            @property
            def model(self) -> str:
                return "test-model"

            def is_available(self) -> bool:
                return True

            def rerank(self, question, candidates, k, **kwargs):
                return candidates[:k]

            def synthesize(self, question, context, mode="grounded", **kwargs):
                return "ok"

            def stream_synthesize(self, question, context, mode="grounded", **kwargs):
                yield "ok"

            def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
                return ["tag1"]

            def _send_message(self, system, user, temperature, max_tokens, **kwargs):
                return "ok"

            def _send_streaming_message(self, system, user, temperature, max_tokens, **kwargs):
                yield "ok"

        provider = CompleteProvider()
        fallback = provider._fallback_answer("Q?", "ctx")
        assert "test-provider" in fallback

    def test_concrete_rerank_and_parse(self):
        provider = ConcreteProvider(
            send_response='{"ranking":[{"index":1,"reason":"better"},{"index":0,"reason":"backup"}]}'
        )
        candidates = [
            {"text": "alpha", "metadata": {"source_path": "a.txt"}},
            {"text": "beta", "metadata": {"source_path": "b.txt"}},
        ]

        result = provider.rerank("q", candidates, 1)

        assert result == [candidates[1]]
        assert provider.last_send_args is not None
        assert provider.last_send_args["kwargs"]["json_schema_name"] == "rerank_response"

    def test_concrete_synthesize_and_fallback(self):
        provider = ConcreteProvider(send_response="answer text")
        answer = provider.synthesize("what?", "[S1] context")
        assert answer == "answer text"

        failed = ConcreteProvider(fail_send=True)
        fallback = failed.synthesize("what?", "[S1] context")
        assert "Offline mode" in fallback

    def test_concrete_stream_and_generate_tags(self):
        stream_provider = ConcreteProvider(stream_chunks=["a", "b"])
        assert list(stream_provider.stream_synthesize("q", "ctx")) == ["a", "b"]

        tag_provider = ConcreteProvider(send_response='{"tags":["Tag One","Tag Two"]}')
        tags = tag_provider.generate_tags("text", num_tags=1)
        assert tags == ["tag one"]
