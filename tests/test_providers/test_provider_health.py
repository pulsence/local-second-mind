"""
Tests for provider health monitoring and retry behavior.
"""

from unittest.mock import Mock

from lsm.providers.base import BaseLLMProvider


class DummyProvider(BaseLLMProvider):
    def __init__(self):
        self._name = "dummy"
        self._model = "dummy-model"
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return True

    def rerank(self, question, candidates, k, **kwargs):
        return candidates[:k]

    def synthesize(self, question, context, mode="grounded", **kwargs):
        return "answer"

    def stream_synthesize(self, question, context, mode="grounded", **kwargs):
        yield "answer"

    def generate_tags(self, text, num_tags=3, existing_tags=None, **kwargs):
        return ["tag1"]

    def _send_message(self, system, user, temperature, max_tokens, **kwargs):
        return "answer"

    def _send_streaming_message(self, system, user, temperature, max_tokens, **kwargs):
        yield "answer"


def test_health_check_updates():
    provider = DummyProvider()

    provider._record_success("rerank")
    provider._record_failure(Exception("boom"), "synthesize")

    health = provider.health_check()
    stats = health["stats"]

    assert stats["success_count"] == 1
    assert stats["failure_count"] == 1
    assert stats["consecutive_failures"] == 1
    assert stats["last_error"] == "boom"


def test_retry_helper_retries():
    provider = DummyProvider()
    counter = Mock()
    counter.value = 0

    def _flaky():
        counter.value += 1
        if counter.value < 3:
            raise TimeoutError("fail")
        return "ok"

    result = provider._with_retry(_flaky, "flaky", max_attempts=3, base_delay=0.0)
    assert result == "ok"
    assert counter.value == 3
