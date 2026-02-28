"""Tests for BaseLLMProvider abstract interface and health utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lsm.providers.base import BaseLLMProvider


class ConcreteProvider(BaseLLMProvider):
    def __init__(self, *, fail_send: bool = False, stream_chunks: list[str] | None = None):
        self._fail_send = fail_send
        self._stream_chunks = list(stream_chunks or ["chunk"])
        super().__init__()

    @property
    def name(self) -> str:
        return "test"

    @property
    def model(self) -> str:
        return "test-model"

    def is_available(self) -> bool:
        return True

    def send_message(
        self,
        input,
        instruction=None,
        prompt=None,
        temperature=None,
        max_tokens=4096,
        previous_response_id=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        **kwargs,
    ):
        _ = (
            input,
            instruction,
            prompt,
            temperature,
            max_tokens,
            previous_response_id,
            prompt_cache_key,
            prompt_cache_retention,
            kwargs,
        )
        if self._fail_send:
            raise RuntimeError("boom")
        return "ok"

    def send_streaming_message(
        self,
        input,
        instruction=None,
        prompt=None,
        temperature=None,
        max_tokens=4096,
        previous_response_id=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        **kwargs,
    ):
        _ = (
            input,
            instruction,
            prompt,
            temperature,
            max_tokens,
            previous_response_id,
            prompt_cache_key,
            prompt_cache_retention,
            kwargs,
        )
        if self._fail_send:
            raise RuntimeError("boom")
        yield from self._stream_chunks


class TestBaseLLMProvider:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_abstract_methods_required(self):
        class MissingSend(BaseLLMProvider):
            @property
            def name(self) -> str:
                return "x"

            @property
            def model(self) -> str:
                return "m"

            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            MissingSend()

    def test_string_representation(self):
        provider = ConcreteProvider()
        assert str(provider) == "test/test-model"

    def test_estimate_cost_default_implementation(self):
        provider = ConcreteProvider()
        assert provider.get_model_pricing() is None
        assert provider.estimate_cost(1000, 500) is None

    def test_error_categorization_and_circuit_breaker(self):
        provider = ConcreteProvider()

        assert provider.categorize_error(ValueError("bad")) == "fatal"
        assert provider.categorize_error(TimeoutError("t")) == "retryable"

        for _ in range(provider.CIRCUIT_BREAKER_THRESHOLD):
            provider._record_failure(RuntimeError("x"), "send_message")

        health = provider.health_check()
        assert health["stats"]["consecutive_failures"] >= provider.CIRCUIT_BREAKER_THRESHOLD
        assert provider._is_circuit_open() is True
        assert health["stats"]["circuit_open_until"] is not None

    def test_retry_helper_retries_once_then_succeeds(self):
        provider = ConcreteProvider()
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("temporary")
            return "ok"

        with patch("lsm.providers.base.time.sleep"):
            result = provider._with_retry(
                flaky,
                "send_message",
                max_attempts=2,
                retry_on=lambda exc: isinstance(exc, TimeoutError),
            )

        assert result == "ok"
        assert calls["n"] == 2

    def test_retry_helper_raises_final_failure(self):
        provider = ConcreteProvider()
        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise RuntimeError("nope")

        with patch("lsm.providers.base.time.sleep"):
            with pytest.raises(RuntimeError):
                provider._with_retry(always_fail, "send_message", max_attempts=1)

        assert calls["n"] == 1
