"""Tests for Local (Ollama) provider implementation."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.providers.local import LocalProvider


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        provider="local",
        model="llama2",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2000,
    )


def _mock_response(payload: str) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": payload}}
    return response


def test_send_message_maps_instruction_to_system_role(llm_config: LLMConfig) -> None:
    with patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")

        provider = LocalProvider(llm_config)
        provider.send_message(
            input="hello",
            instruction="system-rule",
            prompt="prefix",
            max_tokens=32,
        )

        kwargs = mock_post.call_args.kwargs
        payload = kwargs["json"]
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system-rule"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "prefix\n\nhello"


def test_send_message_logs_unsupported_cache_params(llm_config: LLMConfig, caplog) -> None:
    import lsm.providers.local as local_module

    local_module._UNSUPPORTED_PARAM_TRACKER._unsupported.clear()
    with patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")

        provider = LocalProvider(llm_config)
        with caplog.at_level("DEBUG", logger="lsm.providers.local"):
            provider.send_message(
                input="hello",
                instruction="system-rule",
                previous_response_id="prev",
                prompt_cache_key="cache-key",
                prompt_cache_retention=60,
                max_tokens=32,
            )

    assert "does not support 'previous_response_id'" in caplog.text
    assert "does not support 'prompt_cache_key'" in caplog.text
    assert "does not support 'prompt_cache_retention'" in caplog.text


def test_send_message_returns_chat_content(llm_config: LLMConfig) -> None:
    tags_payload = json.dumps({"tags": ["local", "model"]})

    with patch("lsm.providers.local.requests.post") as mock_post:
        mock_post.return_value = _mock_response(tags_payload)

        provider = LocalProvider(llm_config)
        response = provider.send_message("Local model", instruction="tag", max_tokens=64)

        assert response == tags_payload
