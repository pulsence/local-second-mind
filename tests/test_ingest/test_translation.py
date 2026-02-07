"""Tests for LLM-based translation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lsm.config.models import LLMConfig
from lsm.ingest.translation import translate_chunk


def _make_llm_config() -> LLMConfig:
    """Create a minimal LLMConfig for testing."""
    return LLMConfig(
        provider="openai",
        model="gpt-5-nano",
        api_key="test-key",
    )


class TestTranslateChunk:
    """Tests for translate_chunk()."""

    @patch("lsm.ingest.translation.create_provider")
    def test_calls_provider_synthesize(self, mock_create: MagicMock) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.return_value = "Translated text"
        mock_create.return_value = mock_provider

        result = translate_chunk(
            text="Texte en francais",
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="fr",
        )

        assert result == "Translated text"
        mock_provider.synthesize.assert_called_once()

    @patch("lsm.ingest.translation.create_provider")
    def test_includes_target_lang_in_prompt(self, mock_create: MagicMock) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.return_value = "Translated"
        mock_create.return_value = mock_provider

        translate_chunk(
            text="Hallo Welt",
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="de",
        )

        call_args = mock_provider.synthesize.call_args
        prompt = call_args[1].get("question", call_args[0][0] if call_args[0] else "")
        assert "en" in prompt

    @patch("lsm.ingest.translation.create_provider")
    def test_includes_source_lang_in_prompt_when_provided(
        self, mock_create: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.return_value = "Translated"
        mock_create.return_value = mock_provider

        translate_chunk(
            text="Hallo Welt",
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="de",
        )

        call_args = mock_provider.synthesize.call_args
        prompt = call_args[1].get("question", call_args[0][0] if call_args[0] else "")
        assert "de" in prompt


class TestTranslateChunkSameLanguage:
    """Tests for same-language short-circuit."""

    def test_returns_original_when_source_equals_target(self) -> None:
        original = "Already in English"
        result = translate_chunk(
            text=original,
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="en",
        )
        assert result == original


class TestTranslateChunkEmpty:
    """Tests for empty text handling."""

    def test_returns_empty_for_empty_string(self) -> None:
        result = translate_chunk(
            text="",
            target_lang="en",
            llm_config=_make_llm_config(),
        )
        assert result == ""

    def test_returns_whitespace_for_whitespace_only(self) -> None:
        result = translate_chunk(
            text="   ",
            target_lang="en",
            llm_config=_make_llm_config(),
        )
        assert result == "   "


class TestTranslateChunkFailure:
    """Tests for error handling."""

    @patch("lsm.ingest.translation.create_provider")
    def test_returns_original_on_provider_exception(
        self, mock_create: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.side_effect = RuntimeError("API error")
        mock_create.return_value = mock_provider

        original = "Texte original en francais"
        result = translate_chunk(
            text=original,
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="fr",
        )
        assert result == original

    @patch("lsm.ingest.translation.create_provider")
    def test_returns_original_when_provider_returns_empty(
        self, mock_create: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.return_value = ""
        mock_create.return_value = mock_provider

        original = "Texte en francais"
        result = translate_chunk(
            text=original,
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="fr",
        )
        assert result == original


class TestTranslateChunkRetry:
    """Tests for retry logic."""

    @patch("lsm.ingest.translation.create_provider")
    def test_retries_on_failure_then_succeeds(
        self, mock_create: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.side_effect = [
            RuntimeError("Temporary error"),
            "Translated text",
        ]
        mock_create.return_value = mock_provider

        result = translate_chunk(
            text="Texte en francais",
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="fr",
            max_retries=1,
        )
        assert result == "Translated text"
        assert mock_provider.synthesize.call_count == 2

    @patch("lsm.ingest.translation.create_provider")
    def test_no_retry_when_max_retries_zero(
        self, mock_create: MagicMock,
    ) -> None:
        mock_provider = MagicMock()
        mock_provider.synthesize.side_effect = RuntimeError("API error")
        mock_create.return_value = mock_provider

        original = "Texte en francais"
        result = translate_chunk(
            text=original,
            target_lang="en",
            llm_config=_make_llm_config(),
            source_lang="fr",
            max_retries=0,
        )
        assert result == original
        assert mock_provider.synthesize.call_count == 1
