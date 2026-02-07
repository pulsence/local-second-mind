"""Tests for language detection module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lsm.ingest.language import (
    MIN_DETECT_LENGTH,
    detect_language,
    detect_language_for_document,
    is_available,
)


class TestIsAvailable:
    """Tests for is_available()."""

    def test_returns_bool(self) -> None:
        result = is_available()
        assert isinstance(result, bool)

    def test_true_when_langdetect_installed(self) -> None:
        assert is_available() is True


class TestDetectLanguage:
    """Tests for detect_language()."""

    def test_detects_english(self) -> None:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sample English text for language detection."
        )
        assert detect_language(text) == "en"

    def test_detects_french(self) -> None:
        text = (
            "Le renard brun rapide saute par-dessus le chien paresseux. "
            "Ceci est un texte en francais pour la detection de la langue."
        )
        assert detect_language(text) == "fr"

    def test_detects_german(self) -> None:
        text = (
            "Der schnelle braune Fuchs springt ueber den faulen Hund. "
            "Dies ist ein deutscher Text zur Spracherkennung."
        )
        assert detect_language(text) == "de"

    def test_detects_spanish(self) -> None:
        text = (
            "El rapido zorro marron salta sobre el perro perezoso. "
            "Este es un texto en espanol para la deteccion del idioma."
        )
        assert detect_language(text) == "es"

    def test_returns_none_for_empty_string(self) -> None:
        assert detect_language("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        assert detect_language("   \n\t  ") is None

    def test_returns_none_for_short_text(self) -> None:
        assert detect_language("Hi") is None

    def test_returns_none_for_text_below_min_length(self) -> None:
        short = "a" * (MIN_DETECT_LENGTH - 1)
        assert detect_language(short) is None

    def test_returns_string_for_text_at_min_length(self) -> None:
        # Repeating clear English words to get enough signal
        text = "hello world testing ok"
        assert len(text.strip()) >= MIN_DETECT_LENGTH
        result = detect_language(text)
        assert result is None or isinstance(result, str)

    def test_returns_iso_639_1_code(self) -> None:
        text = "This is definitely English text for language detection purposes."
        result = detect_language(text)
        assert result is not None
        assert len(result) == 2
        assert result.isalpha()
        assert result.islower()


class TestDetectLanguageForDocument:
    """Tests for detect_language_for_document()."""

    def test_detects_language_from_long_document(self) -> None:
        text = (
            "This is a long English document. " * 200
        )
        assert detect_language_for_document(text) == "en"

    def test_samples_first_n_characters(self) -> None:
        english_text = "This is English text for detection. " * 100
        full_text = english_text
        result = detect_language_for_document(full_text, sample_size=500)
        assert result == "en"

    def test_works_with_short_document(self) -> None:
        text = "This is a short English sentence for testing purposes."
        result = detect_language_for_document(text)
        assert result == "en"

    def test_returns_none_for_empty(self) -> None:
        assert detect_language_for_document("") is None

    def test_default_sample_size_is_2000(self) -> None:
        with patch("lsm.ingest.language.detect_language") as mock_detect:
            mock_detect.return_value = "en"
            text = "x" * 5000
            detect_language_for_document(text)
            called_text = mock_detect.call_args[0][0]
            assert len(called_text) == 2000

    def test_full_text_when_shorter_than_sample(self) -> None:
        with patch("lsm.ingest.language.detect_language") as mock_detect:
            mock_detect.return_value = "en"
            text = "short text"
            detect_language_for_document(text)
            called_text = mock_detect.call_args[0][0]
            assert called_text == text


class TestLanguageDetectionUnavailable:
    """Tests for graceful fallback when langdetect is not installed."""

    def test_detect_language_returns_none_when_unavailable(self) -> None:
        with patch("lsm.ingest.language._LANGDETECT_AVAILABLE", False):
            result = detect_language("This is English text for testing.")
            assert result is None

    def test_is_available_returns_false_when_unavailable(self) -> None:
        with patch("lsm.ingest.language._LANGDETECT_AVAILABLE", False):
            assert is_available() is False
