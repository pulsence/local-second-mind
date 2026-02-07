"""
Language detection for ingested documents.

Uses langdetect to identify document language (ISO 639-1 codes).
"""

from __future__ import annotations

from typing import Optional

from lsm.logging import get_logger

logger = get_logger(__name__)

# Minimum text length for reliable language detection
MIN_DETECT_LENGTH = 20

_LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect, DetectorFactory

    # Make detection deterministic
    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:
    logger.debug("langdetect not installed; language detection unavailable")


def is_available() -> bool:
    """Check if language detection is available.

    Returns:
        True if langdetect is installed and usable.
    """
    return _LANGDETECT_AVAILABLE


def detect_language(text: str) -> Optional[str]:
    """Detect the language of the given text.

    Args:
        text: Text content to detect language for.

    Returns:
        ISO 639-1 language code (e.g., ``'en'``, ``'de'``, ``'fr'``),
        or ``None`` if detection fails or text is too short.
    """
    if not _LANGDETECT_AVAILABLE:
        logger.warning(
            "Language detection requested but langdetect is not installed. "
            "Install with: pip install langdetect"
        )
        return None

    if not text or len(text.strip()) < MIN_DETECT_LENGTH:
        return None

    try:
        return detect(text)
    except Exception as exc:
        logger.debug("Language detection failed: %s", exc)
        return None


def detect_language_for_document(
    text: str,
    sample_size: int = 2000,
) -> Optional[str]:
    """Detect the language of a document using a text sample.

    For performance, only uses the first *sample_size* characters.
    Call this once per file (not once per chunk).

    Args:
        text: Full document text.
        sample_size: Number of characters to sample (default 2000).

    Returns:
        ISO 639-1 language code, or ``None`` if detection fails.
    """
    sample = text[:sample_size] if len(text) > sample_size else text
    return detect_language(sample)
