"""
LLM-based machine translation for ingested chunks.

Uses the configured LLM provider to translate text chunks to a target
language, enabling cross-language search on multilingual corpora.
"""

from __future__ import annotations

from typing import Optional

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from lsm.providers.factory import create_provider

logger = get_logger(__name__)


def translate_chunk(
    text: str,
    target_lang: str,
    llm_config: LLMConfig,
    source_lang: Optional[str] = None,
    max_retries: int = 1,
) -> str:
    """Translate a text chunk to the target language using an LLM.

    Args:
        text: Text content to translate.
        target_lang: Target language code (ISO 639-1, e.g. ``'en'``).
        llm_config: LLM configuration for the translation provider.
        source_lang: Optional source language code. If ``None``, the LLM
            auto-detects.
        max_retries: Maximum retry attempts on failure (default 1).

    Returns:
        Translated text.  Returns original text if translation fails
        or if text is already in the target language.
    """
    if source_lang and source_lang == target_lang:
        return text

    if not text or not text.strip():
        return text

    provider = create_provider(llm_config)

    source_desc = f"from {source_lang} " if source_lang else ""
    prompt = (
        f"Translate the following text {source_desc}to {target_lang}. "
        "Return ONLY the translated text with no explanations, "
        "preamble, or formatting. Preserve the original structure "
        "(paragraphs, headings, lists).\n\n"
        f"{text}"
    )

    for attempt in range(max_retries + 1):
        try:
            translated = provider.send_message(input=prompt)
            if translated and translated.strip():
                if attempt > 0:
                    logger.info("Translation succeeded on retry %d", attempt)
                return translated.strip()
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    "Translation failed (attempt %d), retrying: %s",
                    attempt + 1,
                    exc,
                )
            else:
                logger.error(
                    "Translation failed after %d attempts: %s",
                    max_retries + 1,
                    exc,
                )

    logger.warning("Translation failed; returning original text")
    return text
