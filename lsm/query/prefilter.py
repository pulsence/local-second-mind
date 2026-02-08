"""
Metadata prefiltering utilities for query retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from lsm.logging import get_logger
from lsm.query.decomposition import decompose_query

logger = get_logger(__name__)


_TAG_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def _normalize_values(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        return [str(values)]
    if isinstance(values, dict):
        return [str(v) for v in values.values()]
    if isinstance(values, Iterable):
        return [str(v) for v in values if v is not None]
    return []


def extract_tags_from_prompt(
    query: str,
    llm_config: Optional[Any] = None,
) -> List[str]:
    """
    Extract likely tags from a user prompt.

    Uses deterministic token extraction first. If llm_config is provided,
    this function currently keeps deterministic behavior and logs that the
    AI hook is available for future enhancement.
    """
    text = (query or "").strip().lower()
    if not text:
        return []

    if llm_config is not None:
        try:
            fields = decompose_query(query, method="ai", llm_config=llm_config)
            if fields.keywords:
                return sorted({str(k).strip().lower() for k in fields.keywords if str(k).strip()})[:8]
        except Exception as exc:
            logger.debug(f"AI tag extraction failed; falling back to deterministic tokens: {exc}")

    tokens = [m.group(0) for m in _TAG_WORD_RE.finditer(text)]
    stop_words = {
        "what", "when", "where", "which", "with", "from", "into", "about",
        "that", "this", "those", "these", "show", "give", "find", "please",
        "does", "have", "been", "were", "will", "would", "could", "should",
        "article", "paper", "papers", "document", "documents",
    }
    tags = sorted({token for token in tokens if token not in stop_words})[:8]

    if llm_config is not None:
        logger.debug("extract_tags_from_prompt used deterministic fallback after AI attempt")

    return tags


def prefilter_by_metadata(
    query: str,
    available_metadata: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build a best-effort metadata where-filter from natural language query.

    Supports deterministic extraction for author/year/title hints, and tag-based
    matching across ai_tags, user_tags, root_tags, folder_tags, and content_type.
    """
    text = (query or "").strip()
    if not text:
        return {}

    query_lower = text.lower()
    metadata = available_metadata or {}
    where: Dict[str, Any] = {}

    fields = decompose_query(
        text,
        method="ai" if llm_config is not None else "deterministic",
        llm_config=llm_config,
    )

    if fields.author:
        where["author"] = fields.author
    if fields.date_range:
        where["year"] = fields.date_range[0]

    content_types = _normalize_values(metadata.get("content_type"))
    for value in content_types:
        if value and value.lower() in query_lower:
            where["content_type"] = value
            break

    # Tag-aware matching across all configured tag metadata fields.
    query_tags = extract_tags_from_prompt(text, llm_config=llm_config) or list(fields.keywords)
    for tag_field in ("ai_tags", "user_tags", "root_tags", "folder_tags"):
        known_tags = [v.lower() for v in _normalize_values(metadata.get(tag_field))]
        for tag in query_tags:
            if tag.lower() in known_tags:
                where[tag_field] = tag
                break

    # Title matching if provided by metadata inventory.
    if fields.title:
        where["title"] = fields.title
    else:
        titles = _normalize_values(metadata.get("title"))
        for title in titles:
            if title and title.lower() in query_lower:
                where["title"] = title
                break

    return where
