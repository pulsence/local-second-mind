"""
Natural language query decomposition into structured fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple, Any

from lsm.logging import get_logger

logger = get_logger(__name__)


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_AUTHOR_RE = re.compile(r"(?:author|by)\s*[:=]?\s*([A-Za-z][\w .,'-]{1,120})", re.IGNORECASE)
_TITLE_RE = re.compile(r"(?:title\s*[:=]\s*|\"|')([^\"']{3,180})(?:\"|')", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


@dataclass
class QueryFields:
    """
    Structured representation of a natural language query.
    """

    author: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    title: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None
    doi: Optional[str] = None
    raw_query: str = ""


def extract_fields_deterministic(query: str) -> QueryFields:
    """
    Regex-based extraction of query fields.
    """
    text = (query or "").strip()
    fields = QueryFields(raw_query=text)
    if not text:
        return fields

    doi_match = _DOI_RE.search(text)
    if doi_match:
        fields.doi = doi_match.group(0)

    author_match = _AUTHOR_RE.search(text)
    if author_match:
        fields.author = author_match.group(1).strip()

    title_match = _TITLE_RE.search(text)
    if title_match:
        fields.title = title_match.group(1).strip()

    years = [m.group(0) for m in _YEAR_RE.finditer(text)]
    if years:
        if len(years) == 1:
            fields.date_range = (years[0], years[0])
        else:
            fields.date_range = (years[0], years[-1])

    stop_words = {
        "show", "find", "paper", "papers", "document", "documents",
        "about", "with", "from", "into", "that", "this", "those", "these",
        "author", "title", "doi", "between", "after", "before",
    }
    fields.keywords = sorted(
        {
            token.lower()
            for token in _WORD_RE.findall(text)
            if token.lower() not in stop_words and not _YEAR_RE.fullmatch(token)
        }
    )[:12]

    return fields


def extract_fields_ai(query: str, llm_config: Optional[Any] = None) -> QueryFields:
    """
    AI-assisted extraction hook.

    Currently falls back to deterministic extraction and logs intent. This keeps
    behavior stable in offline/unit-test contexts while preserving extension points.
    """
    if llm_config is not None:
        logger.debug("extract_fields_ai called with llm_config; using deterministic fallback")
    return extract_fields_deterministic(query)


def decompose_query(
    query: str,
    method: str = "deterministic",
    llm_config: Optional[Any] = None,
) -> QueryFields:
    """
    Decompose a query using deterministic or AI-assisted extraction.
    """
    normalized = (method or "deterministic").strip().lower()
    if normalized == "ai":
        return extract_fields_ai(query, llm_config=llm_config)
    return extract_fields_deterministic(query)
