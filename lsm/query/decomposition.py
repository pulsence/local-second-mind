"""
Natural language query decomposition into structured fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import List, Optional, Tuple, Any, Dict

from lsm.logging import get_logger
from lsm.providers import create_provider

logger = get_logger(__name__)


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_AUTHOR_RE = re.compile(r"(?:author|by)\s*[:=]?\s*([A-Za-z][\w .,'-]{1,120})", re.IGNORECASE)
_TITLE_RE = re.compile(r"(?:title\s*[:=]\s*|\"|')([^\"']{3,180})(?:\"|')", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


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
    AI-assisted extraction of query fields with deterministic fallback.
    """
    deterministic = extract_fields_deterministic(query)
    if llm_config is None:
        return deterministic

    try:
        provider = create_provider(llm_config)
        prompt = (
            "Extract structured search fields from the user query and return ONLY JSON with keys:\n"
            "author (string|null), keywords (array of strings), title (string|null),\n"
            "date_range (object with start/end year as strings or null), doi (string|null), raw_query (string).\n"
            "Do not include markdown or explanations.\n\n"
            f"Query: {query}"
        )
        response = provider.send_message(input=prompt)
        parsed = _parse_ai_decomposition_response(response)
        return _merge_fields(deterministic, parsed, query)
    except Exception as exc:
        logger.debug(f"AI decomposition failed; using deterministic fallback: {exc}")
        return deterministic


def _parse_ai_decomposition_response(response_text: str) -> Dict[str, Any]:
    """
    Parse provider response into a dictionary.
    """
    text = (response_text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _merge_fields(
    deterministic: QueryFields,
    parsed: Dict[str, Any],
    raw_query: str,
) -> QueryFields:
    """
    Merge AI-extracted fields with deterministic defaults.
    """
    if not isinstance(parsed, dict):
        return deterministic

    author = parsed.get("author")
    title = parsed.get("title")
    doi = parsed.get("doi")
    keywords = parsed.get("keywords")
    date_range = parsed.get("date_range")

    merged = QueryFields(
        author=str(author).strip() if author else deterministic.author,
        keywords=deterministic.keywords,
        title=str(title).strip() if title else deterministic.title,
        date_range=deterministic.date_range,
        doi=str(doi).strip() if doi else deterministic.doi,
        raw_query=(str(parsed.get("raw_query")).strip() if parsed.get("raw_query") else raw_query),
    )

    if isinstance(keywords, list):
        merged.keywords = sorted(
            {
                str(k).strip().lower()
                for k in keywords
                if str(k).strip()
            }
        )[:12] or deterministic.keywords

    if isinstance(date_range, dict):
        start = str(date_range.get("start", "")).strip()
        end = str(date_range.get("end", "")).strip()
        if start and end:
            merged.date_range = (start, end)

    return merged


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
