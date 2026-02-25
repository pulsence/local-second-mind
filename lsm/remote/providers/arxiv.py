"""
arXiv API provider for remote search.

Provides academic paper search via the arXiv API.
API documentation: https://info.arxiv.org/help/api/
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

from lsm.remote.base import RemoteResult
from lsm.remote.providers.base_oai import BaseOAIProvider
from lsm.logging import get_logger

logger = get_logger(__name__)


class ArXivProvider(BaseOAIProvider):
    """
    Remote provider using the arXiv API.

    Supports search by title, author, topic (abstract), and category.
    """

    API_ENDPOINT = "https://export.arxiv.org/api/query"
    DEFAULT_TIMEOUT = 10
    DEFAULT_MIN_INTERVAL_SECONDS = 3.0
    DEFAULT_SNIPPET_MAX_CHARS = 700
    DEFAULT_SORT_BY = "relevance"
    DEFAULT_SORT_ORDER = "descending"

    _NAMESPACES = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize arXiv provider.

        Args:
            config: Configuration dict with optional keys:
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 10)
                - min_interval_seconds: Minimum seconds between API requests
                - user_agent: Custom User-Agent header (or from env)
                - snippet_max_chars: Max characters per snippet (default: 700)
                - sort_by: arXiv sort field (relevance|lastUpdatedDate|submittedDate)
                - sort_order: arXiv sort order (ascending|descending)
                - categories: Optional list of category filters (e.g., ["cs.AI"])
        """
        super().__init__(config)

        self.endpoint = config.get("endpoint") or self.API_ENDPOINT
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        self.sort_by = config.get("sort_by") or self.DEFAULT_SORT_BY
        self.sort_order = config.get("sort_order") or self.DEFAULT_SORT_ORDER
        categories = config.get("categories") or []
        self.categories = [str(cat).strip() for cat in categories if str(cat).strip()]
        self.user_agent = (
            config.get("user_agent")
            or os.getenv("LSM_ARXIV_USER_AGENT")
            or os.getenv("ARXIV_USER_AGENT")
            or "LocalSecondMind/1.0 (local usage; contact: you@example.com)"
        )
        self._last_request_time = 0.0
        self.normalize_snippet_whitespace = False

    @property
    def name(self) -> str:
        return "arxiv"

    def is_available(self) -> bool:
        return True

    def validate_config(self) -> None:
        """Validate arXiv configuration."""
        if not self.user_agent:
            raise ValueError("arXiv provider requires a User-Agent string.")

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search arXiv for relevant papers.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        search_query = self._build_search_query(query)
        if not search_query:
            return []

        logger.debug(f"arXiv: query='{search_query}', max_results={max_results}")

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order,
        }

        try:
            xml_text = self._request(params)
            results = self._parse_feed(xml_text, max_results)
            logger.info(f"arXiv returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"arXiv API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"arXiv parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "arXiv"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Paper/topic query.", "required": True},
            {"name": "title", "type": "string", "description": "Title phrase.", "required": False},
            {"name": "author", "type": "string", "description": "Author name.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Topic keywords.", "required": False},
            {"name": "year", "type": "integer", "description": "Publication year hint.", "required": False},
            {"name": "doi", "type": "string", "description": "DOI hint.", "required": False},
            {"name": "category", "type": "string", "description": "arXiv category hint.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "arxiv_id", "type": "string", "description": "arXiv identifier."},
            {"name": "categories", "type": "array[string]", "description": "arXiv categories."},
            {"name": "primary_category", "type": "string", "description": "Primary arXiv category."},
            {"name": "published", "type": "string", "description": "Published timestamp."},
            {"name": "updated", "type": "string", "description": "Updated timestamp."},
            {"name": "pdf_url", "type": "string", "description": "Direct PDF URL when available."},
            {"name": "citation", "type": "string", "description": "Formatted citation string."},
        ]

    def get_description(self) -> str:
        return "Academic preprint provider for arXiv papers across STEM fields."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return super().search_structured(input_dict, max_results=max_results)

    def _request(self, params: Dict[str, Any]) -> str:
        self._throttle()
        headers = {"User-Agent": self.user_agent}
        response = requests.get(self.endpoint, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _build_search_query(self, query: str) -> str:
        stripped = query.strip()
        if not stripped:
            return ""

        parts = self._parse_fielded_query(stripped)
        clauses = []
        for field, value in parts:
            normalized = self._normalize_value(value)
            if not normalized:
                continue
            clauses.append(f"{field}:{normalized}")

        if not clauses:
            clauses.append(f"all:{self._normalize_value(stripped)}")

        if self.categories:
            category_clause = " OR ".join(f"cat:{cat}" for cat in self.categories)
            clauses.append(f"({category_clause})")

        return " AND ".join(clauses)

    def _parse_fielded_query(self, query: str) -> List[tuple[str, str]]:
        cleaned_query = re.sub(
            r'(?i)"(title|author|category|cat|topic|abstract|all):([^"]+)"',
            r"\1:\2",
            query,
        )
        pattern = re.compile(r"(?i)\b(title|author|category|cat|topic|abstract|all):")
        matches = list(pattern.finditer(cleaned_query))
        if not matches:
            return []

        field_map = {
            "title": "ti",
            "author": "au",
            "category": "cat",
            "cat": "cat",
            "topic": "abs",
            "abstract": "abs",
            "all": "all",
        }

        parts = []
        for idx, match in enumerate(matches):
            field_key = match.group(1).lower()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned_query)
            value = cleaned_query[start:end].strip().strip('"')
            value = value.strip(" ?!.,;:")
            if value:
                parts.append((field_map[field_key], value))

        return parts

    def _normalize_value(self, value: str) -> str:
        cleaned = " ".join(value.replace('"', "").split())
        if not cleaned:
            return ""
        if " " in cleaned and not (cleaned.startswith('"') and cleaned.endswith('"')):
            return f"\"{cleaned}\""
        return cleaned

    def _parse_feed(self, xml_text: str, max_results: int) -> List[RemoteResult]:
        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", self._NAMESPACES)
        results: List[RemoteResult] = []

        for i, entry in enumerate(entries[:max_results]):
            title = self._clean_text(self._get_text(entry, "atom:title"))
            summary = self._clean_text(self._get_text(entry, "atom:summary"))
            abs_url = self._normalize_abs_url(self._get_text(entry, "atom:id"))
            pdf_url = self._extract_pdf_url(entry, abs_url)

            authors = [
                self._clean_text(author.text or "")
                for author in entry.findall("atom:author/atom:name", self._NAMESPACES)
                if author is not None
            ]

            categories = [
                cat.get("term")
                for cat in entry.findall("atom:category", self._NAMESPACES)
                if cat is not None and cat.get("term")
            ]

            primary_category = None
            primary = entry.find("arxiv:primary_category", self._NAMESPACES)
            if primary is not None:
                primary_category = primary.get("term")

            published = self._get_text(entry, "atom:published")
            updated = self._get_text(entry, "atom:updated")
            arxiv_id = self._extract_arxiv_id(abs_url)

            score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))
            snippet = self._truncate(summary)

            metadata = {
                "arxiv_id": arxiv_id,
                "authors": authors,
                "categories": categories,
                "primary_category": primary_category,
                "published": published,
                "updated": updated,
                "pdf_url": pdf_url,
                "citation": self._format_citation(title, abs_url, authors, published, arxiv_id),
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=abs_url,
                    snippet=snippet,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def _get_text(self, entry: ET.Element, path: str) -> str:
        node = entry.find(path, self._NAMESPACES)
        return node.text if node is not None and node.text else ""

    def _clean_text(self, text: str) -> str:
        return " ".join((text or "").split())

    def _normalize_abs_url(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://arxiv.org/abs/{url}"
        if parsed.scheme == "http":
            return url.replace("http://", "https://", 1)
        return url

    def _extract_pdf_url(self, entry: ET.Element, abs_url: str) -> str:
        for link in entry.findall("atom:link", self._NAMESPACES):
            href = link.get("href")
            link_type = link.get("type", "")
            title = link.get("title", "")
            if link_type == "application/pdf" or title.lower() == "pdf":
                return self._normalize_abs_url(href).replace("/abs/", "/pdf/")

        if abs_url:
            return abs_url.replace("/abs/", "/pdf/")
        return ""

    def _extract_arxiv_id(self, abs_url: str) -> Optional[str]:
        if not abs_url:
            return None
        return abs_url.rstrip("/").split("/")[-1]

    def _format_citation(
        self,
        title: str,
        abs_url: str,
        authors: List[str],
        published: str,
        arxiv_id: Optional[str],
    ) -> str:
        year = published[:4] if published else "n.d."
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        author_str = self._format_authors(authors)
        id_part = f"arXiv:{arxiv_id}" if arxiv_id else "arXiv"
        return f"{author_str} ({year}). {title}. {id_part}. {abs_url} (accessed {date_str})."

    def _format_authors(self, authors: List[str]) -> str:
        return super()._format_authors(authors, fallback="arXiv")
