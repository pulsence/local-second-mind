"""
Wikipedia API provider for remote search.

Provides general knowledge lookups using the MediaWiki API.
API documentation: https://www.mediawiki.org/wiki/API:Search
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from urllib.parse import quote

import requests

from .base import BaseRemoteProvider, RemoteResult
from lsm.cli.logging import get_logger

logger = get_logger(__name__)


class WikipediaProvider(BaseRemoteProvider):
    """
    Remote provider using the Wikipedia MediaWiki API.

    Supports search, disambiguation handling, section extraction, and citations.
    """

    API_ENDPOINT_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"
    DEFAULT_LANGUAGE = "en"
    DEFAULT_TIMEOUT = 10
    DEFAULT_MIN_INTERVAL_SECONDS = 1.0
    DEFAULT_SECTION_LIMIT = 2
    DEFAULT_SNIPPET_MAX_CHARS = 600

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Wikipedia provider.

        Args:
            config: Configuration dict with optional keys:
                - language: Wikipedia language code (default: "en")
                - endpoint: Custom API endpoint (default: Wikipedia API)
                - timeout: Request timeout in seconds (default: 10)
                - min_interval_seconds: Minimum seconds between API requests (default: 1.0)
                - section_limit: Max sections to include in snippet (default: 2)
                - snippet_max_chars: Max characters per snippet (default: 600)
                - include_disambiguation: Include disambiguation pages (default: False)
                - user_agent: Custom User-Agent header (or from env)
        """
        super().__init__(config)

        self.language = config.get("language") or os.getenv("WIKIPEDIA_LANGUAGE") or self.DEFAULT_LANGUAGE
        self.endpoint = config.get("endpoint") or self.API_ENDPOINT_TEMPLATE.format(language=self.language)
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        section_limit = config.get("section_limit")
        self.section_limit = int(section_limit if section_limit is not None else self.DEFAULT_SECTION_LIMIT)
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        self.include_disambiguation = bool(config.get("include_disambiguation", False))
        self.user_agent = (
            config.get("user_agent")
            or os.getenv("LSM_WIKIPEDIA_USER_AGENT")
            or os.getenv("WIKIPEDIA_USER_AGENT")
            or "LocalSecondMind/1.0 (local usage)"
        )
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "wikipedia"

    def is_available(self) -> bool:
        return True

    def validate_config(self) -> None:
        """Validate Wikipedia configuration."""
        if not self.user_agent:
            raise ValueError("Wikipedia provider requires a User-Agent string.")

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search Wikipedia for relevant articles.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"Wikipedia: query='{query}', max_results={max_results}")

        try:
            search_results = self._search_titles(query, max_results)
            if not search_results:
                return []

            titles = [item["title"] for item in search_results]
            details_by_title = self._fetch_page_details(titles)

            results: List[RemoteResult] = []
            disambiguation_buffer: List[RemoteResult] = []

            for i, item in enumerate(search_results):
                if len(results) >= max_results:
                    break

                title = item["title"]
                page = details_by_title.get(title)
                if not page or page.get("missing"):
                    continue

                is_disambiguation = self._is_disambiguation_page(page)
                snippet = self._build_snippet(query, page.get("extract", ""))
                url = page.get("fullurl") or self._build_page_url(title)

                score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))
                metadata = {
                    "pageid": page.get("pageid"),
                    "language": self.language,
                    "last_modified": page.get("touched"),
                    "is_disambiguation": is_disambiguation,
                    "citation": self._format_citation(title, url),
                }

                result = RemoteResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                    metadata=metadata,
                )

                if is_disambiguation and not self.include_disambiguation:
                    disambiguation_buffer.append(result)
                    continue

                results.append(result)

            if not results and disambiguation_buffer:
                results = disambiguation_buffer[:max_results]

            logger.info(f"Wikipedia returned {len(results)} results")
            return results

        except requests.exceptions.RequestException as exc:
            logger.error(f"Wikipedia API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Wikipedia parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "Wikipedia"

    def _search_titles(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        search_limit = min(max_results * 3, 20)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": search_limit,
            "format": "json",
            "utf8": 1,
        }
        data = self._request(params)
        return data.get("query", {}).get("search", [])

    def _fetch_page_details(self, titles: List[str]) -> Dict[str, Dict[str, Any]]:
        if not titles:
            return {}

        params = {
            "action": "query",
            "prop": "extracts|pageprops|info|categories",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "cllimit": 10,
            "titles": "|".join(titles),
            "format": "json",
            "utf8": 1,
        }
        data = self._request(params)
        pages = data.get("query", {}).get("pages", {})
        results: Dict[str, Dict[str, Any]] = {}
        for page in pages.values():
            title = page.get("title")
            if title:
                results[title] = page
        return results

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        headers = {"User-Agent": self.user_agent}
        response = requests.get(self.endpoint, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _is_disambiguation_page(self, page: Dict[str, Any]) -> bool:
        pageprops = page.get("pageprops") or {}
        if "disambiguation" in pageprops:
            return True

        for category in page.get("categories", []) or []:
            title = category.get("title", "").lower()
            if "disambiguation pages" in title:
                return True

        return False

    def _build_snippet(self, query: str, extract: str) -> str:
        if not extract:
            return ""

        sections = self._split_sections(extract)
        if not sections:
            return self._truncate(extract)

        query_terms = self._extract_query_terms(query)
        matched = [
            section for section in sections
            if query_terms and any(term in section["text"].lower() for term in query_terms)
        ]

        if matched:
            selected = matched[: self.section_limit]
        else:
            selected = sections[: self.section_limit]

        snippet_parts = []
        for section in selected:
            title = section.get("title")
            text = section.get("text", "")
            if title and title.lower() != "summary":
                snippet_parts.append(f"{title}: {text}")
            else:
                snippet_parts.append(text)

        return self._truncate("\n\n".join(snippet_parts))

    def _split_sections(self, extract: str) -> List[Dict[str, str]]:
        pattern = re.compile(r"\n==\s*(.+?)\s*==\n")
        parts = pattern.split(extract)
        if not parts:
            return []

        sections = []
        lead = parts[0].strip()
        if lead:
            sections.append({"title": "Summary", "text": lead})

        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if text:
                sections.append({"title": title, "text": text})

        return sections

    def _extract_query_terms(self, query: str) -> List[str]:
        terms = [term for term in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(term) > 2]
        return list(dict.fromkeys(terms))

    def _truncate(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= self.snippet_max_chars:
            return cleaned
        return cleaned[: self.snippet_max_chars].rstrip() + "..."

    def _build_page_url(self, title: str) -> str:
        encoded = quote(title.replace(" ", "_"))
        return f"https://{self.language}.wikipedia.org/wiki/{encoded}"

    def _format_citation(self, title: str, url: str) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            f"Wikipedia contributors, \"{title}\", Wikipedia, The Free Encyclopedia, "
            f"{url} (accessed {date_str})."
        )
