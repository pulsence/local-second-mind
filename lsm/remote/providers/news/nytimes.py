"""
New York Times API provider.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class NYTimesProvider(BaseRemoteProvider):
    """
    NYTimes Article Search and Top Stories provider.
    """

    API_BASE = "https://api.nytimes.com/svc"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - api_key: NYTimes API key (required)
                - endpoint: Custom API base
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
                - top_stories_section: Section name for top stories (optional)
                - sort: 'newest' or 'oldest' (article search)
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("NYTIMES_API_KEY")
        self.endpoint = config.get("endpoint") or self.API_BASE
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        self.top_stories_section = config.get("top_stories_section")
        self.sort = config.get("sort") or "newest"
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "nytimes"

    def get_name(self) -> str:
        return "New York Times"

    def get_description(self) -> str:
        return "New York Times Article Search and Top Stories."

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(
                "NYTimes requires an API key. Set 'api_key' in config or NYTIMES_API_KEY env var."
            )

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "NYTimes search query.", "required": True},
            {"name": "section", "type": "string", "description": "Top stories section.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "nyt_id", "type": "string", "description": "NYTimes article ID."},
            {"name": "section", "type": "string", "description": "Section name."},
            {"name": "published_date", "type": "string", "description": "Publication timestamp."},
            {"name": "byline", "type": "string", "description": "Byline text."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip() and not self.top_stories_section:
            return []

        try:
            if self.top_stories_section:
                results = self._fetch_top_stories(self.top_stories_section, max_results)
            else:
                results = self._search_articles(query, max_results)
            logger.info(f"NYTimes returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"NYTimes API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"NYTimes parsing error: {exc}")
            return []

    def _search_articles(self, query: str, max_results: int) -> List[RemoteResult]:
        url = f"{self.endpoint}/search/v2/articlesearch.json"
        params = {
            "q": query,
            "sort": self.sort,
            "api-key": self.api_key,
            "page": 0,
        }
        data = self._request(url, params)
        docs = data.get("response", {}).get("docs", [])
        return self._convert_article_docs(docs, max_results)

    def _fetch_top_stories(self, section: str, max_results: int) -> List[RemoteResult]:
        url = f"{self.endpoint}/topstories/v2/{section}.json"
        params = {"api-key": self.api_key}
        data = self._request(url, params)
        items = data.get("results", [])
        return self._convert_top_stories(items, max_results)

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_article_docs(self, docs: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, doc in enumerate(docs[:max_results]):
            title = doc.get("headline", {}).get("main") or doc.get("abstract") or "Untitled"
            url = doc.get("web_url") or ""
            snippet = doc.get("snippet") or doc.get("lead_paragraph") or title
            byline = doc.get("byline", {}).get("original")
            published = doc.get("pub_date")
            section = doc.get("section_name")
            nyt_id = doc.get("_id")

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))
            metadata = {
                "nyt_id": nyt_id,
                "section": section,
                "published_date": published,
                "byline": byline,
                "source_id": nyt_id or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or nyt_id or title,
                    snippet=self._truncate(snippet),
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def _convert_top_stories(self, items: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, item in enumerate(items[:max_results]):
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            snippet = item.get("abstract") or title
            byline = item.get("byline")
            published = item.get("published_date")
            section = item.get("section")
            nyt_id = item.get("short_url") or item.get("uri")

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))
            metadata = {
                "nyt_id": nyt_id,
                "section": section,
                "published_date": published,
                "byline": byline,
                "source_id": nyt_id or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or nyt_id or title,
                    snippet=self._truncate(snippet),
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _truncate(self, text: str) -> str:
        if len(text) <= self.snippet_max_chars:
            return text
        return text[: self.snippet_max_chars].rstrip() + "..."
