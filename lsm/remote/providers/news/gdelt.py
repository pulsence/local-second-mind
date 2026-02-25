"""
GDELT news API provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class GDELTProvider(BaseRemoteProvider):
    """
    GDELT global news coverage provider.
    """

    API_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
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
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "gdelt"

    def get_name(self) -> str:
        return "GDELT"

    def get_description(self) -> str:
        return "GDELT global news aggregation API."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "GDELT query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "source", "type": "string", "description": "Source domain."},
            {"name": "published_date", "type": "string", "description": "Seen date."},
            {"name": "language", "type": "string", "description": "Language code."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": max_results,
            }
            data = self._request(params)
            articles = data.get("articles", [])
            results = self._convert_articles(articles, max_results)
            logger.info(f"GDELT returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"GDELT API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"GDELT parsing error: {exc}")
            return []

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_articles(self, articles: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, article in enumerate(articles[:max_results]):
            title = article.get("title") or "Untitled"
            url = article.get("url") or ""
            snippet = article.get("seendate") or title
            source = article.get("domain")
            language = article.get("language")
            published = article.get("seendate")

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))
            metadata = {
                "source": source,
                "language": language,
                "published_date": published,
                "source_id": url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or title,
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
