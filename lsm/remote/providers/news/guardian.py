"""
The Guardian Content API provider.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class GuardianProvider(BaseRemoteProvider):
    """
    Guardian Content API provider.
    """

    API_ENDPOINT = "https://content.guardianapis.com/search"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - api_key: Guardian API key (required)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("GUARDIAN_API_KEY")
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
        return "guardian"

    def get_name(self) -> str:
        return "The Guardian"

    def get_description(self) -> str:
        return "The Guardian news content search."

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Guardian requires an API key. Set 'api_key' in config or GUARDIAN_API_KEY env var."
            )

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Guardian search query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "guardian_id", "type": "string", "description": "Guardian content ID."},
            {"name": "section", "type": "string", "description": "Section name."},
            {"name": "published_date", "type": "string", "description": "Publication timestamp."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "q": query,
                "page-size": max_results,
                "api-key": self.api_key,
                "show-fields": "trailText",
            }
            data = self._request(params)
            items = data.get("response", {}).get("results", [])
            results = self._convert_items(items, max_results)
            logger.info(f"Guardian returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Guardian API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Guardian parsing error: {exc}")
            return []

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_items(self, items: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, item in enumerate(items[:max_results]):
            title = item.get("webTitle") or "Untitled"
            url = item.get("webUrl") or ""
            snippet = item.get("fields", {}).get("trailText") or title
            published = item.get("webPublicationDate")
            section = item.get("sectionName")
            guardian_id = item.get("id")

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))
            metadata = {
                "guardian_id": guardian_id,
                "section": section,
                "published_date": published,
                "source_id": guardian_id or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or guardian_id or title,
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
