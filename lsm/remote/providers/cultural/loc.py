"""
Library of Congress (LOC) JSON API provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class LOCProvider(BaseRemoteProvider):
    """
    Library of Congress search provider via loc.gov JSON API.
    """

    API_ENDPOINT = "https://www.loc.gov"
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
                - collection: Optional collection path (e.g., 'maps', 'photos')
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
        self.collection = config.get("collection")
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "loc"

    def get_name(self) -> str:
        return "Library of Congress"

    def get_description(self) -> str:
        return "Library of Congress catalog search."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "LOC search query.", "required": True},
            {"name": "collection", "type": "string", "description": "Collection filter.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "loc_id", "type": "string", "description": "LOC item ID."},
            {"name": "date", "type": "string", "description": "Date string."},
            {"name": "subjects", "type": "array[string]", "description": "Subject terms."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "q": query,
                "fo": "json",
                "c": max_results,
            }
            data = self._request(params)
            items = data.get("results", [])
            results = self._convert_items(items, max_results)
            logger.info(f"LOC returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"LOC API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"LOC parsing error: {exc}")
            return []

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        endpoint = self.endpoint.rstrip("/")
        if self.collection:
            endpoint = f"{endpoint}/{self.collection}"
        response = requests.get(endpoint + "/", params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_items(self, items: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, item in enumerate(items[:max_results]):
            title = item.get("title") or "Untitled"
            url = item.get("url") or item.get("item") or ""
            description = self._coerce_text(item.get("description"))
            subjects = self._coerce_list(item.get("subject"))
            date = item.get("date")

            snippet = description or title
            snippet = self._truncate(snippet)
            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "loc_id": item.get("id"),
                "date": date,
                "subjects": subjects,
                "source_id": item.get("id") or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or item.get("id") or title,
                    snippet=snippet,
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

    @staticmethod
    def _coerce_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(item) for item in value if str(item).strip())
        return str(value)
