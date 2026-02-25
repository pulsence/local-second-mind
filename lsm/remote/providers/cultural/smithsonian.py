"""
Smithsonian Open Access API provider.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class SmithsonianProvider(BaseRemoteProvider):
    """
    Smithsonian Open Access search provider.
    """

    API_ENDPOINT = "https://api.si.edu/openaccess/api/v1.0/search"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - api_key: Smithsonian API key (required)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("SMITHSONIAN_API_KEY")
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
        return "smithsonian"

    def get_name(self) -> str:
        return "Smithsonian"

    def get_description(self) -> str:
        return "Smithsonian Open Access cultural heritage search."

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Smithsonian requires an API key. Set 'api_key' in config or "
                "SMITHSONIAN_API_KEY env var."
            )

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Smithsonian search query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "smithsonian_id", "type": "string", "description": "Smithsonian record ID."},
            {"name": "guid", "type": "string", "description": "Record GUID."},
            {"name": "date", "type": "string", "description": "Date string."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "q": query,
                "rows": max_results,
                "api_key": self.api_key,
            }
            data = self._request(params)
            response = data.get("response", {})
            rows = response.get("rows", []) if isinstance(response, dict) else []
            results = self._convert_rows(rows, max_results)
            logger.info(f"Smithsonian returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Smithsonian API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Smithsonian parsing error: {exc}")
            return []

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_rows(self, rows: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, row in enumerate(rows[:max_results]):
            content = row.get("content") or {}
            descriptive = content.get("descriptiveNonRepeating") or {}
            title = content.get("title") or descriptive.get("title") or "Untitled"
            record_id = descriptive.get("record_ID") or content.get("id") or row.get("id")
            guid = descriptive.get("guid")
            url = guid or descriptive.get("online_media", {}).get("media", [{}])[0].get("guid") or ""
            date = descriptive.get("date") or content.get("date")

            snippet = self._truncate(title)
            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "smithsonian_id": record_id,
                "guid": guid,
                "date": date,
                "source_id": record_id or guid or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or record_id or title,
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
