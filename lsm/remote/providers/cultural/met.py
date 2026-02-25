"""
Metropolitan Museum of Art Collection API provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class MetProvider(BaseRemoteProvider):
    """
    The Met Collection API provider.
    """

    API_ENDPOINT = "https://collectionapi.metmuseum.org/public/collection/v1"
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
                - has_images: Only return objects with images
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
        self.has_images = bool(config.get("has_images", False))
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "met"

    def get_name(self) -> str:
        return "The Met"

    def get_description(self) -> str:
        return "Metropolitan Museum of Art collection search."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Met search query.", "required": True},
            {"name": "has_images", "type": "boolean", "description": "Only return items with images.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "met_id", "type": "integer", "description": "Met object ID."},
            {"name": "artist", "type": "string", "description": "Artist name."},
            {"name": "department", "type": "string", "description": "Department."},
            {"name": "culture", "type": "string", "description": "Culture."},
            {"name": "primary_image", "type": "string", "description": "Primary image URL."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            search_params = {"q": query}
            if self.has_images:
                search_params["hasImages"] = "true"
            search_data = self._request(f"{self.endpoint}/search", search_params)
            object_ids = search_data.get("objectIDs") or []

            results: List[RemoteResult] = []
            for idx, object_id in enumerate(object_ids[:max_results]):
                obj = self._request(f"{self.endpoint}/objects/{object_id}", {})
                result = self._convert_object(obj, idx, max_results)
                if result:
                    results.append(result)

            logger.info(f"Met returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Met API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Met parsing error: {exc}")
            return []

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_object(
        self, obj: Dict[str, Any], idx: int, max_results: int
    ) -> Optional[RemoteResult]:
        if not obj:
            return None
        met_id = obj.get("objectID")
        title = obj.get("title") or obj.get("objectName") or "Untitled"
        url = obj.get("objectURL") or ""
        artist = obj.get("artistDisplayName") or obj.get("artistAlphaSort")
        department = obj.get("department")
        culture = obj.get("culture")
        year = obj.get("objectEndDate") or obj.get("objectBeginDate")
        primary_image = obj.get("primaryImageSmall") or obj.get("primaryImage")

        snippet = self._truncate(obj.get("creditLine") or title)
        score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

        metadata = {
            "met_id": met_id,
            "authors": [artist] if artist else [],
            "year": int(year) if isinstance(year, int) else None,
            "artist": artist,
            "department": department,
            "culture": culture,
            "primary_image": primary_image,
            "source_id": met_id or url,
        }

        return RemoteResult(
            title=title,
            url=url or str(met_id),
            snippet=snippet,
            score=score,
            metadata=metadata,
        )

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
