"""
Rijksmuseum data services provider.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class RijksmuseumProvider(BaseRemoteProvider):
    """
    Rijksmuseum collection search provider.
    """

    API_ENDPOINT = "https://www.rijksmuseum.nl/api/en/collection"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - api_key: Rijksmuseum API key (required)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
                - include_details: Fetch object detail records
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("RIJKSMUSEUM_API_KEY")
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
        self.include_details = bool(config.get("include_details", True))
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "rijksmuseum"

    def get_name(self) -> str:
        return "Rijksmuseum"

    def get_description(self) -> str:
        return "Rijksmuseum collection metadata and images."

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Rijksmuseum requires an API key. Set 'api_key' in config or "
                "RIJKSMUSEUM_API_KEY env var."
            )

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Rijksmuseum search query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "object_number", "type": "string", "description": "Rijksmuseum object number."},
            {"name": "principal_maker", "type": "string", "description": "Principal maker."},
            {"name": "image_url", "type": "string", "description": "Image URL."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "key": self.api_key,
                "q": query,
                "ps": max_results,
            }
            data = self._request(self.endpoint, params)
            objects = data.get("artObjects", []) if isinstance(data, dict) else []
            results: List[RemoteResult] = []
            for idx, obj in enumerate(objects[:max_results]):
                detail = obj
                if self.include_details and obj.get("objectNumber"):
                    detail = self._request(
                        f"{self.endpoint}/{obj['objectNumber']}", {"key": self.api_key}
                    ).get("artObject", obj)
                result = self._convert_object(detail, idx, max_results)
                if result:
                    results.append(result)
            logger.info(f"Rijksmuseum returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Rijksmuseum API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Rijksmuseum parsing error: {exc}")
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
        object_number = obj.get("objectNumber")
        title = obj.get("title") or obj.get("longTitle") or "Untitled"
        maker = obj.get("principalOrFirstMaker")
        image_url = None
        web_image = obj.get("webImage")
        if isinstance(web_image, dict):
            image_url = web_image.get("url")
        url = obj.get("links", {}).get("web") if isinstance(obj.get("links"), dict) else None
        url = url or obj.get("website") or image_url or ""

        snippet = self._truncate(obj.get("longTitle") or title)
        score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

        metadata = {
            "object_number": object_number,
            "authors": [maker] if maker else [],
            "year": obj.get("dating", {}).get("year")
            if isinstance(obj.get("dating"), dict)
            else None,
            "principal_maker": maker,
            "image_url": image_url,
            "source_id": object_number or url,
        }

        return RemoteResult(
            title=title,
            url=url or object_number or title,
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
