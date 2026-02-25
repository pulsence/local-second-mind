"""
IIIF Content Search provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class IIIFProvider(BaseRemoteProvider):
    """
    IIIF Content Search provider (IIIF Search API).
    """

    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - endpoint: IIIF content search endpoint (required)
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
        """
        super().__init__(config)

        self.endpoint = config.get("endpoint")
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
        return "iiif"

    def get_name(self) -> str:
        return "IIIF Search"

    def get_description(self) -> str:
        return "IIIF Content Search for digital manuscripts and images."

    def validate_config(self) -> None:
        if not self.endpoint:
            raise ValueError("IIIF provider requires an endpoint URL.")

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "IIIF search query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "iiif_id", "type": "string", "description": "IIIF resource ID."},
            {"name": "manifest", "type": "string", "description": "Associated IIIF manifest URL."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {"q": query}
            data = self._request(params)
            resources = data.get("resources") or data.get("items") or []
            results = self._convert_resources(resources, max_results)
            logger.info(f"IIIF returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"IIIF API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"IIIF parsing error: {exc}")
            return []

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_resources(self, resources: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, resource in enumerate(resources[:max_results]):
            iiif_id = resource.get("@id") or resource.get("id")
            label = resource.get("label") or resource.get("title") or iiif_id or "Untitled"
            if isinstance(label, dict):
                label = label.get("en", [""])[0] if label.get("en") else ""
            snippet = resource.get("chars") or resource.get("summary") or label
            url = iiif_id or ""
            manifest = resource.get("within") or resource.get("partOf")

            snippet = self._truncate(str(snippet))
            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "iiif_id": iiif_id,
                "manifest": manifest,
                "source_id": iiif_id or url,
            }

            results.append(
                RemoteResult(
                    title=str(label) or "Untitled",
                    url=url or str(label),
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
