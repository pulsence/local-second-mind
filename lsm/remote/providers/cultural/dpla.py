"""
Digital Public Library of America (DPLA) provider.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class DPLAProvider(BaseRemoteProvider):
    """
    DPLA search provider.
    """

    API_ENDPOINT = "https://api.dp.la/v2/items"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - api_key: DPLA API key (required)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("DPLA_API_KEY")
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
        return "dpla"

    def get_name(self) -> str:
        return "DPLA"

    def get_description(self) -> str:
        return "Digital Public Library of America cultural heritage metadata."

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(
                "DPLA requires an API key. Set 'api_key' in config or DPLA_API_KEY env var."
            )

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "DPLA query.", "required": True},
            {"name": "year", "type": "integer", "description": "Year hint.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "dpla_id", "type": "string", "description": "DPLA item ID."},
            {"name": "provider", "type": "string", "description": "Contributing provider."},
            {"name": "subjects", "type": "array[string]", "description": "Subject terms."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "q": query,
                "page_size": max_results,
                "api_key": self.api_key,
            }
            data = self._request(params)
            docs = data.get("docs", []) if isinstance(data, dict) else []
            results = self._convert_docs(docs, max_results)
            logger.info(f"DPLA returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"DPLA API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"DPLA parsing error: {exc}")
            return []

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_docs(self, docs: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, doc in enumerate(docs[:max_results]):
            dpla_id = doc.get("id")
            source_resource = doc.get("sourceResource") or {}
            title = self._coerce_text(source_resource.get("title")) or "Untitled"
            description = self._coerce_text(source_resource.get("description"))
            creators = self._coerce_list(source_resource.get("creator"))
            subjects = [
                subject.get("name")
                for subject in source_resource.get("subject", []) or []
                if isinstance(subject, dict) and subject.get("name")
            ]
            year = self._coerce_year(source_resource.get("date"))

            url = doc.get("isShownAt") or doc.get("isShownBy") or dpla_id or ""
            snippet = description or title
            snippet = self._truncate(snippet)

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "dpla_id": dpla_id,
                "authors": creators,
                "year": year,
                "provider": doc.get("provider", {}).get("name")
                if isinstance(doc.get("provider"), dict)
                else None,
                "subjects": [s for s in subjects if s],
                "source_id": dpla_id or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or dpla_id or title,
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

    @staticmethod
    def _coerce_year(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, dict) and "begin" in value:
            value = value.get("begin")
        if isinstance(value, int):
            return value
        text = str(value)
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
        return None
