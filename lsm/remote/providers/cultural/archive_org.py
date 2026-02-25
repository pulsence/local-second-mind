"""
Archive.org provider using the Advanced Search API.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class ArchiveOrgProvider(BaseRemoteProvider):
    """
    Archive.org metadata search provider.
    """

    API_ENDPOINT = "https://archive.org/advancedsearch.php"
    DEFAULT_TIMEOUT = 20
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - endpoint: Custom endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
                - collection: Optional collection filter
                - media_type: Optional mediatype filter
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
        self.media_type = config.get("media_type")
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "archive_org"

    def get_name(self) -> str:
        return "Archive.org"

    def get_description(self) -> str:
        return "Archive.org metadata search for books, media, and documents."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Archive.org query.", "required": True},
            {"name": "collection", "type": "string", "description": "Collection filter.", "required": False},
            {"name": "media_type", "type": "string", "description": "Media type filter.", "required": False},
            {"name": "year", "type": "integer", "description": "Publication year hint.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "identifier", "type": "string", "description": "Archive.org identifier."},
            {"name": "collection", "type": "array[string]", "description": "Collection list."},
            {"name": "mediatype", "type": "string", "description": "Media type."},
            {"name": "downloads", "type": "integer", "description": "Download count."},
            {"name": "subjects", "type": "array[string]", "description": "Subject tags."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            params = {
                "q": self._build_query(query),
                "rows": max_results,
                "page": 1,
                "output": "json",
                "fl[]": [
                    "identifier",
                    "title",
                    "description",
                    "creator",
                    "year",
                    "date",
                    "mediatype",
                    "collection",
                    "subject",
                    "downloads",
                ],
            }
            data = self._request(params)
            docs = data.get("response", {}).get("docs", [])
            results = self._convert_docs(docs, max_results)
            logger.info(f"Archive.org returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Archive.org API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Archive.org parsing error: {exc}")
            return []

    def _build_query(self, query: str) -> str:
        query_parts = [query]
        collection = self.collection or None
        if collection:
            query_parts.append(f"collection:{collection}")
        media_type = self.media_type or None
        if media_type:
            query_parts.append(f"mediatype:{media_type}")
        return " AND ".join(query_parts)

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_docs(self, docs: List[Dict[str, Any]], max_results: int) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, doc in enumerate(docs[:max_results]):
            identifier = doc.get("identifier")
            title = doc.get("title") or identifier or "Untitled"
            url = f"https://archive.org/details/{identifier}" if identifier else ""
            description = self._coerce_text(doc.get("description"))
            creators = self._coerce_list(doc.get("creator"))
            year = self._coerce_year(doc.get("year") or doc.get("date"))
            subjects = self._coerce_list(doc.get("subject"))

            snippet = description or f"Archive.org item by {', '.join(creators[:3])}" if creators else title
            snippet = self._truncate(snippet)

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "identifier": identifier,
                "authors": creators,
                "year": year,
                "collection": self._coerce_list(doc.get("collection")),
                "mediatype": doc.get("mediatype"),
                "downloads": doc.get("downloads"),
                "subjects": subjects,
                "source_id": identifier or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or identifier or title,
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
        if isinstance(value, int):
            return value
        text = str(value)
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
        return None
