"""
Perseus CTS API provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class PerseusCTSProvider(BaseRemoteProvider):
    """
    Perseus CTS provider for classical text retrieval by CTS URN.
    """

    API_ENDPOINT = "https://perseus.tufts.edu/hopper/CTS"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 800

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - endpoint: CTS endpoint URL
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
        return "perseus_cts"

    def get_name(self) -> str:
        return "Perseus CTS"

    def get_description(self) -> str:
        return "Perseus CTS passage retrieval for classical texts."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "urn", "type": "string", "description": "CTS URN.", "required": False},
            {"name": "passage", "type": "string", "description": "Passage reference.", "required": False},
            {"name": "query", "type": "string", "description": "CTS URN query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "urn", "type": "string", "description": "CTS URN."},
            {"name": "passage", "type": "string", "description": "Passage reference."},
            {"name": "citation", "type": "string", "description": "Citation label."},
        ]

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        urn = None
        passage = None
        if isinstance(input_dict, dict):
            urn = input_dict.get("urn")
            passage = input_dict.get("passage")
        query = urn or input_dict.get("query") if isinstance(input_dict, dict) else None
        if not query:
            return []
        results = self.search(self._compose_query(query, passage), max_results=max_results)
        return [self._to_structured_output(result) for result in results]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        urn, passage = self._parse_query(query)
        if not urn:
            logger.warning("Perseus CTS requires a CTS URN.")
            return []

        try:
            text = self._get_passage(urn, passage)
            if not text:
                return []
            title = urn if not passage else f"{urn} ({passage})"
            url = self._build_url(urn, passage)
            snippet = self._truncate(text)
            metadata = {
                "urn": urn,
                "passage": passage,
                "citation": title,
                "source_id": f"{urn}:{passage}" if passage else urn,
            }
            return [
                RemoteResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=1.0,
                    metadata=metadata,
                )
            ][:max_results]
        except requests.exceptions.RequestException as exc:
            logger.error(f"Perseus CTS API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Perseus CTS parsing error: {exc}")
            return []

    def _compose_query(self, urn: str, passage: Optional[str]) -> str:
        if passage:
            return f"{urn}:{passage}"
        return urn

    def _parse_query(self, query: str) -> tuple[Optional[str], Optional[str]]:
        value = str(query).strip()
        if value.startswith("urn:cts"):
            parts = value.split(":", 4)
            if len(parts) == 5 and parts[-1]:
                urn = ":".join(parts[:4])
                passage = parts[4]
                return urn, passage
            return value, None
        return None, None

    def _get_passage(self, urn: str, passage: Optional[str]) -> str:
        self._throttle()
        params = {"request": "GetPassage", "urn": urn}
        if passage:
            params["psg"] = passage
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return self._extract_text(response.text)

    def _extract_text(self, xml_text: str) -> str:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ""
        return " ".join(root.itertext()).strip()

    def _build_url(self, urn: str, passage: Optional[str]) -> str:
        params = {"request": "GetPassage", "urn": urn}
        if passage:
            params["psg"] = passage
        return f"{self.endpoint}?{urlencode(params)}"

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
