"""
Unpaywall API provider for open access link resolution.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.utils import normalize_doi

logger = get_logger(__name__)


class UnpaywallProvider(BaseRemoteProvider):
    """
    Resolve open access locations for a DOI using Unpaywall.
    """

    API_ENDPOINT = "https://api.unpaywall.org/v2"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 600

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Unpaywall provider.

        Args:
            config: Configuration dict with optional keys:
                - email: Contact email (required by Unpaywall policy)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between API requests
                - snippet_max_chars: Max characters per snippet
        """
        super().__init__(config)

        self.email = config.get("email") or os.getenv("UNPAYWALL_EMAIL")
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
        return "unpaywall"

    def is_available(self) -> bool:
        return bool(self.email)

    def validate_config(self) -> None:
        if not self.email:
            raise ValueError(
                "Unpaywall requires a contact email. Set 'email' in config or "
                "UNPAYWALL_EMAIL environment variable."
            )

    def get_name(self) -> str:
        return "Unpaywall"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "DOI or DOI-based query.", "required": True},
            {"name": "doi", "type": "string", "description": "DOI lookup value.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "oa_status", "type": "string", "description": "OA status label."},
            {"name": "is_oa", "type": "boolean", "description": "Open access flag."},
            {"name": "oa_url", "type": "string", "description": "Best OA URL."},
            {"name": "pdf_url", "type": "string", "description": "Best OA PDF URL."},
            {"name": "license", "type": "string", "description": "License string."},
            {"name": "journal", "type": "string", "description": "Journal name."},
            {"name": "publisher", "type": "string", "description": "Publisher name."},
            {"name": "published_date", "type": "string", "description": "Publication date."},
            {"name": "oa_locations", "type": "array[object]", "description": "OA locations list."},
        ]

    def get_description(self) -> str:
        return "Open access resolver that maps DOI values to free-to-read locations."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if isinstance(input_dict, dict):
            doi_value = normalize_doi(input_dict.get("doi"))
            if doi_value:
                results = self.search(doi_value, max_results=max_results)
                return [self._to_structured_output(result) for result in results]
        return super().search_structured(input_dict, max_results=max_results)

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        doi = self._extract_doi(query)
        if not doi:
            logger.warning("Unpaywall requires a DOI query.")
            return []

        try:
            payload = self._get_by_doi(doi)
        except requests.exceptions.RequestException as exc:
            logger.error(f"Unpaywall API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Unpaywall parsing error: {exc}")
            return []

        if not payload:
            return []

        result = self._convert_payload(payload, doi)
        return [result][:max_results]

    def _extract_doi(self, query: str) -> Optional[str]:
        query = str(query or "").strip()
        if not query:
            return None
        if query.lower().startswith("doi:"):
            query = query[4:].strip()
        doi = normalize_doi(query)
        if doi:
            return doi
        match = re.search(r"10\.\d{4,9}/\S+", query)
        if match:
            return normalize_doi(match.group(0))
        return None

    def _get_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        self._throttle()
        url = f"{self.endpoint}/{doi}"
        params = {"email": self.email}
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None

    def _convert_payload(self, payload: Dict[str, Any], doi: str) -> RemoteResult:
        title = payload.get("title") or f"DOI {doi}"
        best_oa = payload.get("best_oa_location") or {}
        oa_url = best_oa.get("url") or payload.get("doi_url") or f"https://doi.org/{doi}"
        pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
        snippet = payload.get("oa_status") or "open access"
        snippet = self._truncate(str(snippet))

        published_date = payload.get("published_date") or payload.get("year")
        year = None
        if isinstance(published_date, int):
            year = published_date
        elif isinstance(published_date, str):
            try:
                year = int(published_date[:4])
            except ValueError:
                year = None

        metadata = {
            "doi": doi,
            "is_oa": payload.get("is_oa", False),
            "oa_status": payload.get("oa_status"),
            "oa_url": oa_url,
            "pdf_url": pdf_url,
            "license": best_oa.get("license"),
            "journal": payload.get("journal_name"),
            "publisher": payload.get("publisher"),
            "published_date": payload.get("published_date"),
            "year": year,
            "oa_locations": payload.get("oa_locations") or [],
            "source_id": doi,
        }

        return RemoteResult(
            title=title,
            url=oa_url,
            snippet=snippet,
            score=1.0,
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
