"""
Wikidata SPARQL provider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


class WikidataProvider(BaseRemoteProvider):
    """
    Wikidata search via SPARQL endpoint.
    """

    API_ENDPOINT = "https://query.wikidata.org/sparql"
    DEFAULT_TIMEOUT = 20
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - endpoint: Custom SPARQL endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max snippet length
                - language: Label language (default: en)
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
        self.language = config.get("language") or "en"
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "wikidata"

    def get_name(self) -> str:
        return "Wikidata"

    def get_description(self) -> str:
        return "Wikidata entity search via SPARQL."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Wikidata query.", "required": True},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "wikidata_id", "type": "string", "description": "Wikidata Q identifier."},
            {"name": "description", "type": "string", "description": "Entity description."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            sparql = self._build_sparql(query, max_results)
            params = {"query": sparql, "format": "json"}
            data = self._request(params)
            bindings = data.get("results", {}).get("bindings", [])
            results = self._convert_bindings(bindings, max_results)
            logger.info(f"Wikidata returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Wikidata API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Wikidata parsing error: {exc}")
            return []

    def _build_sparql(self, query: str, limit: int) -> str:
        escaped = query.replace('"', '\\"')
        lang = self.language
        return f"""
SELECT ?item ?itemLabel ?description WHERE {{
  ?item rdfs:label ?itemLabel .
  FILTER(CONTAINS(LCASE(STR(?itemLabel)), LCASE("{escaped}"))).
  FILTER(LANG(?itemLabel) = "{lang}").
  OPTIONAL {{
    ?item schema:description ?description .
    FILTER(LANG(?description) = "{lang}")
  }}
}} LIMIT {limit}
"""

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        headers = {"Accept": "application/sparql-results+json"}
        response = requests.get(self.endpoint, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _convert_bindings(
        self, bindings: List[Dict[str, Any]], max_results: int
    ) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, binding in enumerate(bindings[:max_results]):
            item_url = binding.get("item", {}).get("value")
            label = binding.get("itemLabel", {}).get("value") or "Untitled"
            description = binding.get("description", {}).get("value")
            wikidata_id = self._extract_id(item_url)

            snippet = self._truncate(description or label)
            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "wikidata_id": wikidata_id,
                "description": description,
                "source_id": wikidata_id or item_url,
            }

            results.append(
                RemoteResult(
                    title=label,
                    url=item_url or label,
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
    def _extract_id(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        return url.rstrip("/").split("/")[-1]
