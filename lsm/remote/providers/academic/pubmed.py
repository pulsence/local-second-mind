"""
PubMed provider using NCBI E-utilities.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.utils import normalize_doi

logger = get_logger(__name__)


class PubMedProvider(BaseRemoteProvider):
    """
    PubMed / PubMed Central provider via NCBI E-utilities.
    """

    API_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.34
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Optional keys:
                - api_key: NCBI API key for higher rate limits
                - email: Contact email
                - tool: Tool name for NCBI tracking
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds
                - min_interval_seconds: Minimum seconds between API requests
                - snippet_max_chars: Max snippet length
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("PUBMED_API_KEY")
        self.email = config.get("email") or os.getenv("PUBMED_EMAIL")
        self.tool = config.get("tool") or "LocalSecondMind"
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
        return "pubmed"

    def is_available(self) -> bool:
        return True

    def get_name(self) -> str:
        return "PubMed"

    def get_description(self) -> str:
        return "PubMed biomedical literature via NCBI E-utilities."

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Biomedical search query.", "required": True},
            {"name": "title", "type": "string", "description": "Title phrase.", "required": False},
            {"name": "author", "type": "string", "description": "Author name.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Topic keywords.", "required": False},
            {"name": "doi", "type": "string", "description": "DOI hint.", "required": False},
            {"name": "pmid", "type": "string", "description": "PubMed ID.", "required": False},
            {"name": "year", "type": "integer", "description": "Publication year hint.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "pmid", "type": "string", "description": "PubMed ID."},
            {"name": "pmcid", "type": "string", "description": "PubMed Central ID."},
            {"name": "journal", "type": "string", "description": "Journal title."},
            {"name": "publication_date", "type": "string", "description": "Publication date."},
            {"name": "pmc_url", "type": "string", "description": "PMC full text URL."},
            {"name": "pdf_url", "type": "string", "description": "PMC PDF URL."},
            {"name": "citation", "type": "string", "description": "Formatted citation string."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not query.strip():
            return []

        try:
            pmids = self._esearch(query, max_results)
            summaries = self._esummary(pmids)
            results = self._convert_to_results(summaries, max_results)
            logger.info(f"PubMed returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"PubMed API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"PubMed parsing error: {exc}")
            return []

    def _esearch(self, query: str, max_results: int) -> List[str]:
        url = f"{self.endpoint}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
        }
        self._apply_common_params(params)
        data = self._request(url, params)
        return data.get("esearchresult", {}).get("idlist", []) or []

    def _esummary(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        url = f"{self.endpoint}/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        self._apply_common_params(params)
        data = self._request(url, params)
        result = data.get("result", {}) if isinstance(data, dict) else {}
        uids = result.get("uids", []) if isinstance(result, dict) else []
        summaries: List[Dict[str, Any]] = []
        for uid in uids:
            item = result.get(uid)
            if isinstance(item, dict):
                summaries.append(item)
        return summaries

    def _convert_to_results(
        self, summaries: List[Dict[str, Any]], max_results: int
    ) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        for idx, item in enumerate(summaries[:max_results]):
            pmid = str(item.get("uid") or "").strip()
            title = item.get("title") or "Untitled"
            authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
            pub_date = item.get("pubdate") or item.get("sortpubdate") or ""
            year = self._extract_year(pub_date)
            journal = item.get("fulljournalname") or item.get("source")

            doi = None
            pmcid = None
            for article_id in item.get("articleids", []) or []:
                id_type = article_id.get("idtype")
                value = article_id.get("value")
                if id_type == "doi":
                    doi = normalize_doi(value)
                elif id_type == "pmcid":
                    pmcid = value

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            pmc_url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else None
            )
            pdf_url = (
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/" if pmcid else None
            )

            snippet = self._truncate(item.get("elocationid") or title)

            score = max(0.2, 1.0 - (idx * 0.8 / max(1, max_results - 1)))

            metadata = {
                "pmid": pmid or None,
                "pmcid": pmcid,
                "doi": doi,
                "authors": authors,
                "year": year,
                "journal": journal,
                "publication_date": pub_date or None,
                "pmc_url": pmc_url,
                "pdf_url": pdf_url,
                "citation": self._format_citation(title, url, authors, year, doi),
                "source_id": doi or pmid or url,
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or pmc_url or (f"https://doi.org/{doi}" if doi else pmid),
                    snippet=snippet,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def _apply_common_params(self, params: Dict[str, Any]) -> None:
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

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
    def _extract_year(value: str) -> Optional[int]:
        if not value:
            return None
        for token in str(value).split():
            if token.isdigit() and len(token) == 4:
                try:
                    return int(token)
                except ValueError:
                    return None
        try:
            return int(str(value)[:4])
        except ValueError:
            return None

    def _format_citation(
        self,
        title: str,
        url: str,
        authors: List[str],
        year: Optional[int],
        doi: Optional[str],
    ) -> str:
        year_str = str(year) if year else "n.d."
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        author_str = self._format_authors(authors)
        doi_part = f"https://doi.org/{doi}" if doi else url
        return f"{author_str} ({year_str}). {title}. {doi_part} (accessed {date_str})."

    @staticmethod
    def _format_authors(authors: List[str]) -> str:
        if not authors:
            return "Unknown"
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + ", et al."
