"""
Semantic Scholar API provider for remote search.

Provides academic paper search across computer science, neuroscience, and
other disciplines via the Semantic Scholar API.
API documentation: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests

from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.logging import get_logger

logger = get_logger(__name__)


class SemanticScholarProvider(BaseRemoteProvider):
    """
    Remote provider using the Semantic Scholar API.

    Supports search for academic papers across multiple disciplines,
    with rich citation metadata and influential citation metrics.
    """

    API_ENDPOINT = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 1.0  # S2 allows ~100 requests/5 min without key
    DEFAULT_SNIPPET_MAX_CHARS = 700
    DEFAULT_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "url",
        "year",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "venue",
        "publicationDate",
        "externalIds",
        "isOpenAccess",
        "openAccessPdf",
        "fieldsOfStudy",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Semantic Scholar provider.

        Args:
            config: Configuration dict with optional keys:
                - api_key: Semantic Scholar API key (increases rate limit)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between API requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - fields_of_study: Filter by fields (e.g., ["Computer Science"])
                - year_range: Filter by year range (e.g., "2020-2024")
                - open_access_only: Only return open access papers
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
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
        self.fields_of_study = config.get("fields_of_study") or []
        self.year_range = config.get("year_range")
        self.open_access_only = config.get("open_access_only", False)
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "semantic_scholar"

    def is_available(self) -> bool:
        """API is available without key, but rate-limited."""
        return True

    def validate_config(self) -> None:
        """Validate Semantic Scholar configuration."""
        if self.year_range:
            if not self._parse_year_range(self.year_range):
                raise ValueError(
                    f"Invalid year_range format: {self.year_range}. Use 'YYYY-YYYY' or 'YYYY-'."
                )

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search Semantic Scholar for relevant papers.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"Semantic Scholar: query='{query}', max_results={max_results}")

        try:
            papers = self._search_papers(query, max_results)
            results = self._convert_to_results(papers, max_results)
            logger.info(f"Semantic Scholar returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Semantic Scholar API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Semantic Scholar parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "Semantic Scholar"

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, or arXiv ID

        Returns:
            Paper details dict or None if not found
        """
        url = f"{self.endpoint}/paper/{paper_id}"
        params = {"fields": ",".join(self.DEFAULT_FIELDS + ["references", "citations"])}

        try:
            data = self._request(url, params)
            return data
        except Exception as exc:
            logger.error(f"Error fetching paper details: {exc}")
            return None

    def get_paper_citations(
        self, paper_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get papers that cite a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum citations to return

        Returns:
            List of citing paper details
        """
        url = f"{self.endpoint}/paper/{paper_id}/citations"
        params = {
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": max_results,
        }

        try:
            data = self._request(url, params)
            return [item.get("citingPaper", {}) for item in data.get("data", [])]
        except Exception as exc:
            logger.error(f"Error fetching citations: {exc}")
            return []

    def get_paper_references(
        self, paper_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get papers referenced by a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum references to return

        Returns:
            List of referenced paper details
        """
        url = f"{self.endpoint}/paper/{paper_id}/references"
        params = {
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": max_results,
        }

        try:
            data = self._request(url, params)
            return [item.get("citedPaper", {}) for item in data.get("data", [])]
        except Exception as exc:
            logger.error(f"Error fetching references: {exc}")
            return []

    def get_author_papers(
        self, author_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get papers by a specific author.

        Args:
            author_id: Semantic Scholar author ID
            max_results: Maximum papers to return

        Returns:
            List of paper details
        """
        url = f"{self.endpoint}/author/{author_id}/papers"
        params = {
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": max_results,
        }

        try:
            data = self._request(url, params)
            return data.get("data", [])
        except Exception as exc:
            logger.error(f"Error fetching author papers: {exc}")
            return []

    def get_recommendations(
        self, paper_id: str, max_results: int = 5
    ) -> List[RemoteResult]:
        """
        Get paper recommendations based on a seed paper.

        Args:
            paper_id: Semantic Scholar paper ID to base recommendations on
            max_results: Maximum recommendations to return

        Returns:
            List of recommended papers as RemoteResult objects
        """
        url = f"{self.endpoint}/recommendations/v1/papers/forpaper/{paper_id}"
        params = {
            "fields": ",".join(self.DEFAULT_FIELDS),
            "limit": max_results,
        }

        try:
            data = self._request(url, params)
            papers = data.get("recommendedPapers", [])
            return self._convert_to_results(papers, max_results)
        except Exception as exc:
            logger.error(f"Error fetching recommendations: {exc}")
            return []

    def _search_papers(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute paper search against the API."""
        url = f"{self.endpoint}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(self.DEFAULT_FIELDS),
        }

        # Add optional filters
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)

        if self.year_range:
            year_start, year_end = self._parse_year_range(self.year_range)
            if year_start:
                params["year"] = f"{year_start}-" if not year_end else f"{year_start}-{year_end}"

        if self.open_access_only:
            params["openAccessPdf"] = ""

        data = self._request(url, params)
        return data.get("data", [])

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request with throttling and error handling."""
        self._throttle()

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        retry_statuses = {429, 500, 502, 503, 504}
        max_retries = 2

        for attempt in range(max_retries + 1):
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            if response.status_code in retry_statuses and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()

        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        """Enforce rate limiting between requests."""
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _parse_year_range(self, year_range: str) -> Optional[tuple]:
        """Parse year range string (e.g., '2020-2024' or '2020-')."""
        if not year_range:
            return None
        parts = year_range.split("-")
        if len(parts) != 2:
            return None
        try:
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if parts[1] else None
            return (start, end)
        except ValueError:
            return None

    def _convert_to_results(
        self, papers: List[Dict[str, Any]], max_results: int
    ) -> List[RemoteResult]:
        """Convert API response to RemoteResult objects."""
        results: List[RemoteResult] = []

        for i, paper in enumerate(papers[:max_results]):
            if not paper:
                continue

            title = paper.get("title") or "Untitled"
            abstract = paper.get("abstract") or ""
            paper_id = paper.get("paperId", "")
            url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"
            year = paper.get("year")

            authors = []
            for author in paper.get("authors", []):
                if author and author.get("name"):
                    authors.append(author["name"])

            # Calculate score based on position and citation metrics
            citation_count = paper.get("citationCount", 0) or 0
            influential_count = paper.get("influentialCitationCount", 0) or 0

            # Base position score (0.2 to 1.0)
            position_score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))

            # Boost for highly cited papers (up to 0.1 bonus)
            citation_boost = min(0.1, citation_count / 10000)

            # Boost for influential citations (up to 0.1 bonus)
            influential_boost = min(0.1, influential_count / 100)

            score = min(1.0, position_score + citation_boost + influential_boost)

            # Build snippet from abstract
            snippet = self._truncate(abstract) if abstract else f"Paper by {', '.join(authors[:3]) or 'Unknown'}"

            # Extract external IDs
            external_ids = paper.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")

            # Get open access PDF URL if available
            open_access_pdf = paper.get("openAccessPdf")
            pdf_url = open_access_pdf.get("url") if open_access_pdf else None

            metadata = {
                "paper_id": paper_id,
                "authors": authors,
                "year": year,
                "venue": paper.get("venue"),
                "publication_date": paper.get("publicationDate"),
                "citation_count": citation_count,
                "influential_citation_count": influential_count,
                "fields_of_study": paper.get("fieldsOfStudy", []),
                "is_open_access": paper.get("isOpenAccess", False),
                "pdf_url": pdf_url,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "citation": self._format_citation(title, url, authors, year, doi),
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def _truncate(self, text: str) -> str:
        """Truncate text to snippet length."""
        if len(text) <= self.snippet_max_chars:
            return text
        return text[: self.snippet_max_chars].rstrip() + "..."

    def _format_citation(
        self,
        title: str,
        url: str,
        authors: List[str],
        year: Optional[int],
        doi: Optional[str],
    ) -> str:
        """Format an academic citation string."""
        year_str = str(year) if year else "n.d."
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        author_str = self._format_authors(authors)
        doi_part = f"https://doi.org/{doi}" if doi else url
        return f"{author_str} ({year_str}). {title}. {doi_part} (accessed {date_str})."

    def _format_authors(self, authors: List[str]) -> str:
        """Format author list for citation."""
        if not authors:
            return "Unknown"
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + ", et al."
