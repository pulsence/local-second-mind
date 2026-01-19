"""
CORE API provider for remote search.

Provides access to open access research papers from repositories worldwide
via the CORE API.
API documentation: https://core.ac.uk/documentation/api
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


class COREProvider(BaseRemoteProvider):
    """
    Remote provider using the CORE API.

    CORE aggregates open access research outputs from repositories
    and journals worldwide, providing access to millions of full-text papers.
    """

    API_ENDPOINT = "https://api.core.ac.uk/v3"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 1.0
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CORE provider.

        Args:
            config: Configuration dict with optional keys:
                - api_key: CORE API key (required for higher rate limits)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between API requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - repository_ids: Filter by specific repository IDs
                - year_from: Filter papers from this year onwards
                - year_to: Filter papers up to this year
                - full_text_only: Only return papers with full text available
                - language: Filter by language code (e.g., "en")
        """
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("CORE_API_KEY")
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
        self.repository_ids = config.get("repository_ids") or []
        self.year_from = config.get("year_from")
        self.year_to = config.get("year_to")
        self.full_text_only = config.get("full_text_only", False)
        self.language = config.get("language")
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "core"

    def is_available(self) -> bool:
        """Check if API key is configured (required for CORE API v3)."""
        return bool(self.api_key)

    def validate_config(self) -> None:
        """Validate CORE configuration."""
        if not self.api_key:
            raise ValueError(
                "CORE provider requires an API key. "
                "Register at https://core.ac.uk/services/api to get one."
            )
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from cannot be greater than year_to")

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search CORE for relevant papers.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"CORE: query='{query}', max_results={max_results}")

        try:
            works = self._search_works(query, max_results)
            results = self._convert_to_results(works, max_results)
            logger.info(f"CORE returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"CORE API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"CORE parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "CORE"

    def get_work_details(self, work_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific work.

        Args:
            work_id: CORE work ID

        Returns:
            Work details dict or None if not found
        """
        url = f"{self.endpoint}/works/{work_id}"

        try:
            data = self._request(url)
            return data
        except Exception as exc:
            logger.error(f"Error fetching work details: {exc}")
            return None

    def get_full_text(self, work_id: str) -> Optional[str]:
        """
        Get the full text of a work if available.

        Args:
            work_id: CORE work ID

        Returns:
            Full text string or None if not available
        """
        details = self.get_work_details(work_id)
        if details:
            return details.get("fullText")
        return None

    def search_by_repository(
        self, repository_id: str, query: str = "", max_results: int = 10
    ) -> List[RemoteResult]:
        """
        Search within a specific repository.

        Args:
            repository_id: CORE repository ID
            query: Optional search query
            max_results: Maximum results to return

        Returns:
            List of RemoteResult objects
        """
        url = f"{self.endpoint}/search/works"
        search_query = f"repositories.id:{repository_id}"
        if query:
            search_query = f"({query}) AND {search_query}"

        params = {
            "q": search_query,
            "limit": max_results,
        }

        try:
            data = self._request(url, params)
            works = data.get("results", [])
            return self._convert_to_results(works, max_results)
        except Exception as exc:
            logger.error(f"Error searching repository: {exc}")
            return []

    def get_repository_info(self, repository_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a repository.

        Args:
            repository_id: CORE repository ID

        Returns:
            Repository info dict or None
        """
        url = f"{self.endpoint}/data-providers/{repository_id}"

        try:
            return self._request(url)
        except Exception as exc:
            logger.error(f"Error fetching repository info: {exc}")
            return None

    def _search_works(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute work search against the API."""
        url = f"{self.endpoint}/search/works"

        # Build query with filters
        search_query = self._build_query(query)

        params = {
            "q": search_query,
            "limit": max_results,
        }

        data = self._request(url, params)
        return data.get("results", [])

    def _build_query(self, query: str) -> str:
        """Build search query with filters."""
        parts = [query]

        if self.year_from:
            parts.append(f"yearPublished>={self.year_from}")

        if self.year_to:
            parts.append(f"yearPublished<={self.year_to}")

        if self.full_text_only:
            parts.append("fullText:*")

        if self.language:
            parts.append(f"language.code:{self.language}")

        if self.repository_ids:
            repo_filter = " OR ".join(
                f"repositories.id:{rid}" for rid in self.repository_ids
            )
            parts.append(f"({repo_filter})")

        return " AND ".join(f"({p})" for p in parts)

    def _request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an API request with throttling and error handling."""
        self._throttle()

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=self.timeout
        )
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

    def _convert_to_results(
        self, works: List[Dict[str, Any]], max_results: int
    ) -> List[RemoteResult]:
        """Convert API response to RemoteResult objects."""
        results: List[RemoteResult] = []

        for i, work in enumerate(works[:max_results]):
            if not work:
                continue

            title = work.get("title") or "Untitled"
            abstract = work.get("abstract") or ""
            work_id = work.get("id", "")

            # Get the download URL or CORE page URL
            download_url = work.get("downloadUrl")
            core_url = f"https://core.ac.uk/works/{work_id}"
            url = download_url or core_url

            year = work.get("yearPublished")

            # Extract authors
            authors = []
            for author in work.get("authors", []):
                if author and author.get("name"):
                    authors.append(author["name"])

            # Calculate score based on position
            score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))

            # Build snippet from abstract
            snippet = (
                self._truncate(abstract)
                if abstract
                else f"Work by {', '.join(authors[:3]) or 'Unknown'}"
            )

            # Extract additional metadata
            doi = None
            identifiers = work.get("identifiers", [])
            for ident in identifiers:
                if ident and ident.get("type") == "doi":
                    doi = ident.get("identifier")
                    break

            # Get repository info
            repositories = work.get("repositories", [])
            repository_names = [
                repo.get("name") for repo in repositories if repo and repo.get("name")
            ]

            # Check for full text availability
            has_full_text = bool(work.get("fullText"))
            full_text_url = work.get("fullTextUrl")

            # Get language
            language_info = work.get("language")
            language = language_info.get("code") if language_info else None

            metadata = {
                "core_id": work_id,
                "authors": authors,
                "year": year,
                "doi": doi,
                "repositories": repository_names,
                "has_full_text": has_full_text,
                "full_text_url": full_text_url,
                "download_url": download_url,
                "language": language,
                "publisher": work.get("publisher"),
                "journal": work.get("journals", [{}])[0].get("title")
                if work.get("journals")
                else None,
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
