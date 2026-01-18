"""
Crossref Metadata API provider for remote search.

Provides DOI-based metadata retrieval and search for scholarly works via the
Crossref REST API - comprehensive bibliographic metadata for over 150 million
records.
API documentation: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests

from .base import BaseRemoteProvider, RemoteResult
from lsm.cli.logging import get_logger

logger = get_logger(__name__)


class CrossrefProvider(BaseRemoteProvider):
    """
    Remote provider using the Crossref Metadata API.

    Crossref is a DOI registration agency providing metadata for scholarly
    works including journal articles, books, conference proceedings, and more.
    No API key required for basic access (polite pool with email recommended).
    """

    API_ENDPOINT = "https://api.crossref.org"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.5  # Polite pool allows higher rate
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Crossref provider.

        Args:
            config: Configuration dict with optional keys:
                - email: Contact email for polite pool (recommended)
                - api_key: Crossref Metadata Plus API key (optional, for higher limits)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between API requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - year_from: Filter by publication year (start)
                - year_to: Filter by publication year (end)
                - type: Filter by work type (journal-article, book, etc.)
                - has_full_text: Only return works with full text links
                - has_references: Only return works with references
                - has_orcid: Only return works with ORCID IDs
        """
        super().__init__(config)

        self.email = config.get("email") or os.getenv("CROSSREF_EMAIL")
        self.api_key = config.get("api_key") or os.getenv("CROSSREF_API_KEY")
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
        self.year_from = config.get("year_from")
        self.year_to = config.get("year_to")
        self.work_type = config.get("type")
        self.has_full_text = config.get("has_full_text", False)
        self.has_references = config.get("has_references", False)
        self.has_orcid = config.get("has_orcid", False)
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "crossref"

    def is_available(self) -> bool:
        """API is always available - no key required for basic access."""
        return True

    def validate_config(self) -> None:
        """Validate Crossref configuration."""
        if self.year_from is not None:
            try:
                int(self.year_from)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid year_from: {self.year_from}. Must be an integer.")

        if self.year_to is not None:
            try:
                int(self.year_to)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid year_to: {self.year_to}. Must be an integer.")

        if self.work_type and self.work_type not in self._get_valid_types():
            logger.warning(f"Unknown work type: {self.work_type}. Query may return no results.")

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search Crossref for relevant scholarly works.

        Args:
            query: Search query string. Supports field-specific syntax:
                   - author:Name - Search by author name
                   - title:phrase - Search in titles
                   - doi:10.xxxx/xxx - Lookup by DOI
                   - orcid:0000-0000-0000-0000 - Search by ORCID
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"Crossref: query='{query}', max_results={max_results}")

        try:
            # Check for DOI lookup
            if query.lower().startswith("doi:"):
                doi = query[4:].strip()
                work = self._get_work_by_doi(doi)
                if work:
                    return self._convert_to_results([work], 1)
                return []

            works = self._search_works(query, max_results)
            results = self._convert_to_results(works, max_results)
            logger.info(f"Crossref returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"Crossref API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Crossref parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "Crossref"

    def get_work_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a work by DOI.

        Args:
            doi: DOI string (e.g., 10.1234/example)

        Returns:
            Work details dict or None if not found
        """
        return self._get_work_by_doi(doi)

    def get_journal_works(
        self, issn: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent works from a specific journal.

        Args:
            issn: Journal ISSN
            max_results: Maximum works to return

        Returns:
            List of work details
        """
        url = f"{self.endpoint}/journals/{issn}/works"
        params = {
            "rows": max_results,
            "sort": "published",
            "order": "desc",
        }

        try:
            data = self._request(url, params)
            return data.get("message", {}).get("items", [])
        except Exception as exc:
            logger.error(f"Error fetching journal works: {exc}")
            return []

    def get_funder_works(
        self, funder_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get works funded by a specific funder.

        Args:
            funder_id: Crossref funder ID
            max_results: Maximum works to return

        Returns:
            List of work details
        """
        url = f"{self.endpoint}/funders/{funder_id}/works"
        params = {
            "rows": max_results,
            "sort": "published",
            "order": "desc",
        }

        try:
            data = self._request(url, params)
            return data.get("message", {}).get("items", [])
        except Exception as exc:
            logger.error(f"Error fetching funder works: {exc}")
            return []

    def get_member_works(
        self, member_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get works from a specific Crossref member (publisher).

        Args:
            member_id: Crossref member ID
            max_results: Maximum works to return

        Returns:
            List of work details
        """
        url = f"{self.endpoint}/members/{member_id}/works"
        params = {
            "rows": max_results,
            "sort": "published",
            "order": "desc",
        }

        try:
            data = self._request(url, params)
            return data.get("message", {}).get("items", [])
        except Exception as exc:
            logger.error(f"Error fetching member works: {exc}")
            return []

    def _get_work_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific work by DOI."""
        # Clean DOI
        doi = doi.strip()
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        elif doi.startswith("doi:"):
            doi = doi[4:]

        url = f"{self.endpoint}/works/{doi}"

        try:
            data = self._request(url, {})
            return data.get("message")
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 404:
                logger.warning(f"DOI not found: {doi}")
            else:
                logger.error(f"Error fetching DOI: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Error fetching DOI: {exc}")
            return None

    def _search_works(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute work search against the API."""
        url = f"{self.endpoint}/works"

        params = {
            "rows": max_results,
            "sort": "relevance",
            "order": "desc",
        }

        # Parse field-specific queries
        filters = []

        if query.lower().startswith("author:"):
            author_name = query[7:].strip()
            params["query.author"] = author_name
        elif query.lower().startswith("title:"):
            title = query[6:].strip()
            params["query.title"] = title
        elif query.lower().startswith("orcid:"):
            orcid = query[6:].strip()
            filters.append(f"orcid:{orcid}")
        else:
            params["query"] = query

        # Add configured filters
        if self.year_from:
            filters.append(f"from-pub-date:{self.year_from}")
        if self.year_to:
            filters.append(f"until-pub-date:{self.year_to}")
        if self.work_type:
            filters.append(f"type:{self.work_type}")
        if self.has_full_text:
            filters.append("has-full-text:true")
        if self.has_references:
            filters.append("has-references:true")
        if self.has_orcid:
            filters.append("has-orcid:true")

        if filters:
            params["filter"] = ",".join(filters)

        data = self._request(url, params)
        return data.get("message", {}).get("items", [])

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request with throttling and error handling."""
        self._throttle()

        headers = {
            "Accept": "application/json",
        }

        # Add polite pool email or API key
        if self.api_key:
            headers["Crossref-Plus-API-Token"] = f"Bearer {self.api_key}"
        elif self.email:
            params["mailto"] = self.email

        # Set user agent for polite pool
        user_agent = f"LocalSecondMind/1.0 (mailto:{self.email or 'anonymous'})"
        headers["User-Agent"] = user_agent

        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
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

            # Get title (may be a list)
            title_list = work.get("title", [])
            title = title_list[0] if title_list else "Untitled"

            doi = work.get("DOI")
            url = f"https://doi.org/{doi}" if doi else work.get("URL", "")

            # Get abstract (if available)
            abstract = work.get("abstract", "")
            # Clean HTML from abstract
            if abstract:
                import re
                abstract = re.sub(r"<[^>]+>", "", abstract)

            # Extract authors
            authors = []
            for author in work.get("author", []):
                name_parts = []
                if author.get("given"):
                    name_parts.append(author["given"])
                if author.get("family"):
                    name_parts.append(author["family"])
                if name_parts:
                    authors.append(" ".join(name_parts))
                elif author.get("name"):
                    authors.append(author["name"])

            # Get publication year
            year = None
            published = work.get("published") or work.get("published-print") or work.get("published-online")
            if published:
                date_parts = published.get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]

            # Get citation count (if available)
            cited_by_count = work.get("is-referenced-by-count", 0) or 0

            # Calculate score based on position and citation count
            position_score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))
            citation_boost = min(0.15, cited_by_count / 10000)
            score = min(1.0, position_score + citation_boost)

            # Build snippet
            if abstract:
                snippet = self._truncate(abstract)
            else:
                author_str = ", ".join(authors[:3]) if authors else "Unknown"
                snippet = f"Paper by {author_str}"

            # Get container (journal/book) title
            container = work.get("container-title", [])
            venue = container[0] if container else None

            # Get publisher
            publisher = work.get("publisher")

            # Get ISSN and ISBN
            issn = work.get("ISSN", [])
            isbn = work.get("ISBN", [])

            # Get work type
            work_type = work.get("type")

            # Get links
            links = work.get("link", [])
            pdf_url = None
            for link in links:
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break

            # Get license info
            licenses = work.get("license", [])
            license_url = licenses[0].get("URL") if licenses else None

            # Get ORCID IDs
            orcids = []
            for author in work.get("author", []):
                if author.get("ORCID"):
                    orcids.append(author["ORCID"])

            # Get references count
            references_count = work.get("references-count", 0)

            # Get subject areas
            subjects = work.get("subject", [])

            metadata = {
                "doi": doi,
                "authors": authors,
                "year": year,
                "venue": venue,
                "publisher": publisher,
                "type": work_type,
                "issn": issn,
                "isbn": isbn,
                "cited_by_count": cited_by_count,
                "references_count": references_count,
                "subjects": subjects,
                "orcids": orcids,
                "license_url": license_url,
                "pdf_url": pdf_url,
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

    def _get_valid_types(self) -> set:
        """Return valid work types in Crossref."""
        return {
            "book",
            "book-chapter",
            "book-part",
            "book-section",
            "book-series",
            "book-set",
            "book-track",
            "component",
            "database",
            "dataset",
            "dissertation",
            "edited-book",
            "grant",
            "journal",
            "journal-article",
            "journal-issue",
            "journal-volume",
            "monograph",
            "other",
            "peer-review",
            "posted-content",
            "proceedings",
            "proceedings-article",
            "proceedings-series",
            "reference-book",
            "reference-entry",
            "report",
            "report-component",
            "report-series",
            "standard",
            "standard-series",
        }
