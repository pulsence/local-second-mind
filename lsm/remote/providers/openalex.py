"""
OpenAlex API provider for remote search.

Provides comprehensive academic paper search across all disciplines via the
OpenAlex API - a free and open catalog of scholarly works.
API documentation: https://docs.openalex.org/
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import quote

import requests

from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.logging import get_logger

logger = get_logger(__name__)


class OpenAlexProvider(BaseRemoteProvider):
    """
    Remote provider using the OpenAlex API.

    OpenAlex is a fully open catalog of the global research system,
    indexing over 240 million scholarly works across all disciplines.
    No API key required - completely free and open access.
    """

    API_ENDPOINT = "https://api.openalex.org"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.1  # 10 req/sec with email (polite pool)
    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAlex provider.

        Args:
            config: Configuration dict with optional keys:
                - email: Contact email for polite pool (recommended, increases rate limit)
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between API requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - year_from: Filter by publication year (start)
                - year_to: Filter by publication year (end)
                - open_access_only: Only return open access papers
                - type: Filter by work type (article, book, dataset, etc.)
                - concepts: Filter by OpenAlex concept IDs
        """
        super().__init__(config)

        self.email = config.get("email") or os.getenv("OPENALEX_EMAIL")
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
        self.open_access_only = config.get("open_access_only", False)
        self.work_type = self._resolve_work_type(config)
        self.concepts = config.get("concepts") or []
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "openalex"

    def is_available(self) -> bool:
        """API is always available - no key required."""
        return True

    def validate_config(self) -> None:
        """Validate OpenAlex configuration."""
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
        Search OpenAlex for relevant scholarly works.

        Args:
            query: Search query string. Supports field-specific syntax:
                   - author:Name - Search by author name
                   - title:phrase - Search in titles
                   - doi:10.xxxx/xxx - Search by DOI
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"OpenAlex: query='{query}', max_results={max_results}")

        try:
            works = self._search_works(query, max_results)
            results = self._convert_to_results(works, max_results)
            logger.info(f"OpenAlex returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"OpenAlex API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"OpenAlex parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "OpenAlex"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Academic search query.", "required": True},
            {"name": "title", "type": "string", "description": "Title phrase.", "required": False},
            {"name": "author", "type": "string", "description": "Author name.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Topic keywords.", "required": False},
            {"name": "doi", "type": "string", "description": "DOI.", "required": False},
            {"name": "year", "type": "integer", "description": "Publication year hint.", "required": False},
            {"name": "concepts", "type": "array[string]", "description": "OpenAlex concept filters.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields()

    def get_description(self) -> str:
        return "Open scholarly graph provider for papers, authors, venues, and concepts."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return super().search_structured(input_dict, max_results=max_results)

    def get_work_details(self, work_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific work.

        Args:
            work_id: OpenAlex work ID (e.g., W2741809807) or DOI

        Returns:
            Work details dict or None if not found
        """
        # Handle DOI format
        if work_id.startswith("10.") or work_id.startswith("https://doi.org/"):
            work_id = f"https://doi.org/{work_id}" if not work_id.startswith("https://") else work_id

        url = f"{self.endpoint}/works/{quote(work_id, safe='')}"

        try:
            data = self._request(url, {})
            return data
        except Exception as exc:
            logger.error(f"Error fetching work details: {exc}")
            return None

    def get_author_works(
        self, author_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get works by a specific author.

        Args:
            author_id: OpenAlex author ID (e.g., A5023888391)
            max_results: Maximum works to return

        Returns:
            List of work details
        """
        url = f"{self.endpoint}/works"
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "per_page": max_results,
            "sort": "cited_by_count:desc",
        }

        try:
            data = self._request(url, params)
            return data.get("results", [])
        except Exception as exc:
            logger.error(f"Error fetching author works: {exc}")
            return []

    def get_cited_by(
        self, work_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get works that cite a specific work.

        Args:
            work_id: OpenAlex work ID
            max_results: Maximum citations to return

        Returns:
            List of citing work details
        """
        url = f"{self.endpoint}/works"
        params = {
            "filter": f"cites:{work_id}",
            "per_page": max_results,
            "sort": "cited_by_count:desc",
        }

        try:
            data = self._request(url, params)
            return data.get("results", [])
        except Exception as exc:
            logger.error(f"Error fetching citations: {exc}")
            return []

    def _resolve_work_type(self, config: Dict[str, Any]) -> Optional[str]:
        """Resolve optional work type without clashing with provider config."""
        work_type = config.get("work_type")
        if work_type:
            return work_type
        fallback = config.get("type")
        if fallback and fallback not in {"openalex", "open-alex"}:
            return fallback
        return None

    def get_references(
        self, work_id: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get works referenced by a specific work.

        Args:
            work_id: OpenAlex work ID
            max_results: Maximum references to return

        Returns:
            List of referenced work details
        """
        # First get the work to find its referenced_works
        work = self.get_work_details(work_id)
        if not work:
            return []

        referenced_ids = work.get("referenced_works", [])[:max_results]
        if not referenced_ids:
            return []

        # Fetch details for referenced works
        url = f"{self.endpoint}/works"
        # Create filter for multiple work IDs
        id_filter = "|".join(referenced_ids)
        params = {
            "filter": f"openalex_id:{id_filter}",
            "per_page": max_results,
        }

        try:
            data = self._request(url, params)
            return data.get("results", [])
        except Exception as exc:
            logger.error(f"Error fetching references: {exc}")
            return []

    def _search_works(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute work search against the API."""
        url = f"{self.endpoint}/works"

        # Parse field-specific queries
        filters = []
        search_query = query

        # Check for field prefixes
        if query.lower().startswith("author:"):
            author_name = query[7:].strip()
            filters.append(f"authorships.author.display_name.search:{author_name}")
            search_query = None
        elif query.lower().startswith("title:"):
            title = query[6:].strip()
            filters.append(f"title.search:{title}")
            search_query = None
        elif query.lower().startswith("doi:"):
            doi = query[4:].strip()
            # Normalize DOI format
            if not doi.startswith("https://doi.org/"):
                doi = f"https://doi.org/{doi}"
            filters.append(f"doi:{doi}")
            search_query = None

        # Add configured filters
        if self.year_from:
            filters.append(f"publication_year:>{int(self.year_from) - 1}")
        if self.year_to:
            filters.append(f"publication_year:<{int(self.year_to) + 1}")
        if self.open_access_only:
            filters.append("open_access.is_oa:true")
        if self.work_type:
            if self.work_type in self._get_valid_types():
                filters.append(f"type:{self.work_type}")
            else:
                logger.warning(f"Ignoring invalid OpenAlex type filter: {self.work_type}")
        for concept in self.concepts:
            filters.append(f"concepts.id:{concept}")

        params = {
            "per_page": max_results,
            "sort": "relevance_score:desc",
        }

        if search_query:
            params["search"] = search_query

        if filters:
            params["filter"] = ",".join(filters)

        data = self._request(url, params)
        return data.get("results", [])

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request with throttling and error handling."""
        self._throttle()

        # Add email for polite pool (increases rate limit)
        if self.email:
            params["mailto"] = self.email

        headers = {
            "Accept": "application/json",
            "User-Agent": f"LocalSecondMind/1.0 (mailto:{self.email or 'anonymous'})",
        }

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

            title = work.get("title") or "Untitled"
            work_id = work.get("id", "")
            doi = work.get("doi")
            url = doi if doi else work_id

            # Get abstract
            abstract_inverted = work.get("abstract_inverted_index")
            abstract = self._reconstruct_abstract(abstract_inverted) if abstract_inverted else ""

            # Extract authors
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author and author.get("display_name"):
                    authors.append(author["display_name"])

            # Get publication year
            year = work.get("publication_year")

            # Calculate score based on position and citation count
            cited_by_count = work.get("cited_by_count", 0) or 0

            # Base position score (0.2 to 1.0)
            position_score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))

            # Boost for highly cited papers (up to 0.15 bonus)
            citation_boost = min(0.15, cited_by_count / 10000)

            score = min(1.0, position_score + citation_boost)

            # Build snippet from abstract
            snippet = self._truncate(abstract) if abstract else f"Paper by {', '.join(authors[:3]) or 'Unknown'}"

            # Extract topics/concepts
            topics = []
            for topic in work.get("topics", [])[:5]:
                if topic.get("display_name"):
                    topics.append(topic["display_name"])

            concepts = []
            for concept in work.get("concepts", [])[:5]:
                if concept.get("display_name"):
                    concepts.append(concept["display_name"])

            # Get open access info
            open_access = work.get("open_access", {})
            is_oa = open_access.get("is_oa", False)
            oa_url = open_access.get("oa_url")

            # Get source info
            primary_location = work.get("primary_location", {}) or {}
            source = primary_location.get("source", {}) or {}
            venue = source.get("display_name")

            # Get PDF URL if available
            pdf_url = None
            if primary_location.get("pdf_url"):
                pdf_url = primary_location["pdf_url"]
            elif oa_url:
                pdf_url = oa_url

            metadata = {
                "openalex_id": work_id,
                "authors": authors,
                "year": year,
                "venue": venue,
                "publication_date": work.get("publication_date"),
                "cited_by_count": cited_by_count,
                "topics": topics,
                "concepts": concepts,
                "type": work.get("type"),
                "is_open_access": is_oa,
                "oa_url": oa_url,
                "pdf_url": pdf_url,
                "doi": doi,
                "citation": self._format_citation(title, url, authors, year, doi),
            }

            results.append(
                RemoteResult(
                    title=title,
                    url=url or work_id,
                    snippet=snippet,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """
        Reconstruct abstract from OpenAlex's inverted index format.

        OpenAlex stores abstracts as inverted indices for space efficiency.
        """
        if not inverted_index:
            return ""

        # Find the total length needed
        max_pos = 0
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))

        # Reconstruct the text
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word

        return " ".join(words)

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
        doi_part = doi if doi else url
        return f"{author_str} ({year_str}). {title}. {doi_part} (accessed {date_str})."

    def _format_authors(self, authors: List[str]) -> str:
        """Format author list for citation."""
        if not authors:
            return "Unknown"
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + ", et al."

    def _get_valid_types(self) -> set:
        """Return valid work types in OpenAlex."""
        return {
            "article",
            "book",
            "book-chapter",
            "dataset",
            "dissertation",
            "editorial",
            "erratum",
            "letter",
            "monograph",
            "other",
            "paratext",
            "peer-review",
            "posted-content",
            "proceedings",
            "proceedings-article",
            "reference-entry",
            "report",
            "review",
            "standard",
        }
