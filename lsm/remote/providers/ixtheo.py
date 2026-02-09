"""
Index Theologicus (IxTheo) provider for remote search.

Provides access to theology and religious studies literature from
the Index Theologicus database at the University of Tübingen.
Website: https://ixtheo.de
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests

from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.logging import get_logger

logger = get_logger(__name__)


class IxTheoProvider(BaseRemoteProvider):
    """
    Remote provider for Index Theologicus (IxTheo).

    IxTheo is an international scientific open access bibliography for
    theology and religious studies, maintained by the University Library
    of Tübingen. It covers Christianity as well as dialogue with other
    religions, across confessions, languages, and media types.

    The system is built on VuFind and provides a search API.

    Supported languages for subject headings and search:
    - English, German, French, Italian, Spanish
    - Portuguese, Greek, Russian, Chinese

    Special features:
    - Bible passage search using standard abbreviations
    - Denominational classification
    - Religious studies across traditions
    """

    API_ENDPOINT = "https://ixtheo.de/Search/Results"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 1.0
    DEFAULT_SNIPPET_MAX_CHARS = 700

    # IxTheo search types (VuFind search handlers)
    SEARCH_TYPES = {
        "all": "AllFields",
        "title": "Title",
        "author": "Author",
        "subject": "Subject",
        "series": "Series",
        "toc": "TableOfContents",
        "isbn": "ISN",
        "publisher": "Publisher",
    }

    # Denominational/tradition classifications
    TRADITIONS = {
        "christian": "Christianity",
        "catholic": "Catholic",
        "protestant": "Protestant",
        "orthodox": "Orthodox",
        "jewish": "Judaism",
        "islamic": "Islam",
        "buddhist": "Buddhism",
        "hindu": "Hinduism",
        "comparative": "Comparative Religion",
        "secular": "Religious Studies",
    }

    # Language codes
    LANGUAGES = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "it": "Italian",
        "es": "Spanish",
        "pt": "Portuguese",
        "el": "Greek",
        "ru": "Russian",
        "zh": "Chinese",
        "la": "Latin",
        "he": "Hebrew",
        "ar": "Arabic",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize IxTheo provider.

        Args:
            config: Configuration dict with optional keys:
                - endpoint: Custom API endpoint
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - language: Preferred language for results (e.g., "en", "de")
                - traditions: Filter by religious traditions
                - search_type: Default search type (default: "all")
                - include_reviews: Include book reviews (default: True)
                - year_from: Filter from this year
                - year_to: Filter up to this year
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
        self.traditions = config.get("traditions") or []
        self.search_type = config.get("search_type") or "all"
        self.include_reviews = config.get("include_reviews", True)
        self.year_from = config.get("year_from")
        self.year_to = config.get("year_to")
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "ixtheo"

    def is_available(self) -> bool:
        """IxTheo is freely available without API key."""
        return True

    def validate_config(self) -> None:
        """Validate IxTheo configuration."""
        if self.search_type and self.search_type.lower() not in self.SEARCH_TYPES:
            valid_types = ", ".join(sorted(self.SEARCH_TYPES.keys()))
            raise ValueError(
                f"Invalid search_type: {self.search_type}. "
                f"Valid types are: {valid_types}"
            )

        if self.language and self.language.lower() not in self.LANGUAGES:
            valid_langs = ", ".join(sorted(self.LANGUAGES.keys()))
            raise ValueError(
                f"Invalid language: {self.language}. "
                f"Valid languages are: {valid_langs}"
            )

        invalid_traditions = [
            t for t in self.traditions
            if t.lower() not in self.TRADITIONS
        ]
        if invalid_traditions:
            valid_trads = ", ".join(sorted(self.TRADITIONS.keys()))
            raise ValueError(
                f"Invalid traditions: {invalid_traditions}. "
                f"Valid traditions are: {valid_trads}"
            )

        if self.year_from and self.year_to:
            if int(self.year_from) > int(self.year_to):
                raise ValueError("year_from cannot be greater than year_to")

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search IxTheo for theology and religious studies literature.

        Args:
            query: Search query string. Supports:
                - Regular text search
                - Bible references (e.g., "Gen 1:1", "Mt 5:1-12")
                - author:Name - search by author
                - title:phrase - search in titles
                - subject:topic - search by subject
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"IxTheo: query='{query}', max_results={max_results}")

        try:
            # Parse query for field-specific searches
            search_params = self._build_search_params(query, max_results)

            # Execute search
            results = self._execute_search(search_params, max_results)

            logger.info(f"IxTheo returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"IxTheo API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"IxTheo parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "Index Theologicus"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Theology/religious studies query.", "required": True},
            {"name": "title", "type": "string", "description": "Title phrase.", "required": False},
            {"name": "author", "type": "string", "description": "Author name.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Topic keywords.", "required": False},
            {"name": "subject", "type": "string", "description": "Subject/tradition hint.", "required": False},
            {"name": "year", "type": "integer", "description": "Publication year hint.", "required": False},
            {"name": "language", "type": "string", "description": "Language code.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "ixtheo_id", "type": "string", "description": "IxTheo record ID."},
            {"name": "subjects", "type": "array[string]", "description": "Thematic subject labels."},
            {"name": "source", "type": "string", "description": "Source origin label."},
            {"name": "language", "type": "string", "description": "Language code for result."},
            {"name": "citation", "type": "string", "description": "Formatted citation string."},
        ]

    def get_description(self) -> str:
        return "Theology and religious studies provider using Index Theologicus."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return super().search_structured(input_dict, max_results=max_results)

    def search_bible_passage(
        self, reference: str, max_results: int = 10
    ) -> List[RemoteResult]:
        """
        Search for literature about a specific Bible passage.

        Args:
            reference: Bible reference (e.g., "Gen 1:1-3", "Mt 5:1-12", "Rom 8")
            max_results: Maximum results to return

        Returns:
            List of RemoteResult objects
        """
        # IxTheo has special handling for Bible references
        # Format: book chapter:verse-verse
        return self.search(f'"{reference}"', max_results)

    def search_by_tradition(
        self, tradition: str, query: str = "", max_results: int = 10
    ) -> List[RemoteResult]:
        """
        Search within a specific religious tradition.

        Args:
            tradition: Tradition key (e.g., "catholic", "protestant", "jewish")
            query: Optional additional search terms
            max_results: Maximum results to return

        Returns:
            List of RemoteResult objects
        """
        tradition_lower = tradition.lower()
        if tradition_lower not in self.TRADITIONS:
            logger.warning(f"Unknown tradition: {tradition}")
            return self.search(query, max_results) if query else []

        # Add tradition to temporary filter
        original_traditions = self.traditions
        self.traditions = [tradition_lower]

        try:
            return self.search(query or "*", max_results)
        finally:
            self.traditions = original_traditions

    def list_traditions(self) -> Dict[str, str]:
        """
        Get available religious tradition filters.

        Returns:
            Dict mapping short names to full tradition names
        """
        return dict(self.TRADITIONS)

    def list_languages(self) -> Dict[str, str]:
        """
        Get available languages.

        Returns:
            Dict mapping language codes to full names
        """
        return dict(self.LANGUAGES)

    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific record by IxTheo ID.

        Args:
            record_id: IxTheo record identifier

        Returns:
            Record details dict or None if not found
        """
        url = f"https://ixtheo.de/Record/{record_id}"

        try:
            # Use the export endpoint to get structured data
            export_url = f"https://ixtheo.de/Record/{record_id}/Export"
            params = {"style": "RIS"}

            self._throttle()
            response = requests.get(
                export_url,
                params=params,
                timeout=self.timeout,
                headers={"Accept": "application/x-research-info-systems"},
            )

            if response.status_code == 200:
                return self._parse_ris_response(response.text, record_id, url)
            return None
        except Exception as exc:
            logger.error(f"Error fetching record details: {exc}")
            return None

    def _build_search_params(
        self, query: str, max_results: int
    ) -> Dict[str, Any]:
        """Build search parameters from query."""
        params = {
            "limit": max_results,
        }

        # Parse for field prefixes
        parsed = self._parse_query(query)
        search_type = parsed.get("type") or self.search_type

        # Set search type
        if search_type in self.SEARCH_TYPES:
            params["type"] = self.SEARCH_TYPES[search_type]
        else:
            params["type"] = "AllFields"

        # Set main query
        params["lookfor"] = parsed.get("lookfor", query)

        # Add filters
        filters = []

        # Language filter (only when explicitly configured)
        if self.language:
            language_label = self._normalize_language(self.language)
            if language_label:
                filters.append(f'language:"{language_label}"')

        # Year filters
        if self.year_from:
            filters.append(f"publishDate:[{self.year_from} TO *]")
        if self.year_to:
            filters.append(f"publishDate:[* TO {self.year_to}]")

        # Tradition filters
        for tradition in self.traditions:
            if tradition in self.TRADITIONS:
                filters.append(f'topic:"{self.TRADITIONS[tradition]}"')

        if filters:
            params["filter"] = filters

        return params

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query string for field-specific prefixes."""
        parsed = {
            "lookfor": query,
            "type": None,
        }

        # Check for field prefixes
        query_lower = query.lower()

        if query_lower.startswith("author:"):
            parsed["lookfor"] = query[7:].strip()
            parsed["type"] = "author"
        elif query_lower.startswith("title:"):
            parsed["lookfor"] = query[6:].strip()
            parsed["type"] = "title"
        elif query_lower.startswith("subject:"):
            parsed["lookfor"] = query[8:].strip()
            parsed["type"] = "subject"
        elif query_lower.startswith("isbn:"):
            parsed["lookfor"] = query[5:].strip()
            parsed["type"] = "isbn"

        return parsed

    def _execute_search(
        self, params: Dict[str, Any], max_results: int
    ) -> List[RemoteResult]:
        """Execute search request and parse results."""
        self._throttle()

        # Build URL parameters for HTML scraping
        # IxTheo doesn't have a public JSON API, so we parse HTML results
        search_params = {
            "lookfor": params.get("lookfor", ""),
            "type": params.get("type", "AllFields"),
            "limit": params.get("limit", max_results),
            "view": "rss",  # RSS format is easier to parse
        }

        # Add filters if present
        if params.get("filter"):
            search_params["filter[]"] = params["filter"]

        headers = {
            "Accept": "application/rss+xml, application/xml",
            "User-Agent": "LocalSecondMind/1.0 (Theology Research Tool)",
        }

        response = requests.get(
            self.endpoint,
            params=search_params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return self._parse_rss_response(response.text, max_results)

    def _parse_rss_response(self, rss_text: str, max_results: int) -> List[RemoteResult]:
        """Parse RSS response from IxTheo search."""
        import xml.etree.ElementTree as ET

        results = []

        try:
            root = ET.fromstring(rss_text)

            # Find all items in the RSS feed
            items = root.findall(".//item")

            for i, item in enumerate(items[:max_results]):
                result = self._parse_rss_item(item, i, max_results)
                if result:
                    results.append(result)

        except ET.ParseError as exc:
            logger.warning(f"Error parsing RSS response: {exc}")
            # Try fallback parsing
            results = self._parse_html_fallback(rss_text, max_results)

        return results

    def _parse_rss_item(
        self, item, index: int, max_results: int
    ) -> Optional[RemoteResult]:
        """Parse a single RSS item into RemoteResult."""
        import xml.etree.ElementTree as ET

        try:
            title = item.findtext("title") or "Untitled"
            link = item.findtext("link") or ""
            description = item.findtext("description") or ""

            # Extract author from dc:creator if available
            dc_ns = {"dc": "http://purl.org/dc/elements/1.1/"}
            authors = []
            for creator in item.findall("dc:creator", dc_ns):
                if creator.text:
                    authors.append(creator.text)

            # Also check regular author field
            author_elem = item.findtext("author")
            if author_elem and author_elem not in authors:
                authors.append(author_elem)

            # Get publication date
            pub_date = item.findtext("pubDate") or ""
            year = self._extract_year(pub_date)

            # Extract IxTheo record ID from link
            record_id = self._extract_record_id(link)

            # Build snippet
            if description:
                snippet = self._truncate(self._clean_html(description))
            elif authors:
                snippet = f"Theology work by {', '.join(authors[:3])}"
                if len(authors) > 3:
                    snippet += " et al."
            else:
                snippet = "Theology and religious studies resource from IxTheo"

            # Calculate score
            score = max(0.2, 1.0 - (index * 0.8 / max(1, max_results - 1)))

            # Get categories/subjects
            subjects = []
            for category in item.findall("category"):
                if category.text:
                    subjects.append(category.text)

            metadata = {
                "ixtheo_id": record_id,
                "authors": authors,
                "year": year,
                "subjects": subjects,
                "source": "Index Theologicus",
                "language": self.language,
                "citation": self._format_citation(title, link, authors, year, None),
            }

            return RemoteResult(
                title=title,
                url=link,
                snippet=snippet,
                score=score,
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug(f"Error parsing RSS item: {exc}")
            return None

    def _parse_html_fallback(
        self, html_text: str, max_results: int
    ) -> List[RemoteResult]:
        """Fallback HTML parsing if RSS fails."""
        results = []

        # Simple regex-based extraction for basic info
        # This is a fallback - ideally RSS works
        title_pattern = re.compile(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)</a>')
        link_pattern = re.compile(r'href="(/Record/[^"]+)"')

        titles = title_pattern.findall(html_text)
        links = link_pattern.findall(html_text)

        for i, (title, link) in enumerate(zip(titles[:max_results], links[:max_results])):
            score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))
            full_url = f"https://ixtheo.de{link}" if link.startswith("/") else link

            results.append(RemoteResult(
                title=title,
                url=full_url,
                snippet="Theology resource from Index Theologicus",
                score=score,
                metadata={
                    "ixtheo_id": self._extract_record_id(link),
                    "source": "Index Theologicus",
                },
            ))

        return results

    def _parse_ris_response(
        self, ris_text: str, record_id: str, url: str
    ) -> Dict[str, Any]:
        """Parse RIS format response."""
        result = {
            "ixtheo_id": record_id,
            "url": url,
            "authors": [],
        }

        for line in ris_text.split("\n"):
            line = line.strip()
            if not line or len(line) < 6:
                continue

            tag = line[:2]
            value = line[6:] if len(line) > 6 else ""

            if tag == "TI":
                result["title"] = value
            elif tag == "AU":
                result["authors"].append(value)
            elif tag == "PY":
                result["year"] = self._extract_year(value)
            elif tag == "AB":
                result["abstract"] = value
            elif tag == "KW":
                if "subjects" not in result:
                    result["subjects"] = []
                result["subjects"].append(value)
            elif tag == "DO":
                result["doi"] = value
            elif tag == "PB":
                result["publisher"] = value
            elif tag == "JO" or tag == "T2":
                result["journal"] = value

        return result

    def _extract_record_id(self, url: str) -> str:
        """Extract IxTheo record ID from URL."""
        # URL format: https://ixtheo.de/Record/XXXXX
        match = re.search(r"/Record/([^/?]+)", url)
        if match:
            return match.group(1)
        return ""

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return int(match.group())
        return None

    def _clean_html(self, html: str) -> str:
        """Remove HTML tags from string."""
        clean = re.sub(r"<[^>]+>", "", html)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _normalize_language(self, value: str) -> Optional[str]:
        """Normalize language filter to IxTheo labels."""
        if not value:
            return None
        lower = value.lower()
        if lower in self.LANGUAGES:
            return self.LANGUAGES[lower]
        return value

    def _throttle(self) -> None:
        """Enforce rate limiting between requests."""
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

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
