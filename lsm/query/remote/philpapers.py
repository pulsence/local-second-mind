"""
PhilPapers provider for remote search.

Provides access to philosophy papers and research from PhilPapers,
the premier index and bibliography of philosophy.
API documentation: https://philpapers.org/help/api/
OAI-PMH access: https://philpapers.org/help/oai.html
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests

from .base import BaseRemoteProvider, RemoteResult
from lsm.cli.logging import get_logger

logger = get_logger(__name__)


class PhilPapersProvider(BaseRemoteProvider):
    """
    Remote provider for PhilPapers philosophy database.

    PhilPapers is a comprehensive index and bibliography of philosophy
    maintained by the community of philosophers. It indexes journals,
    books, open access archives, and personal academic pages.

    This provider uses PhilPapers' OAI-PMH interface for open access
    content and supports searching by subject categories.

    Subject categories include:
    - Epistemology
    - Ethics
    - Metaphysics
    - Philosophy of Mind
    - Philosophy of Language
    - Philosophy of Science
    - Logic and Philosophy of Logic
    - History of Philosophy
    - Political Philosophy
    - Aesthetics
    - Philosophy of Religion
    - And many more specialized areas
    """

    OAI_ENDPOINT = "https://philpapers.org/oai.pl"
    SEARCH_ENDPOINT = "https://philpapers.org/s/"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 2.0  # Be respectful to PhilPapers servers
    DEFAULT_SNIPPET_MAX_CHARS = 700

    # Dublin Core namespaces for OAI-PMH
    OAI_NAMESPACES = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    }

    # Philosophy subject categories (PhilPapers categorization)
    SUBJECT_CATEGORIES = {
        "epistemology": "Epistemology",
        "ethics": "Ethics",
        "metaphysics": "Metaphysics",
        "mind": "Philosophy of Mind",
        "language": "Philosophy of Language",
        "science": "Philosophy of Science",
        "logic": "Logic and Philosophy of Logic",
        "history": "History of Philosophy",
        "political": "Political Philosophy",
        "aesthetics": "Aesthetics",
        "religion": "Philosophy of Religion",
        "action": "Philosophy of Action",
        "cognitive": "Philosophy of Cognitive Science",
        "math": "Philosophy of Mathematics",
        "physics": "Philosophy of Physics",
        "biology": "Philosophy of Biology",
        "social": "Philosophy of Social Science",
        "law": "Philosophy of Law",
        "feminism": "Feminist Philosophy",
        "continental": "Continental Philosophy",
        "ancient": "Ancient Philosophy",
        "medieval": "Medieval Philosophy",
        "modern": "Modern Philosophy",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PhilPapers provider.

        Args:
            config: Configuration dict with optional keys:
                - api_id: PhilPapers API ID (for extended access)
                - api_key: PhilPapers API key (for extended access)
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - subject_categories: Filter by categories (e.g., ["ethics", "epistemology"])
                - include_books: Include books in results (default: True)
                - open_access_only: Only return open access papers (default: False)
        """
        super().__init__(config)

        self.api_id = config.get("api_id") or os.getenv("PHILPAPERS_API_ID")
        self.api_key = config.get("api_key") or os.getenv("PHILPAPERS_API_KEY")
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        self.subject_categories = config.get("subject_categories") or []
        self.include_books = config.get("include_books", True)
        self.open_access_only = config.get("open_access_only", False)
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "philpapers"

    def is_available(self) -> bool:
        """OAI-PMH endpoint is freely available for open access content."""
        return True

    def validate_config(self) -> None:
        """Validate PhilPapers configuration."""
        invalid_categories = [
            cat for cat in self.subject_categories
            if cat.lower() not in self.SUBJECT_CATEGORIES
        ]
        if invalid_categories:
            valid_cats = ", ".join(sorted(self.SUBJECT_CATEGORIES.keys()))
            raise ValueError(
                f"Invalid subject categories: {invalid_categories}. "
                f"Valid categories are: {valid_cats}"
            )

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search PhilPapers for philosophy papers.

        Args:
            query: Search query string. Supports:
                - Regular text search
                - author:Name - search by author
                - subject:category - search by subject category
                - title:phrase - search in titles
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"PhilPapers: query='{query}', max_results={max_results}")

        try:
            # Parse query for special fields
            parsed_query = self._parse_query(query)

            # Try OAI-PMH search for open access content
            results = self._search_oai(parsed_query, max_results)

            # If we didn't get enough results and have API credentials,
            # we could try the JSON API (for future extension)

            logger.info(f"PhilPapers returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as exc:
            logger.error(f"PhilPapers API error: {exc}")
            return []
        except Exception as exc:
            logger.error(f"PhilPapers parsing error: {exc}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "PhilPapers"

    def list_subject_categories(self) -> Dict[str, str]:
        """
        Get available subject categories.

        Returns:
            Dict mapping short names to full category names
        """
        return dict(self.SUBJECT_CATEGORIES)

    def search_by_category(
        self, category: str, max_results: int = 10
    ) -> List[RemoteResult]:
        """
        Search papers in a specific philosophy category.

        Args:
            category: Category short name (e.g., "ethics", "epistemology")
            max_results: Maximum results to return

        Returns:
            List of RemoteResult objects
        """
        category_lower = category.lower()
        if category_lower not in self.SUBJECT_CATEGORIES:
            logger.warning(f"Unknown category: {category}")
            return []

        return self.search(f"subject:{category_lower}", max_results)

    def get_paper_by_id(self, philpapers_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific paper by PhilPapers ID.

        Args:
            philpapers_id: PhilPapers identifier (e.g., "SMIETH-3")

        Returns:
            Paper details dict or None if not found
        """
        url = f"{self.OAI_ENDPOINT}"
        params = {
            "verb": "GetRecord",
            "identifier": f"oai:philpapers.org:{philpapers_id}",
            "metadataPrefix": "oai_dc",
        }

        try:
            data = self._request_oai(url, params)
            if data is None:
                return None

            record = data.find(".//oai:record", self.OAI_NAMESPACES)
            if record is None:
                return None

            return self._parse_oai_record(record)
        except Exception as exc:
            logger.error(f"Error fetching paper details: {exc}")
            return None

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query string for special field prefixes."""
        parsed = {
            "text": "",
            "author": None,
            "subject": None,
            "title": None,
        }

        # Extract field-specific queries
        remaining_parts = []
        parts = query.split()

        for part in parts:
            if part.lower().startswith("author:"):
                parsed["author"] = part[7:]
            elif part.lower().startswith("subject:"):
                parsed["subject"] = part[8:]
            elif part.lower().startswith("title:"):
                parsed["title"] = part[6:]
            else:
                remaining_parts.append(part)

        parsed["text"] = " ".join(remaining_parts)

        # Add configured subject categories
        if self.subject_categories and not parsed["subject"]:
            parsed["subject"] = self.subject_categories[0]

        return parsed

    def _search_oai(
        self, parsed_query: Dict[str, Any], max_results: int
    ) -> List[RemoteResult]:
        """Search using OAI-PMH ListRecords with filters."""
        # OAI-PMH doesn't support full-text search directly,
        # so we list records and filter client-side
        # For better performance, we use sets if subject is specified

        params = {
            "verb": "ListRecords",
            "metadataPrefix": "oai_dc",
        }

        # If subject category specified, use set parameter
        if parsed_query.get("subject"):
            subject = parsed_query["subject"].lower()
            if subject in self.SUBJECT_CATEGORIES:
                params["set"] = f"subject:{subject}"

        records = self._fetch_oai_records(params, max_results * 3)  # Fetch more to filter

        # Filter records by query terms
        results = []
        search_text = parsed_query.get("text", "").lower()
        author_filter = parsed_query.get("author", "").lower() if parsed_query.get("author") else None
        title_filter = parsed_query.get("title", "").lower() if parsed_query.get("title") else None

        for i, record in enumerate(records):
            if len(results) >= max_results:
                break

            # Apply filters
            if search_text:
                record_text = f"{record.get('title', '')} {record.get('description', '')}".lower()
                if not any(term in record_text for term in search_text.split()):
                    continue

            if author_filter:
                authors_lower = [a.lower() for a in record.get("authors", [])]
                if not any(author_filter in a for a in authors_lower):
                    continue

            if title_filter:
                if title_filter not in record.get("title", "").lower():
                    continue

            result = self._convert_record_to_result(record, i, max_results)
            if result:
                results.append(result)

        return results

    def _fetch_oai_records(
        self, params: Dict[str, str], max_records: int
    ) -> List[Dict[str, Any]]:
        """Fetch records from OAI-PMH endpoint."""
        records = []
        resumption_token = None

        while len(records) < max_records:
            if resumption_token:
                request_params = {
                    "verb": "ListRecords",
                    "resumptionToken": resumption_token,
                }
            else:
                request_params = params

            data = self._request_oai(self.OAI_ENDPOINT, request_params)
            if data is None:
                break

            # Parse records
            for record in data.findall(".//oai:record", self.OAI_NAMESPACES):
                parsed = self._parse_oai_record(record)
                if parsed:
                    records.append(parsed)
                    if len(records) >= max_records:
                        break

            # Check for resumption token
            token_elem = data.find(".//oai:resumptionToken", self.OAI_NAMESPACES)
            if token_elem is not None and token_elem.text:
                resumption_token = token_elem.text
            else:
                break

        return records

    def _parse_oai_record(self, record: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a single OAI-PMH record into a dict."""
        try:
            header = record.find("oai:header", self.OAI_NAMESPACES)
            metadata = record.find(".//oai_dc:dc", self.OAI_NAMESPACES)

            if header is None or metadata is None:
                return None

            # Check if deleted
            if header.get("status") == "deleted":
                return None

            identifier = header.findtext("oai:identifier", "", self.OAI_NAMESPACES)
            # Extract PhilPapers ID from OAI identifier
            philpapers_id = identifier.replace("oai:philpapers.org:", "") if identifier else ""

            # Extract Dublin Core fields
            title = metadata.findtext("dc:title", "Untitled", self.OAI_NAMESPACES)
            description = metadata.findtext("dc:description", "", self.OAI_NAMESPACES)

            authors = []
            for creator in metadata.findall("dc:creator", self.OAI_NAMESPACES):
                if creator.text:
                    authors.append(creator.text)

            subjects = []
            for subject in metadata.findall("dc:subject", self.OAI_NAMESPACES):
                if subject.text:
                    subjects.append(subject.text)

            # Get date
            date_str = metadata.findtext("dc:date", "", self.OAI_NAMESPACES)
            year = self._extract_year(date_str)

            # Get identifier (URL or DOI)
            url = None
            doi = None
            for ident in metadata.findall("dc:identifier", self.OAI_NAMESPACES):
                if ident.text:
                    if ident.text.startswith("http"):
                        url = ident.text
                    elif ident.text.startswith("10.") or "doi" in ident.text.lower():
                        doi = ident.text.replace("doi:", "").strip()

            if not url:
                url = f"https://philpapers.org/rec/{philpapers_id}"

            # Get type (article, book, etc.)
            doc_type = metadata.findtext("dc:type", "", self.OAI_NAMESPACES)

            # Get publisher
            publisher = metadata.findtext("dc:publisher", "", self.OAI_NAMESPACES)

            # Get source (journal/book title)
            source = metadata.findtext("dc:source", "", self.OAI_NAMESPACES)

            return {
                "philpapers_id": philpapers_id,
                "title": title,
                "description": description,
                "authors": authors,
                "subjects": subjects,
                "year": year,
                "date": date_str,
                "url": url,
                "doi": doi,
                "type": doc_type,
                "publisher": publisher,
                "source": source,
            }
        except Exception as exc:
            logger.debug(f"Error parsing OAI record: {exc}")
            return None

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        # Try to find 4-digit year
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return int(match.group())
        return None

    def _convert_record_to_result(
        self, record: Dict[str, Any], index: int, max_results: int
    ) -> Optional[RemoteResult]:
        """Convert parsed record to RemoteResult."""
        title = record.get("title") or "Untitled"
        url = record.get("url") or ""
        description = record.get("description") or ""
        authors = record.get("authors", [])

        # Build snippet
        if description:
            snippet = self._truncate(description)
        elif authors:
            snippet = f"Philosophy paper by {', '.join(authors[:3])}"
            if len(authors) > 3:
                snippet += " et al."
        else:
            snippet = "Philosophy paper from PhilPapers"

        # Calculate score based on position
        score = max(0.2, 1.0 - (index * 0.8 / max(1, max_results - 1)))

        # Build metadata
        metadata = {
            "philpapers_id": record.get("philpapers_id"),
            "authors": authors,
            "year": record.get("year"),
            "subjects": record.get("subjects", []),
            "doi": record.get("doi"),
            "type": record.get("type"),
            "publisher": record.get("publisher"),
            "source": record.get("source"),
            "citation": self._format_citation(
                title, url, authors, record.get("year"), record.get("doi")
            ),
        }

        return RemoteResult(
            title=title,
            url=url,
            snippet=snippet,
            score=score,
            metadata=metadata,
        )

    def _request_oai(
        self, url: str, params: Dict[str, str]
    ) -> Optional[ET.Element]:
        """Make an OAI-PMH request and parse XML response."""
        self._throttle()

        headers = {
            "Accept": "application/xml",
            "User-Agent": "LocalSecondMind/1.0 (Philosophy Research Tool)",
        }

        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)

        # Check for OAI-PMH errors
        error = root.find(".//oai:error", self.OAI_NAMESPACES)
        if error is not None:
            error_code = error.get("code", "unknown")
            error_msg = error.text or "Unknown error"
            logger.warning(f"OAI-PMH error: {error_code} - {error_msg}")
            return None

        return root

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
