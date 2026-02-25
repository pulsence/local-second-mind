"""
PhilPapers provider for remote search.

Provides access to philosophy papers and research from PhilPapers,
the premier index and bibliography of philosophy.
API documentation: https://philpapers.org/help/api/
OAI-PMH access: https://philpapers.org/help/oai.html

This provider uses the shared OAI-PMH infrastructure from oai_pmh.py
for common OAI-PMH functionality while adding PhilPapers-specific
features like subject category filtering.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests

from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.providers.academic.oai_pmh import (
    OAIPMHClient,
    OAIPMHError,
    DublinCoreParser,
)
from lsm.logging import get_logger

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
    SEARCH_ENDPOINT = "https://philpapers.org/s"
    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 2.0  # Be respectful to PhilPapers servers
    DEFAULT_SNIPPET_MAX_CHARS = 700

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

        # Initialize OAI-PMH client using shared infrastructure
        self._oai_client = OAIPMHClient(
            base_url=self.OAI_ENDPOINT,
            timeout=self.timeout,
            user_agent="LocalSecondMind/1.0 (Philosophy Research Tool)",
            min_interval_seconds=self.min_interval_seconds,
        )

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

            # Fallback to HTML search if OAI results are sparse.
            # Keep OAI results and append unique HTML results instead of replacing.
            if len(results) < max_results:
                html_results = self._search_html(query, max_results - len(results))
                if html_results:
                    existing_urls = {r.url for r in results}
                    for item in html_results:
                        if item.url not in existing_urls:
                            results.append(item)
                            existing_urls.add(item.url)

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

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Philosophy search query.", "required": True},
            {"name": "title", "type": "string", "description": "Title phrase.", "required": False},
            {"name": "author", "type": "string", "description": "Author name.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Topic keywords.", "required": False},
            {"name": "subject", "type": "string", "description": "PhilPapers subject/category.", "required": False},
            {"name": "year", "type": "integer", "description": "Publication year hint.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "philpapers_id", "type": "string", "description": "PhilPapers record ID."},
            {"name": "subjects", "type": "array[string]", "description": "PhilPapers subject categories."},
            {"name": "type", "type": "string", "description": "Publication type."},
            {"name": "publisher", "type": "string", "description": "Publisher name."},
            {"name": "source", "type": "string", "description": "Source origin label."},
            {"name": "citation", "type": "string", "description": "Formatted citation string."},
        ]

    def get_description(self) -> str:
        return "Philosophy-focused provider indexing PhilPapers publications and categories."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return super().search_structured(input_dict, max_results=max_results)

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
        try:
            # Use shared OAI-PMH client
            result = self._oai_client.get_record(
                identifier=f"oai:philpapers.org:{philpapers_id}",
                metadata_prefix="oai_dc",
            )

            if result is None:
                return None

            header, metadata_elem = result
            return self._parse_oai_metadata(header, metadata_elem)
        except OAIPMHError as exc:
            logger.warning(f"OAI-PMH error fetching paper: {exc}")
            return None
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

        fetch_limit = min(max_results * 20, 200)
        records = self._fetch_oai_records(params, fetch_limit)

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

    def _search_html(self, query: str, max_results: int) -> List[RemoteResult]:
        """Fallback HTML search for PhilPapers."""
        url = self.SEARCH_ENDPOINT
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LocalSecondMind/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://philpapers.org/",
        }
        params = {"search": query}

        try:
            session = requests.Session()
            session.get("https://philpapers.org/", headers=headers, timeout=self.timeout)
            response = session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.error(f"PhilPapers HTML search error: {exc}")
            return []

        html = response.text
        results: List[RemoteResult] = []

        pattern = re.compile(r'href="(/rec/[^"#?]+)"[^>]*>([^<]+)</a>', re.IGNORECASE)
        for i, match in enumerate(pattern.finditer(html)):
            if len(results) >= max_results:
                break
            path = match.group(1)
            title = match.group(2).strip() or "Untitled"
            url = f"https://philpapers.org{path}"
            score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))
            results.append(
                RemoteResult(
                    title=title,
                    url=url,
                    snippet="PhilPapers search result",
                    score=score,
                    metadata={"source": "PhilPapers"},
                )
            )

        logger.info(f"PhilPapers HTML returned {len(results)} results")
        return results

    def _fetch_oai_records(
        self, params: Dict[str, str], max_records: int
    ) -> List[Dict[str, Any]]:
        """Fetch records from OAI-PMH endpoint using shared OAI-PMH client."""
        records = []
        resumption_token = None

        # Extract set_spec from params if present
        set_spec = params.get("set")

        while len(records) < max_records:
            try:
                # Use shared OAI-PMH client for listing records
                raw_records, next_token = self._oai_client.list_records(
                    metadata_prefix="oai_dc",
                    set_spec=set_spec,
                    resumption_token=resumption_token,
                )

                # Parse records
                for header, metadata_elem in raw_records:
                    if header.get("deleted"):
                        continue
                    if metadata_elem is None:
                        continue

                    parsed = self._parse_oai_metadata(header, metadata_elem)
                    if parsed:
                        records.append(parsed)
                        if len(records) >= max_records:
                            break

                # Check for resumption token
                if next_token:
                    resumption_token = next_token
                else:
                    break
            except OAIPMHError as exc:
                if exc.code == "noRecordsMatch":
                    break
                logger.warning(f"OAI-PMH error: {exc}")
                break
            except Exception as exc:
                logger.error(f"Error fetching OAI records: {exc}")
                break

        return records

    def _parse_oai_metadata(
        self, header: Dict[str, Any], metadata_elem: ET.Element
    ) -> Optional[Dict[str, Any]]:
        """
        Parse OAI-PMH metadata using shared Dublin Core parser.

        Args:
            header: Header dict from OAI-PMH client
            metadata_elem: Metadata XML element

        Returns:
            Parsed record dict or None
        """
        try:
            # Use shared Dublin Core parser
            parsed = DublinCoreParser.parse(metadata_elem, {})

            identifier = header.get("identifier", "")
            # Extract PhilPapers ID from OAI identifier
            philpapers_id = identifier.replace("oai:philpapers.org:", "") if identifier else ""

            # Get URL and DOI from identifiers
            url = None
            doi = None
            for ident in parsed.get("identifiers", []):
                if ident.startswith("http"):
                    url = ident
                elif ident.startswith("10.") or "doi" in ident.lower():
                    doi = ident.replace("doi:", "").strip()

            if not url:
                url = f"https://philpapers.org/rec/{philpapers_id}"

            # Extract year from date
            date_str = parsed.get("date", "")
            year = self._extract_year(date_str)

            return {
                "philpapers_id": philpapers_id,
                "title": parsed.get("title") or "Untitled",
                "description": parsed.get("description") or "",
                "authors": parsed.get("creators", []),
                "subjects": parsed.get("subjects", []),
                "year": year,
                "date": date_str,
                "url": url,
                "doi": doi,
                "type": parsed.get("types", [""])[0] if parsed.get("types") else "",
                "publisher": parsed.get("publisher") or "",
                "source": parsed.get("source") or "",
            }
        except Exception as exc:
            logger.debug(f"Error parsing OAI metadata: {exc}")
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
