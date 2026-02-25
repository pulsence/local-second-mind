"""
Shared OAI-PMH protocol helpers and base provider.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult

logger = get_logger(__name__)


# ============================================================================
# Metadata Format Parsers
# ============================================================================

@dataclass
class OAIRecord:
    """
    Parsed record from OAI-PMH response.

    Contains common fields extracted from various metadata formats.
    """
    identifier: str
    """OAI identifier (e.g., oai:arxiv.org:2301.00001)"""

    title: str = ""
    """Record title"""

    creators: List[str] = field(default_factory=list)
    """Authors/creators list"""

    description: str = ""
    """Abstract or description"""

    subjects: List[str] = field(default_factory=list)
    """Subject keywords/categories"""

    date: str = ""
    """Publication date string"""

    year: Optional[int] = None
    """Extracted year"""

    publisher: str = ""
    """Publisher name"""

    source: str = ""
    """Source publication (journal, etc.)"""

    identifiers: List[str] = field(default_factory=list)
    """URLs, DOIs, and other identifiers"""

    types: List[str] = field(default_factory=list)
    """Document types"""

    formats: List[str] = field(default_factory=list)
    """File formats available"""

    languages: List[str] = field(default_factory=list)
    """Languages"""

    rights: List[str] = field(default_factory=list)
    """Rights/license information"""

    relations: List[str] = field(default_factory=list)
    """Related resources"""

    coverage: List[str] = field(default_factory=list)
    """Coverage (temporal/spatial)"""

    contributors: List[str] = field(default_factory=list)
    """Contributors"""

    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    """Raw metadata dict for format-specific fields"""

    datestamp: str = ""
    """OAI datestamp (last modified)"""

    set_specs: List[str] = field(default_factory=list)
    """OAI set memberships"""

    deleted: bool = False
    """Whether record is marked as deleted"""


class MetadataParser:
    """Base class for metadata format parsers."""

    @staticmethod
    def parse(metadata_elem: ET.Element, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse metadata element into a dictionary.

        Args:
            metadata_elem: The <metadata> element content
            namespaces: XML namespaces dict

        Returns:
            Dict of parsed metadata fields
        """
        raise NotImplementedError


class DublinCoreParser(MetadataParser):
    """
    Parser for Dublin Core (oai_dc) metadata format.

    Dublin Core is the minimum required format for all OAI-PMH repositories.
    Elements: title, creator, subject, description, publisher, contributor,
    date, type, format, identifier, source, language, relation, coverage, rights
    """

    DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
    OAI_DC_NAMESPACE = "http://www.openarchives.org/OAI/2.0/oai_dc/"

    @staticmethod
    def parse(metadata_elem: ET.Element, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """Parse Dublin Core metadata."""
        _ = namespaces
        dc_ns = DublinCoreParser.DC_NAMESPACE
        oai_dc_ns = DublinCoreParser.OAI_DC_NAMESPACE

        dc_container = metadata_elem.find(f"{{{oai_dc_ns}}}dc")
        if dc_container is None:
            dc_container = metadata_elem

        def get_all_text(tag: str) -> List[str]:
            results = []
            for elem in dc_container.findall(f"{{{dc_ns}}}{tag}"):
                if elem.text:
                    results.append(elem.text.strip())
            return results

        def get_first_text(tag: str) -> str:
            values = get_all_text(tag)
            return values[0] if values else ""

        return {
            "title": get_first_text("title"),
            "creators": get_all_text("creator"),
            "subjects": get_all_text("subject"),
            "description": get_first_text("description"),
            "publisher": get_first_text("publisher"),
            "contributors": get_all_text("contributor"),
            "date": get_first_text("date"),
            "types": get_all_text("type"),
            "formats": get_all_text("format"),
            "identifiers": get_all_text("identifier"),
            "source": get_first_text("source"),
            "languages": get_all_text("language"),
            "relations": get_all_text("relation"),
            "coverage": get_all_text("coverage"),
            "rights": get_all_text("rights"),
        }


class MARCParser(MetadataParser):
    """
    Parser for MARC XML metadata format.

    MARC (Machine-Readable Cataloging) is commonly used by libraries.
    """

    MARC_NAMESPACE = "http://www.loc.gov/MARC21/slim"

    @staticmethod
    def parse(metadata_elem: ET.Element, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """Parse MARC XML metadata."""
        _ = namespaces
        marc_ns = MARCParser.MARC_NAMESPACE

        def get_datafield(tag: str, subfield_code: str = "a") -> List[str]:
            results = []
            for df in metadata_elem.findall(f".//{{{marc_ns}}}datafield[@tag='{tag}']"):
                for sf in df.findall(f"{{{marc_ns}}}subfield[@code='{subfield_code}']"):
                    if sf.text:
                        results.append(sf.text.strip())
            return results

        def get_controlfield(tag: str) -> str:
            cf = metadata_elem.find(f".//{{{marc_ns}}}controlfield[@tag='{tag}']")
            return cf.text.strip() if cf is not None and cf.text else ""

        titles = get_datafield("245", "a")
        title = titles[0] if titles else ""
        subtitle = get_datafield("245", "b")
        if subtitle:
            title = f"{title}: {subtitle[0]}"

        creators = get_datafield("100", "a") + get_datafield("700", "a")

        return {
            "title": title,
            "creators": creators,
            "subjects": get_datafield("650", "a") + get_datafield("653", "a"),
            "description": " ".join(get_datafield("520", "a")),
            "publisher": " ".join(get_datafield("260", "b")),
            "contributors": get_datafield("700", "a"),
            "date": " ".join(get_datafield("260", "c")),
            "types": get_datafield("655", "a"),
            "formats": [],
            "identifiers": get_datafield("856", "u") + get_datafield("020", "a") + get_datafield("022", "a"),
            "source": " ".join(get_datafield("773", "t")),
            "languages": [get_controlfield("008")[35:38]] if len(get_controlfield("008")) >= 38 else [],
            "relations": get_datafield("787", "t"),
            "coverage": get_datafield("651", "a"),
            "rights": get_datafield("540", "a"),
        }


class DataciteParser(MetadataParser):
    """
    Parser for DataCite metadata format.

    DataCite is commonly used for research data repositories.
    """

    DATACITE_NAMESPACE = "http://datacite.org/schema/kernel-4"

    @staticmethod
    def parse(metadata_elem: ET.Element, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """Parse DataCite metadata."""
        _ = namespaces
        dc_ns = DataciteParser.DATACITE_NAMESPACE

        def find_text(path: str) -> str:
            elem = metadata_elem.find(f".//{{{dc_ns}}}{path}")
            return elem.text.strip() if elem is not None and elem.text else ""

        def find_all_text(path: str) -> List[str]:
            results = []
            for elem in metadata_elem.findall(f".//{{{dc_ns}}}{path}"):
                if elem.text:
                    results.append(elem.text.strip())
            return results

        creators = []
        for creator in metadata_elem.findall(f".//{{{dc_ns}}}creator"):
            name = creator.find(f"{{{dc_ns}}}creatorName")
            if name is not None and name.text:
                creators.append(name.text.strip())

        identifiers = []
        for ident in metadata_elem.findall(f".//{{{dc_ns}}}identifier"):
            if ident.text:
                ident_type = ident.get("identifierType", "")
                identifiers.append(
                    f"{ident_type}:{ident.text.strip()}" if ident_type else ident.text.strip()
                )

        return {
            "title": find_text("title"),
            "creators": creators,
            "subjects": find_all_text("subject"),
            "description": find_text("description"),
            "publisher": find_text("publisher"),
            "contributors": find_all_text("contributor/contributorName"),
            "date": find_text("publicationYear"),
            "types": [find_text("resourceType")],
            "formats": find_all_text("format"),
            "identifiers": identifiers,
            "source": "",
            "languages": [find_text("language")],
            "relations": find_all_text("relatedIdentifier"),
            "coverage": find_all_text("geoLocation/geoLocationPlace"),
            "rights": find_all_text("rights"),
        }


METADATA_PARSERS: Dict[str, type] = {
    "oai_dc": DublinCoreParser,
    "dc": DublinCoreParser,
    "marc21": MARCParser,
    "marcxml": MARCParser,
    "datacite": DataciteParser,
}


# ============================================================================
# Repository Configuration
# ============================================================================

@dataclass
class OAIRepository:
    """
    Configuration for an OAI-PMH repository.
    """
    name: str
    """Human-readable repository name"""

    base_url: str
    """OAI-PMH base URL endpoint"""

    metadata_prefix: str = "oai_dc"
    """Preferred metadata format (oai_dc, marc21, datacite, etc.)"""

    set_spec: Optional[str] = None
    """Optional set to restrict harvesting"""

    identifier_prefix: str = ""
    """Prefix to strip from OAI identifiers for display"""

    url_template: Optional[str] = None
    """Template for constructing record URLs. Use {id} placeholder."""

    description: str = ""
    """Repository description"""

    min_interval_seconds: float = 1.0
    """Minimum seconds between requests (rate limiting)"""


# ============================================================================
# OAI-PMH Client
# ============================================================================

class OAIPMHClient:
    """
    Low-level OAI-PMH protocol client.

    Implements all six OAI-PMH verbs.
    """

    OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        user_agent: str = "LocalSecondMind/1.0 (OAI-PMH Harvester)",
        min_interval_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.min_interval_seconds = min_interval_seconds
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _request(self, params: Dict[str, str]) -> ET.Element:
        self._throttle()

        headers = {
            "Accept": "application/xml, text/xml",
            "User-Agent": self.user_agent,
        }

        response = requests.get(
            self.base_url,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)

        error = root.find(f".//{{{self.OAI_NAMESPACE}}}error")
        if error is not None:
            error_code = error.get("code", "unknown")
            error_msg = error.text or "Unknown OAI-PMH error"
            raise OAIPMHError(error_code, error_msg)

        return root

    def identify(self) -> Dict[str, Any]:
        root = self._request({"verb": "Identify"})
        identify = root.find(f".//{{{self.OAI_NAMESPACE}}}Identify")

        if identify is None:
            return {}

        def get_text(tag: str) -> str:
            elem = identify.find(f"{{{self.OAI_NAMESPACE}}}{tag}")
            return elem.text if elem is not None and elem.text else ""

        def get_all_text(tag: str) -> List[str]:
            return [
                elem.text for elem in identify.findall(f"{{{self.OAI_NAMESPACE}}}{tag}")
                if elem.text
            ]

        return {
            "repository_name": get_text("repositoryName"),
            "base_url": get_text("baseURL"),
            "protocol_version": get_text("protocolVersion"),
            "admin_emails": get_all_text("adminEmail"),
            "earliest_datestamp": get_text("earliestDatestamp"),
            "deleted_record": get_text("deletedRecord"),
            "granularity": get_text("granularity"),
        }

    def list_metadata_formats(self) -> List[Dict[str, str]]:
        root = self._request({"verb": "ListMetadataFormats"})
        formats = []

        for fmt in root.findall(f".//{{{self.OAI_NAMESPACE}}}metadataFormat"):
            metadata_prefix = fmt.find(f"{{{self.OAI_NAMESPACE}}}metadataPrefix")
            schema = fmt.find(f"{{{self.OAI_NAMESPACE}}}schema")
            metadata_namespace = fmt.find(f"{{{self.OAI_NAMESPACE}}}metadataNamespace")
            formats.append({
                "metadataPrefix": metadata_prefix.text if metadata_prefix is not None else "",
                "schema": schema.text if schema is not None else "",
                "metadataNamespace": metadata_namespace.text if metadata_namespace is not None else "",
            })

        return formats

    def list_sets(self) -> List[Dict[str, str]]:
        root = self._request({"verb": "ListSets"})
        sets = []

        for set_elem in root.findall(f".//{{{self.OAI_NAMESPACE}}}set"):
            spec = set_elem.find(f"{{{self.OAI_NAMESPACE}}}setSpec")
            name = set_elem.find(f"{{{self.OAI_NAMESPACE}}}setName")
            description = set_elem.find(f"{{{self.OAI_NAMESPACE}}}setDescription")
            sets.append({
                "setSpec": spec.text if spec is not None else "",
                "setName": name.text if name is not None else "",
                "setDescription": description.text if description is not None else "",
            })

        return sets

    def list_identifiers(
        self,
        metadata_prefix: str = "oai_dc",
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        set_spec: Optional[str] = None,
        resumption_token: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        if resumption_token:
            params = {"verb": "ListIdentifiers", "resumptionToken": resumption_token}
        else:
            params = {"verb": "ListIdentifiers", "metadataPrefix": metadata_prefix}
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            if set_spec:
                params["set"] = set_spec

        root = self._request(params)
        headers = []

        for header in root.findall(f".//{{{self.OAI_NAMESPACE}}}header"):
            identifier = header.find(f"{{{self.OAI_NAMESPACE}}}identifier")
            datestamp = header.find(f"{{{self.OAI_NAMESPACE}}}datestamp")
            set_specs = [
                s.text for s in header.findall(f"{{{self.OAI_NAMESPACE}}}setSpec")
                if s.text
            ]

            headers.append({
                "identifier": identifier.text if identifier is not None else "",
                "datestamp": datestamp.text if datestamp is not None else "",
                "setSpecs": set_specs,
                "deleted": header.get("status") == "deleted",
            })

        token_elem = root.find(f".//{{{self.OAI_NAMESPACE}}}resumptionToken")
        next_token = token_elem.text if token_elem is not None and token_elem.text else None

        return headers, next_token

    def list_records(
        self,
        metadata_prefix: str = "oai_dc",
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        set_spec: Optional[str] = None,
        resumption_token: Optional[str] = None,
    ) -> tuple[List[tuple[Dict[str, Any], ET.Element]], Optional[str]]:
        if resumption_token:
            params = {"verb": "ListRecords", "resumptionToken": resumption_token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            if set_spec:
                params["set"] = set_spec

        root = self._request(params)
        records = []

        for record in root.findall(f".//{{{self.OAI_NAMESPACE}}}record"):
            header = record.find(f"{{{self.OAI_NAMESPACE}}}header")
            metadata = record.find(f"{{{self.OAI_NAMESPACE}}}metadata")

            if header is None:
                continue

            identifier = header.find(f"{{{self.OAI_NAMESPACE}}}identifier")
            datestamp = header.find(f"{{{self.OAI_NAMESPACE}}}datestamp")
            set_specs = [
                s.text for s in header.findall(f"{{{self.OAI_NAMESPACE}}}setSpec")
                if s.text
            ]

            header_dict = {
                "identifier": identifier.text if identifier is not None else "",
                "datestamp": datestamp.text if datestamp is not None else "",
                "setSpecs": set_specs,
                "deleted": header.get("status") == "deleted",
            }

            records.append((header_dict, metadata))

        token_elem = root.find(f".//{{{self.OAI_NAMESPACE}}}resumptionToken")
        next_token = token_elem.text if token_elem is not None and token_elem.text else None

        return records, next_token

    def get_record(
        self, identifier: str, metadata_prefix: str = "oai_dc"
    ) -> Optional[tuple[Dict[str, Any], ET.Element]]:
        params = {
            "verb": "GetRecord",
            "identifier": identifier,
            "metadataPrefix": metadata_prefix,
        }

        try:
            root = self._request(params)
        except OAIPMHError as exc:
            if exc.code in ("idDoesNotExist", "cannotDisseminateFormat"):
                return None
            raise

        record = root.find(f".//{{{self.OAI_NAMESPACE}}}record")
        if record is None:
            return None

        header = record.find(f"{{{self.OAI_NAMESPACE}}}header")
        metadata = record.find(f"{{{self.OAI_NAMESPACE}}}metadata")

        if header is None:
            return None

        identifier_elem = header.find(f"{{{self.OAI_NAMESPACE}}}identifier")
        datestamp = header.find(f"{{{self.OAI_NAMESPACE}}}datestamp")
        set_specs = [
            s.text for s in header.findall(f"{{{self.OAI_NAMESPACE}}}setSpec")
            if s.text
        ]

        header_dict = {
            "identifier": identifier_elem.text if identifier_elem is not None else "",
            "datestamp": datestamp.text if datestamp is not None else "",
            "setSpecs": set_specs,
            "deleted": header.get("status") == "deleted",
        }

        return header_dict, metadata


class OAIPMHError(Exception):
    """OAI-PMH protocol error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"OAI-PMH error [{code}]: {message}")


# ============================================================================
# Base OAI Provider
# ============================================================================

class BaseOAIProvider(BaseRemoteProvider):
    """Shared helpers for OAI-PMH based providers."""

    DEFAULT_SNIPPET_MAX_CHARS = 700

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not hasattr(self, "snippet_max_chars"):
            snippet_max_chars = config.get("snippet_max_chars")
            if snippet_max_chars is not None:
                self.snippet_max_chars = int(snippet_max_chars)
            else:
                self.snippet_max_chars = self.DEFAULT_SNIPPET_MAX_CHARS
        if not hasattr(self, "normalize_snippet_whitespace"):
            self.normalize_snippet_whitespace = True

    def _extract_year(self, date_str: str) -> Optional[int]:
        if not date_str:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return int(match.group())
        return None

    def _truncate(self, text: str) -> str:
        text = str(text or "")
        if self.normalize_snippet_whitespace:
            text = " ".join(text.split())
        if len(text) <= self.snippet_max_chars:
            return text
        return text[:self.snippet_max_chars].rstrip() + "..."

    def _format_authors(self, authors: List[str], *, fallback: str = "Unknown") -> str:
        if not authors:
            return fallback
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + ", et al."

    def _format_citation(
        self,
        title: str,
        url: str,
        authors: List[str],
        year: Optional[int],
    ) -> str:
        year_str = str(year) if year else "n.d."
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        author_str = self._format_authors(authors)
        return f"{author_str} ({year_str}). {title}. {url} (accessed {date_str})."

    def _parse_record(
        self,
        header: Dict[str, Any],
        metadata_elem: ET.Element,
        repo: OAIRepository,
        metadata_prefix: Optional[str] = None,
    ) -> Optional[OAIRecord]:
        try:
            prefix = repo.metadata_prefix or metadata_prefix or getattr(self, "metadata_prefix", "oai_dc")
            parser_class = METADATA_PARSERS.get(prefix, DublinCoreParser)

            parsed = parser_class.parse(metadata_elem, {})

            record = OAIRecord(
                identifier=header.get("identifier", ""),
                title=parsed.get("title", "Untitled"),
                creators=parsed.get("creators", []),
                description=parsed.get("description", ""),
                subjects=parsed.get("subjects", []),
                date=parsed.get("date", ""),
                year=self._extract_year(parsed.get("date", "")),
                publisher=parsed.get("publisher", ""),
                source=parsed.get("source", ""),
                identifiers=parsed.get("identifiers", []),
                types=parsed.get("types", []),
                formats=parsed.get("formats", []),
                languages=parsed.get("languages", []),
                rights=parsed.get("rights", []),
                relations=parsed.get("relations", []),
                coverage=parsed.get("coverage", []),
                contributors=parsed.get("contributors", []),
                raw_metadata=parsed,
                datestamp=header.get("datestamp", ""),
                set_specs=header.get("setSpecs", []),
                deleted=header.get("deleted", False),
            )

            return record
        except Exception as exc:
            logger.debug("Error parsing OAI record: %s", exc)
            return None

    def _record_to_result(
        self,
        record: OAIRecord,
        repo: OAIRepository,
        index: int,
        max_results: int,
    ) -> Optional[RemoteResult]:
        url = ""
        if record.identifiers:
            for ident in record.identifiers:
                if ident.startswith("http"):
                    url = ident
                    break
                if ident.startswith("doi:") or ident.startswith("10."):
                    doi = ident.replace("doi:", "").strip()
                    url = f"https://doi.org/{doi}"
                    break

        if not url and repo.url_template:
            record_id = record.identifier
            if repo.identifier_prefix and record_id.startswith(repo.identifier_prefix):
                record_id = record_id[len(repo.identifier_prefix):]
            url = repo.url_template.replace("{id}", record_id)

        if not url:
            url = record.identifier

        snippet = record.description
        if not snippet and record.creators:
            snippet = f"By {', '.join(record.creators[:3])}"
            if len(record.creators) > 3:
                snippet += " et al."
        if not snippet:
            snippet = f"Record from {repo.name}"

        snippet = self._truncate(snippet)

        score = max(0.2, 1.0 - (index * 0.8 / max(1, max_results - 1)))

        metadata = {
            "oai_identifier": record.identifier,
            "repository": repo.name,
            "authors": record.creators,
            "year": record.year,
            "subjects": record.subjects,
            "types": record.types,
            "publisher": record.publisher,
            "source": record.source,
            "languages": record.languages,
            "datestamp": record.datestamp,
            "sets": record.set_specs,
            "citation": self._format_citation(
                record.title, url, record.creators, record.year
            ),
        }

        for ident in record.identifiers:
            if ident.startswith("doi:") or ident.startswith("10."):
                metadata["doi"] = ident.replace("doi:", "").strip()
                break

        return RemoteResult(
            title=record.title,
            url=url,
            snippet=snippet,
            score=score,
            metadata=metadata,
        )
