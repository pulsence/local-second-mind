"""
OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting) provider.

Provides a generic harvester for OAI-PMH compliant repositories, enabling
access to institutional repositories, digital libraries, and archives
worldwide.

Protocol specification: https://www.openarchives.org/pmh/
Version: OAI-PMH 2.0

Common repositories supporting OAI-PMH:
- arXiv (https://export.arxiv.org/oai2)
- PubMed Central (https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi)
- Zenodo (https://zenodo.org/oai2d)
- Figshare (https://api.figshare.com/v2/oai)
- SSRN (https://papers.ssrn.com/sol3/oai.cfm)
- RePEc (https://ideas.repec.org/cgi-bin/oai.pl)
- HAL (https://api.archives-ouvertes.fr/oai/hal)
- DOAJ (https://doaj.org/oai)
- Many institutional repositories (DSpace, EPrints, Fedora, etc.)
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Generator
from urllib.parse import urlencode

import requests

from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.logging import get_logger

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
        # Find the dc element (may be wrapped in oai_dc:dc)
        dc_ns = DublinCoreParser.DC_NAMESPACE
        oai_dc_ns = DublinCoreParser.OAI_DC_NAMESPACE

        # Try to find dc:dc or oai_dc:dc container
        dc_container = metadata_elem.find(f"{{{oai_dc_ns}}}dc")
        if dc_container is None:
            dc_container = metadata_elem

        def get_all_text(tag: str) -> List[str]:
            """Get all text values for a DC element."""
            results = []
            for elem in dc_container.findall(f"{{{dc_ns}}}{tag}"):
                if elem.text:
                    results.append(elem.text.strip())
            return results

        def get_first_text(tag: str) -> str:
            """Get first text value for a DC element."""
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
        marc_ns = MARCParser.MARC_NAMESPACE

        def get_datafield(tag: str, subfield_code: str = "a") -> List[str]:
            """Get values from MARC datafield/subfield."""
            results = []
            for df in metadata_elem.findall(f".//{{{marc_ns}}}datafield[@tag='{tag}']"):
                for sf in df.findall(f"{{{marc_ns}}}subfield[@code='{subfield_code}']"):
                    if sf.text:
                        results.append(sf.text.strip())
            return results

        def get_controlfield(tag: str) -> str:
            """Get value from MARC controlfield."""
            cf = metadata_elem.find(f".//{{{marc_ns}}}controlfield[@tag='{tag}']")
            return cf.text.strip() if cf is not None and cf.text else ""

        # Extract common fields
        # 245 = Title, 100/700 = Authors, 520 = Abstract, 650 = Subjects
        # 260 = Publisher, 856 = URL

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

        # Extract creators
        creators = []
        for creator in metadata_elem.findall(f".//{{{dc_ns}}}creator"):
            name = creator.find(f"{{{dc_ns}}}creatorName")
            if name is not None and name.text:
                creators.append(name.text.strip())

        # Extract identifiers
        identifiers = []
        for ident in metadata_elem.findall(f".//{{{dc_ns}}}identifier"):
            if ident.text:
                ident_type = ident.get("identifierType", "")
                identifiers.append(f"{ident_type}:{ident.text.strip()}" if ident_type else ident.text.strip())

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


# Registry of metadata format parsers
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


# Pre-configured well-known repositories
KNOWN_REPOSITORIES: Dict[str, OAIRepository] = {
    "arxiv": OAIRepository(
        name="arXiv",
        base_url="https://export.arxiv.org/oai2",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:arXiv.org:",
        url_template="https://arxiv.org/abs/{id}",
        description="Open access archive for physics, mathematics, computer science, and more",
        min_interval_seconds=3.0,  # arXiv requires 3s between requests
    ),
    "pubmed": OAIRepository(
        name="PubMed Central",
        base_url="https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:pubmedcentral.nih.gov:",
        url_template="https://www.ncbi.nlm.nih.gov/pmc/articles/{id}",
        description="Biomedical and life sciences journal literature",
        min_interval_seconds=1.0,
    ),
    "zenodo": OAIRepository(
        name="Zenodo",
        base_url="https://zenodo.org/oai2d",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:zenodo.org:",
        url_template="https://zenodo.org/record/{id}",
        description="General-purpose open repository for research outputs",
        min_interval_seconds=1.0,
    ),
    "doaj": OAIRepository(
        name="Directory of Open Access Journals",
        base_url="https://doaj.org/oai",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:doaj.org/article:",
        url_template="https://doaj.org/article/{id}",
        description="Open access journals across all disciplines",
        min_interval_seconds=1.0,
    ),
    "hal": OAIRepository(
        name="HAL (Hyper Articles en Ligne)",
        base_url="https://api.archives-ouvertes.fr/oai/hal",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:HAL:",
        url_template="https://hal.archives-ouvertes.fr/{id}",
        description="French national open archive for scholarly documents",
        min_interval_seconds=1.0,
    ),
    "repec": OAIRepository(
        name="RePEc (Research Papers in Economics)",
        base_url="https://ideas.repec.org/cgi-bin/oai.pl",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:RePEc:",
        url_template="https://ideas.repec.org/{id}",
        description="Economics and related sciences working papers and articles",
        min_interval_seconds=2.0,
    ),
    "europeana": OAIRepository(
        name="Europeana",
        base_url="https://oai.europeana.eu/oai",
        metadata_prefix="oai_dc",
        identifier_prefix="",
        url_template=None,
        description="European cultural heritage digital collections",
        min_interval_seconds=1.0,
    ),
    "philpapers": OAIRepository(
        name="PhilPapers",
        base_url="https://philpapers.org/oai.pl",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:philpapers.org:",
        url_template="https://philpapers.org/rec/{id}",
        description="Philosophy research bibliography",
        min_interval_seconds=2.0,
    ),
    "dspace_demo": OAIRepository(
        name="DSpace Demo",
        base_url="https://demo.dspace.org/server/oai/request",
        metadata_prefix="oai_dc",
        identifier_prefix="",
        url_template=None,
        description="DSpace demonstration repository",
        min_interval_seconds=1.0,
    ),
}


# ============================================================================
# OAI-PMH Client
# ============================================================================

class OAIPMHClient:
    """
    Low-level OAI-PMH protocol client.

    Implements all six OAI-PMH verbs:
    - Identify: Get repository information
    - ListMetadataFormats: List available metadata formats
    - ListSets: List available sets/collections
    - ListIdentifiers: List record headers (for selective harvesting)
    - ListRecords: List full records with metadata
    - GetRecord: Get a single record by identifier
    """

    OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        user_agent: str = "LocalSecondMind/1.0 (OAI-PMH Harvester)",
        min_interval_seconds: float = 1.0,
    ):
        """
        Initialize OAI-PMH client.

        Args:
            base_url: OAI-PMH endpoint URL
            timeout: Request timeout in seconds
            user_agent: User-Agent header for requests
            min_interval_seconds: Minimum seconds between requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.min_interval_seconds = min_interval_seconds
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Enforce rate limiting between requests."""
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _request(self, params: Dict[str, str]) -> ET.Element:
        """
        Make OAI-PMH request and return parsed XML root.

        Args:
            params: Request parameters including 'verb'

        Returns:
            Parsed XML root element

        Raises:
            OAIPMHError: If OAI-PMH error response received
            requests.RequestException: If HTTP error occurs
        """
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

        # Check for OAI-PMH errors
        error = root.find(f".//{{{self.OAI_NAMESPACE}}}error")
        if error is not None:
            error_code = error.get("code", "unknown")
            error_msg = error.text or "Unknown OAI-PMH error"
            raise OAIPMHError(error_code, error_msg)

        return root

    def identify(self) -> Dict[str, Any]:
        """
        Get repository identification information.

        Returns:
            Dict with repository name, base URL, protocol version, etc.
        """
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

    def list_metadata_formats(
        self, identifier: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        List available metadata formats.

        Args:
            identifier: Optional record identifier to get formats for

        Returns:
            List of dicts with metadataPrefix, schema, metadataNamespace
        """
        params = {"verb": "ListMetadataFormats"}
        if identifier:
            params["identifier"] = identifier

        root = self._request(params)
        formats = []

        for mf in root.findall(f".//{{{self.OAI_NAMESPACE}}}metadataFormat"):
            prefix = mf.find(f"{{{self.OAI_NAMESPACE}}}metadataPrefix")
            schema = mf.find(f"{{{self.OAI_NAMESPACE}}}schema")
            namespace = mf.find(f"{{{self.OAI_NAMESPACE}}}metadataNamespace")

            formats.append({
                "metadataPrefix": prefix.text if prefix is not None else "",
                "schema": schema.text if schema is not None else "",
                "metadataNamespace": namespace.text if namespace is not None else "",
            })

        return formats

    def list_sets(self) -> Generator[Dict[str, str], None, None]:
        """
        List available sets/collections.

        Yields:
            Dicts with setSpec, setName, and optional setDescription
        """
        resumption_token = None

        while True:
            if resumption_token:
                params = {"verb": "ListSets", "resumptionToken": resumption_token}
            else:
                params = {"verb": "ListSets"}

            try:
                root = self._request(params)
            except OAIPMHError as e:
                if e.code == "noSetHierarchy":
                    return  # Repository doesn't support sets
                raise

            for set_elem in root.findall(f".//{{{self.OAI_NAMESPACE}}}set"):
                spec = set_elem.find(f"{{{self.OAI_NAMESPACE}}}setSpec")
                name = set_elem.find(f"{{{self.OAI_NAMESPACE}}}setName")
                desc = set_elem.find(f".//{{{self.OAI_NAMESPACE}}}setDescription")

                yield {
                    "setSpec": spec.text if spec is not None else "",
                    "setName": name.text if name is not None else "",
                    "setDescription": desc.text if desc is not None and desc.text else "",
                }

            # Check for resumption token
            token_elem = root.find(f".//{{{self.OAI_NAMESPACE}}}resumptionToken")
            if token_elem is not None and token_elem.text:
                resumption_token = token_elem.text
            else:
                break

    def list_identifiers(
        self,
        metadata_prefix: str = "oai_dc",
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        set_spec: Optional[str] = None,
        resumption_token: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        List record identifiers (headers only).

        Args:
            metadata_prefix: Metadata format prefix
            from_date: Optional start date (YYYY-MM-DD or YYYY-MM-DDThh:mm:ssZ)
            until_date: Optional end date
            set_spec: Optional set to harvest from
            resumption_token: Token for pagination

        Returns:
            Tuple of (list of header dicts, next resumption token or None)
        """
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

        # Get resumption token
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
        """
        List full records with metadata.

        Args:
            metadata_prefix: Metadata format prefix
            from_date: Optional start date
            until_date: Optional end date
            set_spec: Optional set to harvest from
            resumption_token: Token for pagination

        Returns:
            Tuple of (list of (header dict, metadata element) tuples, next token)
        """
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

        # Get resumption token
        token_elem = root.find(f".//{{{self.OAI_NAMESPACE}}}resumptionToken")
        next_token = token_elem.text if token_elem is not None and token_elem.text else None

        return records, next_token

    def get_record(
        self, identifier: str, metadata_prefix: str = "oai_dc"
    ) -> Optional[tuple[Dict[str, Any], ET.Element]]:
        """
        Get a single record by identifier.

        Args:
            identifier: OAI identifier
            metadata_prefix: Metadata format prefix

        Returns:
            Tuple of (header dict, metadata element) or None if not found
        """
        params = {
            "verb": "GetRecord",
            "identifier": identifier,
            "metadataPrefix": metadata_prefix,
        }

        try:
            root = self._request(params)
        except OAIPMHError as e:
            if e.code in ("idDoesNotExist", "cannotDisseminateFormat"):
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
# OAI-PMH Remote Provider
# ============================================================================

class OAIPMHProvider(BaseRemoteProvider):
    """
    Remote provider for OAI-PMH compliant repositories.

    This provider enables harvesting metadata from institutional repositories,
    digital libraries, and archives that implement the OAI-PMH protocol.

    Supports:
    - Pre-configured well-known repositories (arXiv, Zenodo, PubMed, etc.)
    - Custom repository endpoints
    - Multiple metadata formats (Dublin Core, MARC, DataCite)
    - Resumption tokens for large result sets
    - Set-based filtering
    """

    DEFAULT_TIMEOUT = 30
    DEFAULT_MIN_INTERVAL_SECONDS = 1.0
    DEFAULT_SNIPPET_MAX_CHARS = 700
    DEFAULT_METADATA_PREFIX = "oai_dc"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OAI-PMH provider.

        Args:
            config: Configuration dict with keys:
                - repository: Name of known repository OR custom base URL
                - repositories: List of repository names/URLs to search (multi-repo mode)
                - metadata_prefix: Preferred metadata format (default: oai_dc)
                - set_spec: Optional set to filter by
                - timeout: Request timeout in seconds (default: 30)
                - min_interval_seconds: Rate limiting (default: 1.0)
                - snippet_max_chars: Max snippet length (default: 700)
                - user_agent: Custom User-Agent header
                - custom_repositories: Dict of custom repository configs
        """
        super().__init__(config)

        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        self.metadata_prefix = config.get("metadata_prefix") or self.DEFAULT_METADATA_PREFIX
        self.set_spec = config.get("set_spec")
        self.user_agent = (
            config.get("user_agent")
            or os.getenv("LSM_OAI_PMH_USER_AGENT")
            or "LocalSecondMind/1.0 (OAI-PMH Harvester)"
        )

        # Build repository list
        self.repositories: List[OAIRepository] = []
        self._load_repositories(config)

        # Create clients for each repository
        self._clients: Dict[str, OAIPMHClient] = {}
        for repo in self.repositories:
            self._clients[repo.name] = OAIPMHClient(
                base_url=repo.base_url,
                timeout=self.timeout,
                user_agent=self.user_agent,
                min_interval_seconds=repo.min_interval_seconds,
            )

    def _load_repositories(self, config: Dict[str, Any]) -> None:
        """Load repository configurations."""
        # Load custom repositories first
        custom_repos = config.get("custom_repositories") or {}
        for name, repo_config in custom_repos.items():
            self.repositories.append(OAIRepository(
                name=repo_config.get("name", name),
                base_url=repo_config["base_url"],
                metadata_prefix=repo_config.get("metadata_prefix", self.metadata_prefix),
                set_spec=repo_config.get("set_spec", self.set_spec),
                identifier_prefix=repo_config.get("identifier_prefix", ""),
                url_template=repo_config.get("url_template"),
                description=repo_config.get("description", ""),
                min_interval_seconds=repo_config.get(
                    "min_interval_seconds", self.min_interval_seconds
                ),
            ))

        # Single repository mode
        if "repository" in config:
            repo_id = config["repository"]
            if repo_id in KNOWN_REPOSITORIES:
                self.repositories.append(KNOWN_REPOSITORIES[repo_id])
            elif repo_id.startswith("http"):
                # Treat as custom URL
                self.repositories.append(OAIRepository(
                    name="Custom Repository",
                    base_url=repo_id,
                    metadata_prefix=self.metadata_prefix,
                    set_spec=self.set_spec,
                ))

        # Multi-repository mode
        if "repositories" in config:
            for repo_id in config["repositories"]:
                if repo_id in KNOWN_REPOSITORIES:
                    if not any(r.name == KNOWN_REPOSITORIES[repo_id].name for r in self.repositories):
                        self.repositories.append(KNOWN_REPOSITORIES[repo_id])
                elif repo_id.startswith("http"):
                    if not any(r.base_url == repo_id for r in self.repositories):
                        self.repositories.append(OAIRepository(
                            name=f"Repository ({repo_id[:30]}...)",
                            base_url=repo_id,
                            metadata_prefix=self.metadata_prefix,
                        ))

        # Default to Zenodo if no repositories specified
        if not self.repositories:
            self.repositories.append(KNOWN_REPOSITORIES["zenodo"])

    @property
    def name(self) -> str:
        return "oai_pmh"

    def is_available(self) -> bool:
        """OAI-PMH repositories are generally freely available."""
        return len(self.repositories) > 0

    def validate_config(self) -> None:
        """Validate OAI-PMH configuration."""
        if not self.repositories:
            raise ValueError("At least one OAI-PMH repository must be configured.")

        for repo in self.repositories:
            if not repo.base_url:
                raise ValueError(f"Repository '{repo.name}' is missing base_url.")

    def get_name(self) -> str:
        """Get provider name."""
        if len(self.repositories) == 1:
            return f"OAI-PMH ({self.repositories[0].name})"
        return f"OAI-PMH ({len(self.repositories)} repositories)"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Text query used for client-side filtering.", "required": True},
            {"name": "repository", "type": "string", "description": "Repository name or base URL hint.", "required": False},
            {"name": "set_spec", "type": "string", "description": "OAI-PMH set filter.", "required": False},
            {"name": "metadata_prefix", "type": "string", "description": "Requested metadata format.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Additional keywords.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields()

    def get_description(self) -> str:
        return "OAI-PMH harvester for repository metadata with client-side relevance filtering."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return super().search_structured(input_dict, max_results=max_results)

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search OAI-PMH repositories.

        Note: OAI-PMH doesn't support full-text search directly. This method
        fetches recent records and filters them client-side. For better search,
        use repository-specific APIs when available.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"OAI-PMH: query='{query}', max_results={max_results}")

        all_results: List[RemoteResult] = []

        # Search each repository
        results_per_repo = max(3, max_results // len(self.repositories))

        for repo in self.repositories:
            try:
                repo_results = self._search_repository(repo, query, results_per_repo)
                all_results.extend(repo_results)
            except Exception as exc:
                logger.warning(f"OAI-PMH error for {repo.name}: {exc}")
                continue

        # Sort by score and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        results = all_results[:max_results]

        # Re-score for final ranking
        for i, result in enumerate(results):
            result.score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))

        logger.info(f"OAI-PMH returned {len(results)} results")
        return results

    def _search_repository(
        self, repo: OAIRepository, query: str, max_results: int
    ) -> List[RemoteResult]:
        """Search a single repository."""
        client = self._clients[repo.name]

        # Fetch records (OAI-PMH doesn't support search, so we list and filter)
        records_to_fetch = max_results * 5  # Fetch more to filter

        try:
            raw_records, _ = client.list_records(
                metadata_prefix=repo.metadata_prefix or self.metadata_prefix,
                set_spec=repo.set_spec or self.set_spec,
            )
        except OAIPMHError as exc:
            logger.warning(f"OAI-PMH error for {repo.name}: {exc}")
            return []

        # Parse and filter records
        results: List[RemoteResult] = []
        query_terms = query.lower().split()

        for header, metadata_elem in raw_records[:records_to_fetch]:
            if header.get("deleted"):
                continue

            if metadata_elem is None:
                continue

            # Parse metadata
            record = self._parse_record(header, metadata_elem, repo)
            if record is None:
                continue

            # Filter by query terms
            searchable_text = f"{record.title} {record.description} {' '.join(record.subjects)}".lower()
            if not any(term in searchable_text for term in query_terms):
                continue

            # Convert to RemoteResult
            result = self._record_to_result(record, repo, len(results), max_results)
            if result:
                results.append(result)

            if len(results) >= max_results:
                break

        return results

    def _parse_record(
        self, header: Dict[str, Any], metadata_elem: ET.Element, repo: OAIRepository
    ) -> Optional[OAIRecord]:
        """Parse OAI-PMH record using appropriate metadata parser."""
        try:
            # Determine parser based on metadata prefix
            prefix = repo.metadata_prefix or self.metadata_prefix
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
            logger.debug(f"Error parsing OAI record: {exc}")
            return None

    def _record_to_result(
        self, record: OAIRecord, repo: OAIRepository, index: int, max_results: int
    ) -> Optional[RemoteResult]:
        """Convert OAIRecord to RemoteResult."""
        # Determine URL
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
            # Extract ID from OAI identifier
            record_id = record.identifier
            if repo.identifier_prefix and record_id.startswith(repo.identifier_prefix):
                record_id = record_id[len(repo.identifier_prefix):]
            url = repo.url_template.replace("{id}", record_id)

        if not url:
            url = record.identifier

        # Build snippet
        snippet = record.description
        if not snippet and record.creators:
            snippet = f"By {', '.join(record.creators[:3])}"
            if len(record.creators) > 3:
                snippet += " et al."
        if not snippet:
            snippet = f"Record from {repo.name}"

        snippet = self._truncate(snippet)

        # Calculate score
        score = max(0.2, 1.0 - (index * 0.8 / max(1, max_results - 1)))

        # Build metadata
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

        # Add DOI if found
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

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return int(match.group())
        return None

    def _truncate(self, text: str) -> str:
        """Truncate text to snippet length."""
        text = " ".join(text.split())  # Normalize whitespace
        if len(text) <= self.snippet_max_chars:
            return text
        return text[:self.snippet_max_chars].rstrip() + "..."

    def _format_citation(
        self,
        title: str,
        url: str,
        authors: List[str],
        year: Optional[int],
    ) -> str:
        """Format an academic citation string."""
        year_str = str(year) if year else "n.d."
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        author_str = self._format_authors(authors)
        return f"{author_str} ({year_str}). {title}. {url} (accessed {date_str})."

    def _format_authors(self, authors: List[str]) -> str:
        """Format author list for citation."""
        if not authors:
            return "Unknown"
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + ", et al."

    # ========================================================================
    # Additional Methods for Advanced Usage
    # ========================================================================

    def list_available_repositories(self) -> List[Dict[str, str]]:
        """
        List all known OAI-PMH repositories.

        Returns:
            List of dicts with name, base_url, description
        """
        repos = []
        for key, repo in KNOWN_REPOSITORIES.items():
            repos.append({
                "key": key,
                "name": repo.name,
                "base_url": repo.base_url,
                "description": repo.description,
            })
        return repos

    def identify_repository(self, repository: str) -> Dict[str, Any]:
        """
        Get identification information for a repository.

        Args:
            repository: Repository name key or base URL

        Returns:
            Dict with repository identification info
        """
        if repository in KNOWN_REPOSITORIES:
            base_url = KNOWN_REPOSITORIES[repository].base_url
        else:
            base_url = repository

        client = OAIPMHClient(
            base_url=base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
        )
        return client.identify()

    def list_repository_sets(self, repository: str) -> List[Dict[str, str]]:
        """
        List available sets/collections in a repository.

        Args:
            repository: Repository name key or base URL

        Returns:
            List of set dicts
        """
        if repository in KNOWN_REPOSITORIES:
            base_url = KNOWN_REPOSITORIES[repository].base_url
        else:
            base_url = repository

        client = OAIPMHClient(
            base_url=base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
        )
        return list(client.list_sets())

    def list_metadata_formats(self, repository: str) -> List[Dict[str, str]]:
        """
        List available metadata formats for a repository.

        Args:
            repository: Repository name key or base URL

        Returns:
            List of metadata format dicts
        """
        if repository in KNOWN_REPOSITORIES:
            base_url = KNOWN_REPOSITORIES[repository].base_url
        else:
            base_url = repository

        client = OAIPMHClient(
            base_url=base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
        )
        return client.list_metadata_formats()

    def get_record(
        self, identifier: str, repository: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific record by OAI identifier.

        Args:
            identifier: Full OAI identifier
            repository: Repository to query (if not in identifier)

        Returns:
            Parsed record dict or None
        """
        # Determine repository from identifier or parameter
        if repository:
            if repository in KNOWN_REPOSITORIES:
                repo = KNOWN_REPOSITORIES[repository]
            else:
                repo = OAIRepository(name="Custom", base_url=repository)
        elif self.repositories:
            repo = self.repositories[0]
        else:
            raise ValueError("No repository specified")

        client = self._clients.get(repo.name)
        if client is None:
            client = OAIPMHClient(
                base_url=repo.base_url,
                timeout=self.timeout,
                user_agent=self.user_agent,
            )

        result = client.get_record(
            identifier=identifier,
            metadata_prefix=repo.metadata_prefix or self.metadata_prefix,
        )

        if result is None:
            return None

        header, metadata_elem = result
        record = self._parse_record(header, metadata_elem, repo)

        if record is None:
            return None

        return {
            "identifier": record.identifier,
            "title": record.title,
            "creators": record.creators,
            "description": record.description,
            "subjects": record.subjects,
            "date": record.date,
            "year": record.year,
            "publisher": record.publisher,
            "source": record.source,
            "identifiers": record.identifiers,
            "types": record.types,
            "languages": record.languages,
            "datestamp": record.datestamp,
            "sets": record.set_specs,
        }

    def harvest_records(
        self,
        repository: str,
        max_records: int = 100,
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        set_spec: Optional[str] = None,
        callback: Optional[Callable[[OAIRecord], None]] = None,
    ) -> List[OAIRecord]:
        """
        Harvest records from a repository.

        Args:
            repository: Repository name key or base URL
            max_records: Maximum records to harvest
            from_date: Start date (YYYY-MM-DD)
            until_date: End date (YYYY-MM-DD)
            set_spec: Optional set to filter by
            callback: Optional callback for each record

        Returns:
            List of OAIRecord objects
        """
        if repository in KNOWN_REPOSITORIES:
            repo = KNOWN_REPOSITORIES[repository]
        else:
            repo = OAIRepository(name="Custom", base_url=repository)

        client = OAIPMHClient(
            base_url=repo.base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
            min_interval_seconds=repo.min_interval_seconds,
        )

        records: List[OAIRecord] = []
        resumption_token = None

        while len(records) < max_records:
            try:
                raw_records, next_token = client.list_records(
                    metadata_prefix=repo.metadata_prefix or self.metadata_prefix,
                    from_date=from_date,
                    until_date=until_date,
                    set_spec=set_spec or repo.set_spec,
                    resumption_token=resumption_token,
                )
            except OAIPMHError as exc:
                if exc.code == "noRecordsMatch":
                    break
                raise

            for header, metadata_elem in raw_records:
                if len(records) >= max_records:
                    break

                if header.get("deleted"):
                    continue

                if metadata_elem is None:
                    continue

                record = self._parse_record(header, metadata_elem, repo)
                if record:
                    records.append(record)
                    if callback:
                        callback(record)

            if next_token:
                resumption_token = next_token
            else:
                break

        return records
