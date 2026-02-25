"""
Tests for OAI-PMH provider implementation.

Tests cover:
- OAI-PMH client functionality
- Metadata format parsers (Dublin Core, MARC, DataCite)
- OAI-PMH provider integration
- Repository configuration
- Error handling
"""

import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch, MagicMock

import pytest

from lsm.remote.providers.academic.oai_pmh import (
    OAIPMHProvider,
    OAIPMHClient,
    OAIPMHError,
    OAIRecord,
    OAIRepository,
    DublinCoreParser,
    MARCParser,
    DataciteParser,
    KNOWN_REPOSITORIES,
    METADATA_PARSERS,
)


# ============================================================================
# Sample XML Responses
# ============================================================================

IDENTIFY_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <request verb="Identify">http://example.org/oai</request>
  <Identify>
    <repositoryName>Test Repository</repositoryName>
    <baseURL>http://example.org/oai</baseURL>
    <protocolVersion>2.0</protocolVersion>
    <adminEmail>admin@example.org</adminEmail>
    <earliestDatestamp>2000-01-01</earliestDatestamp>
    <deletedRecord>persistent</deletedRecord>
    <granularity>YYYY-MM-DD</granularity>
  </Identify>
</OAI-PMH>
"""

LIST_METADATA_FORMATS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <request verb="ListMetadataFormats">http://example.org/oai</request>
  <ListMetadataFormats>
    <metadataFormat>
      <metadataPrefix>oai_dc</metadataPrefix>
      <schema>http://www.openarchives.org/OAI/2.0/oai_dc.xsd</schema>
      <metadataNamespace>http://www.openarchives.org/OAI/2.0/oai_dc/</metadataNamespace>
    </metadataFormat>
    <metadataFormat>
      <metadataPrefix>marc21</metadataPrefix>
      <schema>http://www.loc.gov/standards/marcxml/schema/MARC21slim.xsd</schema>
      <metadataNamespace>http://www.loc.gov/MARC21/slim</metadataNamespace>
    </metadataFormat>
  </ListMetadataFormats>
</OAI-PMH>
"""

LIST_SETS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <request verb="ListSets">http://example.org/oai</request>
  <ListSets>
    <set>
      <setSpec>physics</setSpec>
      <setName>Physics</setName>
    </set>
    <set>
      <setSpec>math</setSpec>
      <setName>Mathematics</setName>
    </set>
  </ListSets>
</OAI-PMH>
"""

LIST_RECORDS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <request verb="ListRecords" metadataPrefix="oai_dc">http://example.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:example.org:record1</identifier>
        <datestamp>2024-01-01</datestamp>
        <setSpec>physics</setSpec>
      </header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Test Paper on Physics</dc:title>
          <dc:creator>Jane Doe</dc:creator>
          <dc:creator>John Smith</dc:creator>
          <dc:subject>Physics</dc:subject>
          <dc:subject>Quantum Mechanics</dc:subject>
          <dc:description>This is a test abstract for the paper.</dc:description>
          <dc:date>2024-01-01</dc:date>
          <dc:identifier>http://example.org/record1</dc:identifier>
          <dc:identifier>doi:10.1234/test.2024.001</dc:identifier>
          <dc:type>Article</dc:type>
          <dc:publisher>Test Publisher</dc:publisher>
          <dc:language>en</dc:language>
        </oai_dc:dc>
      </metadata>
    </record>
    <record>
      <header>
        <identifier>oai:example.org:record2</identifier>
        <datestamp>2024-01-02</datestamp>
      </header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Second Test Paper</dc:title>
          <dc:creator>Alice Johnson</dc:creator>
          <dc:description>Another test abstract.</dc:description>
          <dc:date>2024-01-02</dc:date>
        </oai_dc:dc>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""

LIST_RECORDS_WITH_RESUMPTION = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:example.org:record1</identifier>
        <datestamp>2024-01-01</datestamp>
      </header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Test Paper</dc:title>
          <dc:description>Test abstract.</dc:description>
        </oai_dc:dc>
      </metadata>
    </record>
    <resumptionToken completeListSize="100" cursor="0">token123</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""

GET_RECORD_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <request verb="GetRecord" identifier="oai:example.org:record1" metadataPrefix="oai_dc">http://example.org/oai</request>
  <GetRecord>
    <record>
      <header>
        <identifier>oai:example.org:record1</identifier>
        <datestamp>2024-01-01</datestamp>
      </header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Specific Test Paper</dc:title>
          <dc:creator>Test Author</dc:creator>
          <dc:description>Test description for specific record.</dc:description>
          <dc:date>2024-01-01</dc:date>
        </oai_dc:dc>
      </metadata>
    </record>
  </GetRecord>
</OAI-PMH>
"""

ERROR_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <request verb="GetRecord">http://example.org/oai</request>
  <error code="idDoesNotExist">The identifier does not exist</error>
</OAI-PMH>
"""

NO_RECORDS_ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2024-01-01T00:00:00Z</responseDate>
  <error code="noRecordsMatch">No records match the request</error>
</OAI-PMH>
"""

DELETED_RECORD_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header status="deleted">
        <identifier>oai:example.org:deleted1</identifier>
        <datestamp>2024-01-01</datestamp>
      </header>
    </record>
    <record>
      <header>
        <identifier>oai:example.org:active1</identifier>
        <datestamp>2024-01-02</datestamp>
      </header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Active Record</dc:title>
        </oai_dc:dc>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""


# ============================================================================
# Dublin Core Parser Tests
# ============================================================================

class TestDublinCoreParser:
    """Tests for Dublin Core metadata parser."""

    def test_parse_basic_dublin_core(self):
        """Test parsing basic Dublin Core metadata."""
        xml_str = """
        <metadata>
          <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                     xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Test Title</dc:title>
            <dc:creator>Author One</dc:creator>
            <dc:creator>Author Two</dc:creator>
            <dc:subject>Subject 1</dc:subject>
            <dc:subject>Subject 2</dc:subject>
            <dc:description>Test description text.</dc:description>
            <dc:date>2024-05-15</dc:date>
            <dc:identifier>http://example.org/123</dc:identifier>
            <dc:identifier>doi:10.1234/test</dc:identifier>
            <dc:type>Article</dc:type>
            <dc:publisher>Test Publisher</dc:publisher>
            <dc:language>en</dc:language>
          </oai_dc:dc>
        </metadata>
        """
        metadata_elem = ET.fromstring(xml_str)
        result = DublinCoreParser.parse(metadata_elem, {})

        assert result["title"] == "Test Title"
        assert result["creators"] == ["Author One", "Author Two"]
        assert result["subjects"] == ["Subject 1", "Subject 2"]
        assert result["description"] == "Test description text."
        assert result["date"] == "2024-05-15"
        assert "http://example.org/123" in result["identifiers"]
        assert "doi:10.1234/test" in result["identifiers"]
        assert result["types"] == ["Article"]
        assert result["publisher"] == "Test Publisher"
        assert result["languages"] == ["en"]

    def test_parse_empty_fields(self):
        """Test parsing with empty/missing fields."""
        xml_str = """
        <metadata>
          <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                     xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Only Title</dc:title>
          </oai_dc:dc>
        </metadata>
        """
        metadata_elem = ET.fromstring(xml_str)
        result = DublinCoreParser.parse(metadata_elem, {})

        assert result["title"] == "Only Title"
        assert result["creators"] == []
        assert result["description"] == ""
        assert result["date"] == ""


# ============================================================================
# MARC Parser Tests
# ============================================================================

class TestMARCParser:
    """Tests for MARC XML metadata parser."""

    def test_parse_basic_marc(self):
        """Test parsing basic MARC XML metadata."""
        xml_str = """
        <record xmlns="http://www.loc.gov/MARC21/slim">
          <controlfield tag="008">240101s2024    xx            000 0 eng d</controlfield>
          <datafield tag="245" ind1="0" ind2="0">
            <subfield code="a">Test MARC Title</subfield>
            <subfield code="b">A Subtitle</subfield>
          </datafield>
          <datafield tag="100" ind1="1" ind2=" ">
            <subfield code="a">Doe, Jane</subfield>
          </datafield>
          <datafield tag="700" ind1="1" ind2=" ">
            <subfield code="a">Smith, John</subfield>
          </datafield>
          <datafield tag="520" ind1=" " ind2=" ">
            <subfield code="a">This is the abstract.</subfield>
          </datafield>
          <datafield tag="650" ind1=" " ind2="0">
            <subfield code="a">Computer Science</subfield>
          </datafield>
        </record>
        """
        metadata_elem = ET.fromstring(xml_str)
        result = MARCParser.parse(metadata_elem, {})

        assert "Test MARC Title" in result["title"]
        assert "Doe, Jane" in result["creators"]
        assert "Smith, John" in result["creators"]
        assert "This is the abstract." in result["description"]
        assert "Computer Science" in result["subjects"]


# ============================================================================
# DataCite Parser Tests
# ============================================================================

class TestDataciteParser:
    """Tests for DataCite metadata parser."""

    def test_parse_basic_datacite(self):
        """Test parsing basic DataCite metadata."""
        xml_str = """
        <resource xmlns="http://datacite.org/schema/kernel-4">
          <identifier identifierType="DOI">10.1234/test</identifier>
          <creators>
            <creator>
              <creatorName>Doe, Jane</creatorName>
            </creator>
          </creators>
          <titles>
            <title>DataCite Test Title</title>
          </titles>
          <publisher>Test Publisher</publisher>
          <publicationYear>2024</publicationYear>
          <subjects>
            <subject>Data Science</subject>
          </subjects>
          <descriptions>
            <description>Test dataset description.</description>
          </descriptions>
          <resourceType resourceTypeGeneral="Dataset">Dataset</resourceType>
        </resource>
        """
        metadata_elem = ET.fromstring(xml_str)
        result = DataciteParser.parse(metadata_elem, {})

        assert result["title"] == "DataCite Test Title"
        assert "Doe, Jane" in result["creators"]
        assert result["publisher"] == "Test Publisher"
        assert result["date"] == "2024"
        assert "Data Science" in result["subjects"]


# ============================================================================
# OAI-PMH Client Tests
# ============================================================================

class TestOAIPMHClient:
    """Tests for OAI-PMH client."""

    @patch("requests.get")
    def test_identify(self, mock_get):
        """Test Identify verb."""
        mock_response = Mock()
        mock_response.content = IDENTIFY_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        result = client.identify()

        assert result["repository_name"] == "Test Repository"
        assert result["protocol_version"] == "2.0"
        assert "admin@example.org" in result["admin_emails"]
        assert result["deleted_record"] == "persistent"

    @patch("requests.get")
    def test_list_metadata_formats(self, mock_get):
        """Test ListMetadataFormats verb."""
        mock_response = Mock()
        mock_response.content = LIST_METADATA_FORMATS_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        formats = client.list_metadata_formats()

        assert len(formats) == 2
        prefixes = [f["metadataPrefix"] for f in formats]
        assert "oai_dc" in prefixes
        assert "marc21" in prefixes

    @patch("requests.get")
    def test_list_sets(self, mock_get):
        """Test ListSets verb."""
        mock_response = Mock()
        mock_response.content = LIST_SETS_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        sets = list(client.list_sets())

        assert len(sets) == 2
        specs = [s["setSpec"] for s in sets]
        assert "physics" in specs
        assert "math" in specs

    @patch("requests.get")
    def test_list_records(self, mock_get):
        """Test ListRecords verb."""
        mock_response = Mock()
        mock_response.content = LIST_RECORDS_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        records, next_token = client.list_records()

        assert len(records) == 2
        assert next_token is None

        header, metadata = records[0]
        assert header["identifier"] == "oai:example.org:record1"
        assert "physics" in header["setSpecs"]

    @patch("requests.get")
    def test_list_records_with_resumption_token(self, mock_get):
        """Test ListRecords with resumption token."""
        mock_response = Mock()
        mock_response.content = LIST_RECORDS_WITH_RESUMPTION.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        records, next_token = client.list_records()

        assert len(records) == 1
        assert next_token == "token123"

    @patch("requests.get")
    def test_get_record(self, mock_get):
        """Test GetRecord verb."""
        mock_response = Mock()
        mock_response.content = GET_RECORD_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        result = client.get_record("oai:example.org:record1")

        assert result is not None
        header, metadata = result
        assert header["identifier"] == "oai:example.org:record1"

    @patch("requests.get")
    def test_error_handling(self, mock_get):
        """Test OAI-PMH error handling."""
        mock_response = Mock()
        mock_response.content = ERROR_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )

        # get_record catches idDoesNotExist and returns None
        # Test that by checking it returns None for this error
        result = client.get_record("nonexistent")
        assert result is None

    @patch("requests.get")
    def test_error_raises_for_unexpected_errors(self, mock_get):
        """Test OAI-PMH raises for unexpected errors."""
        # Create a response with an unexpected error code
        unexpected_error_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
          <responseDate>2024-01-01T00:00:00Z</responseDate>
          <error code="badVerb">Illegal verb</error>
        </OAI-PMH>
        """
        mock_response = Mock()
        mock_response.content = unexpected_error_xml.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )

        # For unexpected errors like badVerb, the client should raise
        with pytest.raises(OAIPMHError) as exc_info:
            client.identify()

        assert exc_info.value.code == "badVerb"

    @patch("requests.get")
    def test_deleted_records_skipped(self, mock_get):
        """Test that deleted records are properly flagged."""
        mock_response = Mock()
        mock_response.content = DELETED_RECORD_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OAIPMHClient(
            base_url="http://example.org/oai",
            min_interval_seconds=0,
        )
        records, _ = client.list_records()

        # Should have both records, but one marked as deleted
        deleted_records = [r for r in records if r[0].get("deleted")]
        active_records = [r for r in records if not r[0].get("deleted")]

        assert len(deleted_records) == 1
        assert len(active_records) == 1


# ============================================================================
# OAI-PMH Provider Tests
# ============================================================================

class TestOAIPMHProvider:
    """Tests for OAI-PMH provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = OAIPMHProvider({"type": "oai_pmh"})

        assert provider.timeout == 30
        assert provider.min_interval_seconds == 1.0
        assert provider.snippet_max_chars == 700
        assert provider.metadata_prefix == "oai_dc"
        # Default to Zenodo if no repository specified
        assert len(provider.repositories) == 1
        assert provider.repositories[0].name == "Zenodo"

    def test_provider_with_known_repository(self):
        """Test provider with known repository."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "arxiv",
        })

        assert len(provider.repositories) == 1
        assert provider.repositories[0].name == "arXiv"
        assert "arxiv.org" in provider.repositories[0].base_url

    def test_provider_with_custom_url(self):
        """Test provider with custom repository URL."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "http://custom.example.org/oai",
        })

        assert len(provider.repositories) == 1
        assert provider.repositories[0].base_url == "http://custom.example.org/oai"

    def test_provider_with_multiple_repositories(self):
        """Test provider with multiple repositories."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repositories": ["arxiv", "zenodo", "pubmed"],
        })

        assert len(provider.repositories) == 3
        names = [r.name for r in provider.repositories]
        assert "arXiv" in names
        assert "Zenodo" in names
        assert "PubMed Central" in names

    def test_provider_with_custom_repositories_config(self):
        """Test provider with custom repository configuration."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "custom_repositories": {
                "my_repo": {
                    "name": "My Repository",
                    "base_url": "http://myrepo.org/oai",
                    "metadata_prefix": "marc21",
                    "url_template": "http://myrepo.org/item/{id}",
                }
            }
        })

        assert len(provider.repositories) >= 1
        my_repo = next(r for r in provider.repositories if r.name == "My Repository")
        assert my_repo.base_url == "http://myrepo.org/oai"
        assert my_repo.metadata_prefix == "marc21"

    def test_validate_config_empty_repositories(self):
        """Test validation fails with no repositories."""
        provider = OAIPMHProvider({"type": "oai_pmh"})
        provider.repositories = []

        with pytest.raises(ValueError) as exc_info:
            provider.validate_config()

        assert "at least one" in str(exc_info.value).lower()

    def test_get_name_single_repository(self):
        """Test get_name with single repository."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "zenodo",
        })

        name = provider.get_name()
        assert "Zenodo" in name

    def test_get_name_multiple_repositories(self):
        """Test get_name with multiple repositories."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repositories": ["arxiv", "zenodo"],
        })

        name = provider.get_name()
        assert "2 repositories" in name

    @patch("requests.get")
    def test_search_parses_results(self, mock_get):
        """Test search returns parsed results."""
        mock_response = Mock()
        mock_response.content = LIST_RECORDS_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "http://example.org/oai",
            "min_interval_seconds": 0,
        })

        results = provider.search("physics", max_results=5)

        assert len(results) >= 1
        # First result should match physics query
        physics_result = next((r for r in results if "physics" in r.title.lower()), None)
        assert physics_result is not None
        assert "Jane Doe" in physics_result.metadata.get("authors", [])

    @patch("requests.get")
    def test_search_empty_query(self, mock_get):
        """Test search with empty query returns empty list."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "zenodo",
        })

        results = provider.search("", max_results=5)
        assert results == []

    def test_list_available_repositories(self):
        """Test listing known repositories."""
        provider = OAIPMHProvider({"type": "oai_pmh"})
        repos = provider.list_available_repositories()

        assert len(repos) >= 5  # Should have several known repos
        names = [r["name"] for r in repos]
        assert "arXiv" in names
        assert "Zenodo" in names

    @patch("requests.get")
    def test_identify_repository(self, mock_get):
        """Test identifying a repository."""
        mock_response = Mock()
        mock_response.content = IDENTIFY_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "min_interval_seconds": 0,
        })

        info = provider.identify_repository("http://example.org/oai")
        assert info["repository_name"] == "Test Repository"

    @patch("requests.get")
    def test_get_record_by_identifier(self, mock_get):
        """Test getting a specific record."""
        mock_response = Mock()
        mock_response.content = GET_RECORD_RESPONSE.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "http://example.org/oai",
            "min_interval_seconds": 0,
        })

        record = provider.get_record("oai:example.org:record1")
        assert record is not None
        assert record["title"] == "Specific Test Paper"


# ============================================================================
# Repository Configuration Tests
# ============================================================================

class TestKnownRepositories:
    """Tests for known repository configurations."""

    def test_known_repositories_exist(self):
        """Test that expected repositories are configured."""
        expected = ["arxiv", "zenodo", "pubmed", "doaj", "hal", "repec", "philpapers"]
        for repo_key in expected:
            assert repo_key in KNOWN_REPOSITORIES

    def test_repository_has_required_fields(self):
        """Test that repositories have required fields."""
        for key, repo in KNOWN_REPOSITORIES.items():
            assert repo.name, f"{key} missing name"
            assert repo.base_url, f"{key} missing base_url"
            assert repo.base_url.startswith("http"), f"{key} has invalid URL"

    def test_arxiv_has_special_rate_limit(self):
        """Test arXiv has appropriate rate limiting."""
        arxiv = KNOWN_REPOSITORIES["arxiv"]
        assert arxiv.min_interval_seconds >= 3.0  # arXiv requires 3s


# ============================================================================
# Metadata Parser Registry Tests
# ============================================================================

class TestMetadataParsers:
    """Tests for metadata parser registry."""

    def test_oai_dc_parser_registered(self):
        """Test Dublin Core parser is registered."""
        assert "oai_dc" in METADATA_PARSERS
        assert METADATA_PARSERS["oai_dc"] == DublinCoreParser

    def test_marc_parser_registered(self):
        """Test MARC parser is registered."""
        assert "marc21" in METADATA_PARSERS
        assert METADATA_PARSERS["marc21"] == MARCParser

    def test_datacite_parser_registered(self):
        """Test DataCite parser is registered."""
        assert "datacite" in METADATA_PARSERS
        assert METADATA_PARSERS["datacite"] == DataciteParser


# ============================================================================
# Integration Tests
# ============================================================================

class TestOAIPMHIntegration:
    """Integration tests for OAI-PMH provider."""

    def test_provider_available(self):
        """Test provider reports as available."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "repository": "zenodo",
        })
        assert provider.is_available()

    def test_provider_registered(self):
        """Test provider is registered in factory."""
        from lsm.remote import create_remote_provider

        provider = create_remote_provider("oai_pmh", {
            "type": "oai_pmh",
            "repository": "zenodo",
        })
        assert isinstance(provider, OAIPMHProvider)

    def test_year_extraction(self):
        """Test year extraction from various date formats."""
        provider = OAIPMHProvider({"type": "oai_pmh"})

        # Standard format
        assert provider._extract_year("2024-01-15") == 2024
        # ISO format
        assert provider._extract_year("2023-12-01T10:00:00Z") == 2023
        # Year only
        assert provider._extract_year("2022") == 2022
        # Text format
        assert provider._extract_year("January 2021") == 2021
        # Invalid
        assert provider._extract_year("") is None
        assert provider._extract_year("not a date") is None

    def test_truncate(self):
        """Test text truncation."""
        provider = OAIPMHProvider({
            "type": "oai_pmh",
            "snippet_max_chars": 20,
        })

        # Short text unchanged
        assert provider._truncate("Short") == "Short"
        # Long text truncated
        result = provider._truncate("This is a much longer piece of text")
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    def test_citation_formatting(self):
        """Test citation string formatting."""
        provider = OAIPMHProvider({"type": "oai_pmh"})

        citation = provider._format_citation(
            title="Test Paper",
            url="http://example.org/paper",
            authors=["Doe, Jane", "Smith, John"],
            year=2024,
        )

        assert "Test Paper" in citation
        assert "2024" in citation
        assert "Doe, Jane" in citation
        assert "Smith, John" in citation
        assert "http://example.org/paper" in citation

    def test_author_formatting(self):
        """Test author list formatting."""
        provider = OAIPMHProvider({"type": "oai_pmh"})

        # No authors
        assert provider._format_authors([]) == "Unknown"

        # One author
        assert provider._format_authors(["Doe, Jane"]) == "Doe, Jane"

        # Multiple authors
        assert provider._format_authors(["A", "B", "C"]) == "A, B, C"

        # Many authors (truncated)
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert "et al." in result
        assert result.count(",") <= 3
