from __future__ import annotations

import xml.etree.ElementTree as ET

from lsm.remote.providers.base_oai import BaseOAIProvider, OAIPMHClient, OAIRepository


class DummyOAIProvider(BaseOAIProvider):
    """Minimal OAI provider for shared helper tests."""

    def get_name(self) -> str:
        return "dummy"

    def search(self, query: str, max_results: int = 5):
        return []


def test_base_oai_parse_record_and_mapping() -> None:
    provider = DummyOAIProvider({"snippet_max_chars": 80})
    repo = OAIRepository(name="ExampleRepo", base_url="http://example.org/oai")

    header = {
        "identifier": "oai:example.org:record1",
        "datestamp": "2024-01-02",
        "setSpecs": [],
        "deleted": False,
    }

    metadata_xml = """<metadata xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
        xmlns:dc="http://purl.org/dc/elements/1.1/">
        <oai_dc:dc>
            <dc:title>Sample Record</dc:title>
            <dc:creator>Jane Doe</dc:creator>
            <dc:subject>Testing</dc:subject>
            <dc:description>Example description.</dc:description>
            <dc:date>2024-01-02</dc:date>
            <dc:identifier>https://example.org/item/record1</dc:identifier>
        </oai_dc:dc>
    </metadata>"""

    metadata_elem = ET.fromstring(metadata_xml)
    record = provider._parse_record(header, metadata_elem, repo)

    assert record is not None
    assert record.title == "Sample Record"
    assert record.year == 2024

    result = provider._record_to_result(record, repo, index=0, max_results=5)
    assert result is not None
    assert result.title == "Sample Record"
    assert result.url.startswith("https://example.org/item")
    assert result.metadata["oai_identifier"] == "oai:example.org:record1"
    assert result.metadata["year"] == 2024


def test_oai_client_resumption_token(monkeypatch) -> None:
    xml = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
        xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
        xmlns:dc="http://purl.org/dc/elements/1.1/">
      <ListRecords>
        <record>
          <header>
            <identifier>oai:example.org:record1</identifier>
            <datestamp>2024-01-02</datestamp>
          </header>
          <metadata>
            <oai_dc:dc>
              <dc:title>Record 1</dc:title>
            </oai_dc:dc>
          </metadata>
        </record>
        <resumptionToken>token-123</resumptionToken>
      </ListRecords>
    </OAI-PMH>"""

    client = OAIPMHClient(base_url="http://example.org/oai")
    monkeypatch.setattr(client, "_request", lambda params: ET.fromstring(xml))

    records, token = client.list_records(metadata_prefix="oai_dc")
    assert token == "token-123"
    assert records[0][0]["identifier"] == "oai:example.org:record1"
