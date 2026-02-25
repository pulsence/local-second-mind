"""
Tests for Perseus CTS provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.cultural.perseus_cts import PerseusCTSProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


CTS_XML_RESPONSE = """<?xml version="1.0"?>
<TEI>
  <text>
    <body>
      <p>μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος</p>
    </body>
  </text>
</TEI>
"""


class TestPerseusCTSProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.text = CTS_XML_RESPONSE
        response.raise_for_status = lambda: None
        mock_get.return_value = response

        provider = PerseusCTSProvider({})
        results = provider.search("urn:cts:greekLit:tlg0012.tlg001:1.1", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["urn"] == "urn:cts:greekLit:tlg0012.tlg001"
        self.assert_valid_output(results)
