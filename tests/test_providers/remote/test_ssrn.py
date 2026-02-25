"""
Tests for SSRN provider implementation.
"""

from lsm.remote.base import RemoteResult
from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider
from lsm.remote.providers.academic.ssrn import SSRNProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


class TestSSRNProvider(RemoteProviderOutputTest):
    def test_provider_initialization(self):
        provider = SSRNProvider({})
        assert provider.name == "ssrn"
        assert provider.repositories[0].name == "SSRN"

    def test_search_returns_results(self, monkeypatch):
        provider = SSRNProvider({})

        def _search_repository(self, _repo, _query, _max_results):
            return [
                RemoteResult(
                    title="SSRN Paper",
                    url="https://example.com/ssrn",
                    snippet="Working paper",
                    score=0.8,
                )
            ]

        monkeypatch.setattr(OAIPMHProvider, "_search_repository", _search_repository)
        results = provider.search("test", max_results=1)
        assert len(results) == 1
        self.assert_valid_output(results)
