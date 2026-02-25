"""
Tests for PhilArchive provider implementation.
"""

from lsm.remote.base import RemoteResult
from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider
from lsm.remote.providers.academic.philarchive import PhilArchiveProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


class TestPhilArchiveProvider(RemoteProviderOutputTest):
    def test_provider_initialization(self):
        provider = PhilArchiveProvider({})
        assert provider.name == "philarchive"
        assert provider.repositories[0].name == "PhilArchive"

    def test_search_returns_results(self, monkeypatch):
        provider = PhilArchiveProvider({})

        def _search_repository(self, _repo, _query, _max_results):
            return [
                RemoteResult(
                    title="PhilArchive Paper",
                    url="https://example.com/philarchive",
                    snippet="Philosophy preprint",
                    score=0.9,
                )
            ]

        monkeypatch.setattr(OAIPMHProvider, "_search_repository", _search_repository)
        results = provider.search("ethics", max_results=1)
        assert len(results) == 1
        self.assert_valid_output(results)
