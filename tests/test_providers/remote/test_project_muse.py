"""
Tests for Project MUSE provider implementation.
"""

from lsm.remote.base import RemoteResult
from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider
from lsm.remote.providers.academic.project_muse import ProjectMUSEProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


class TestProjectMUSEProvider(RemoteProviderOutputTest):
    def test_provider_initialization(self):
        provider = ProjectMUSEProvider({})
        assert provider.name == "project_muse"
        assert provider.repositories[0].name == "Project MUSE"

    def test_search_returns_results(self, monkeypatch):
        provider = ProjectMUSEProvider({})

        def _search_repository(self, _repo, _query, _max_results):
            return [
                RemoteResult(
                    title="MUSE Article",
                    url="https://example.com/muse",
                    snippet="Humanities article",
                    score=0.85,
                )
            ]

        monkeypatch.setattr(OAIPMHProvider, "_search_repository", _search_repository)
        results = provider.search("literature", max_results=1)
        assert len(results) == 1
        self.assert_valid_output(results)
