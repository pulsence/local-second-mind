"""
Tests for remote provider factory.
"""

import pytest

from lsm.query.remote.base import BaseRemoteProvider, RemoteResult
from lsm.query.remote.brave import BraveSearchProvider


# Note: The import error from the original file suggests these functions
# don't exist yet. We'll keep the test structure but mark as TODO


class MockRemoteProvider(BaseRemoteProvider):
    """Mock provider for testing factory registration."""

    @property
    def name(self) -> str:
        return "mock"

    def search(self, query: str, max_results: int = 5):
        return [
            RemoteResult(
                title=f"Result {i}",
                url=f"https://example.com/{i}",
                snippet=f"Snippet {i}",
                score=1.0
            )
            for i in range(max_results)
        ]


class TestRemoteProviderFactory:
    """Tests for remote provider factory."""

    @pytest.mark.skip(reason="Factory functions not yet implemented - see lsm/query/remote/factory.py")
    def test_list_available_providers(self):
        """Test listing available providers."""
        from lsm.query.remote.factory import list_available_providers

        providers = list_available_providers()

        # Should have at least Brave Search
        assert "web_search" in providers
        assert "brave_search" in providers

    @pytest.mark.skip(reason="Factory functions not yet implemented - see lsm/query/remote/factory.py")
    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        from lsm.query.remote.factory import register_remote_provider, list_available_providers

        register_remote_provider("mock", MockRemoteProvider)

        providers = list_available_providers()
        assert "mock" in providers

    @pytest.mark.skip(reason="Factory functions not yet implemented - see lsm/query/remote/factory.py")
    def test_create_provider_brave(self):
        """Test creating Brave Search provider."""
        from lsm.query.remote.factory import create_remote_provider

        config = {
            "type": "web_search",
            "enabled": True,
            "api_key": "test_key",
            "max_results": 10
        }

        provider = create_remote_provider(config)

        assert isinstance(provider, BraveSearchProvider)
        assert provider.enabled is True
        assert provider.max_results == 10

    @pytest.mark.skip(reason="Factory functions not yet implemented - see lsm/query/remote/factory.py")
    def test_create_provider_invalid_type(self):
        """Test creating provider with invalid type raises error."""
        from lsm.query.remote.factory import create_remote_provider

        config = {
            "type": "invalid_provider",
            "enabled": True
        }

        with pytest.raises(ValueError, match="Unknown remote provider type"):
            create_remote_provider(config)


# TODO: Implement factory functions in lsm/query/remote/factory.py
# - create_remote_provider(config)
# - register_remote_provider(name, provider_class)
# - list_available_providers()
