import pytest

from lsm.config.models.modes import (
    ChainLink,
    ModeConfig,
    OAuthConfig,
    RemoteProviderChainConfig,
    RemoteProviderConfig,
)


def test_mode_config_validate_accepts_default_style() -> None:
    cfg = ModeConfig()
    cfg.validate()


def test_remote_provider_config_has_no_enabled_field() -> None:
    provider = RemoteProviderConfig(name="wikipedia", type="wikipedia", weight=0.7)
    assert not hasattr(provider, "enabled")


def test_remote_provider_config_validate_rejects_negative_weight() -> None:
    provider = RemoteProviderConfig(name="wikipedia", type="wikipedia", weight=-1.0)
    with pytest.raises(ValueError, match="weight must be non-negative"):
        provider.validate()


def test_remote_provider_config_cache_defaults() -> None:
    provider = RemoteProviderConfig(name="wikipedia", type="wikipedia")
    assert provider.cache_results is False
    assert provider.cache_ttl == 86400


def test_remote_provider_config_validate_rejects_invalid_cache_ttl() -> None:
    provider = RemoteProviderConfig(name="wikipedia", type="wikipedia", cache_ttl=0)
    with pytest.raises(ValueError, match="cache_ttl must be positive"):
        provider.validate()


def test_oauth_config_requires_client_id() -> None:
    oauth = OAuthConfig(client_id="", client_secret="secret", scopes=[], redirect_uri="http://localhost")
    with pytest.raises(ValueError, match="oauth.client_id is required"):
        oauth.validate()


def test_chain_link_validate_rejects_bad_map_format() -> None:
    link = ChainLink(source="openalex", map=["badmap"])
    with pytest.raises(ValueError, match="must be 'output:input'"):
        link.validate()


def test_remote_provider_chain_config_requires_links() -> None:
    chain = RemoteProviderChainConfig(name="digest", links=[])
    with pytest.raises(ValueError, match="must include at least one link"):
        chain.validate()
