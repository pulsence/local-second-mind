import pytest

from lsm.config.models.modes import ModeConfig, RemoteProviderConfig


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
