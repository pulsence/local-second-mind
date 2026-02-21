"""
Tests for LLM registry configuration (providers/services model).
"""

import pytest

from lsm.config.models import (
    LLMRegistryConfig,
    LLMProviderConfig,
    LLMServiceConfig,
    LLMTierConfig,
)


def test_resolve_service_basic():
    """Test basic service resolution merges provider + service."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai", api_key="test-key"),
        ],
        services={
            "query": LLMServiceConfig(
                provider="openai", model="gpt-5.2", temperature=0.7, max_tokens=2000
            ),
        },
    )
    config = registry.resolve_service("query")
    assert config.provider == "openai"
    assert config.model == "gpt-5.2"
    assert config.api_key == "test-key"
    assert config.temperature == 0.7
    assert config.max_tokens == 2000


def test_resolve_service_falls_back_to_default():
    """Test that unknown service names fall back to 'default'."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "default": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    config = registry.resolve_service("some_custom_service")
    assert config.model == "gpt-5.2"
    assert config.provider == "openai"


def test_resolve_tier_basic() -> None:
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "default": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
        tiers={
            "quick": LLMTierConfig(provider="openai", model="gpt-5-nano"),
        },
    )
    config = registry.resolve_tier("quick")
    assert config.provider == "openai"
    assert config.model == "gpt-5-nano"


def test_resolve_direct_uses_provider_details() -> None:
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="azure_openai", api_key="azure-key", endpoint="https://example"),
        ],
        services={
            "default": LLMServiceConfig(provider="azure_openai", model="gpt-4"),
        },
    )
    config = registry.resolve_direct("azure_openai", "gpt-4o", temperature=0.1, max_tokens=123)
    assert config.provider == "azure_openai"
    assert config.model == "gpt-4o"
    assert config.api_key == "azure-key"
    assert config.endpoint == "https://example"
    assert config.temperature == 0.1
    assert config.max_tokens == 123


def test_resolve_service_raises_when_no_default():
    """Test that missing service without default raises ValueError."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    with pytest.raises(ValueError, match="No LLM service configured for 'tagging'"):
        registry.resolve_service("tagging")


def test_resolve_service_raises_for_missing_provider():
    """Test that referencing a nonexistent provider raises ValueError."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="nonexistent", model="model"),
        },
    )
    with pytest.raises(ValueError, match="not in the providers list"):
        registry.resolve_service("query")


def test_multi_provider_services():
    """Test services pointing to different providers."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai", api_key="oai-key"),
            LLMProviderConfig(provider_name="anthropic", api_key="ant-key"),
            LLMProviderConfig(provider_name="gemini", api_key="gem-key"),
        ],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            "tagging": LLMServiceConfig(provider="gemini", model="gemini-2.5-flash-lite"),
            "ranking": LLMServiceConfig(provider="anthropic", model="claude-haiku-4-5"),
        },
    )
    assert registry.get_query_config().provider == "openai"
    assert registry.get_query_config().model == "gpt-5.2"
    assert registry.get_tagging_config().provider == "gemini"
    assert registry.get_tagging_config().model == "gemini-2.5-flash-lite"
    assert registry.get_ranking_config().provider == "anthropic"
    assert registry.get_ranking_config().model == "claude-haiku-4-5"


def test_get_feature_provider_map():
    """Test feature provider map generation."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            "tagging": LLMServiceConfig(provider="openai", model="gpt-5-nano"),
        },
    )
    mapping = registry.get_feature_provider_map()
    assert mapping == {"query": "openai", "tagging": "openai"}
    assert "ranking" not in mapping


def test_set_feature_selection_updates_existing():
    """Test setting provider + model for an existing service."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai", api_key="key"),
            LLMProviderConfig(provider_name="gemini", api_key="key"),
        ],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    registry.set_feature_selection("query", "gemini", "gemini-1.5-pro")
    config = registry.get_query_config()
    assert config.provider == "gemini"
    assert config.model == "gemini-1.5-pro"


def test_set_feature_selection_creates_new_service():
    """Test that set_feature_selection creates a new service entry."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    registry.set_feature_selection("tagging", "openai", "gpt-5-nano")
    assert "tagging" in registry.services
    config = registry.get_tagging_config()
    assert config.model == "gpt-5-nano"
    assert config.provider == "openai"


def test_set_feature_selection_rejects_unknown_provider():
    """Test that set_feature_selection raises for unknown provider."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    with pytest.raises(ValueError, match="not found in providers list"):
        registry.set_feature_selection("query", "nonexistent", "model")


def test_provider_connection_details_merged():
    """Test that Azure connection details are properly merged into resolved config."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(
                provider_name="azure_openai",
                api_key="azure-key",
                endpoint="https://myresource.openai.azure.com",
                api_version="2024-02-15",
                deployment_name="my-deployment",
            ),
        ],
        services={
            "query": LLMServiceConfig(provider="azure_openai", model="gpt-4"),
        },
    )
    config = registry.resolve_service("query")
    assert config.provider == "azure_openai"
    assert config.api_key == "azure-key"
    assert config.endpoint == "https://myresource.openai.azure.com"
    assert config.api_version == "2024-02-15"
    assert config.deployment_name == "my-deployment"


def test_service_temperature_and_max_tokens_defaults():
    """Test that None temperature/max_tokens use defaults."""
    from lsm.config.models.constants import DEFAULT_LLM_TEMPERATURE, DEFAULT_LLM_MAX_TOKENS

    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    config = registry.resolve_service("query")
    assert config.temperature == DEFAULT_LLM_TEMPERATURE
    assert config.max_tokens == DEFAULT_LLM_MAX_TOKENS


def test_override_feature_model():
    """Test overriding just the model on an existing service."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    registry.override_feature_model("query", "gpt-4o")
    assert registry.resolve_service("query").model == "gpt-4o"


def test_override_feature_model_falls_back_to_default():
    """Test override_feature_model falls back to default service."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "default": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    registry.override_feature_model("query", "gpt-4o")
    assert registry.services["default"].model == "gpt-4o"


def test_override_feature_model_raises_when_no_service():
    """Test override_feature_model raises when no matching service or default."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    with pytest.raises(ValueError, match="No service 'tagging' or 'default'"):
        registry.override_feature_model("tagging", "gpt-4o")


def test_resolve_any_for_provider():
    """Test resolving any service that uses a given provider."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai", api_key="key"),
            LLMProviderConfig(provider_name="gemini", api_key="key"),
        ],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
            "tagging": LLMServiceConfig(provider="gemini", model="gemini-2.5-flash"),
        },
    )
    config = registry.resolve_any_for_provider("gemini")
    assert config is not None
    assert config.provider == "gemini"


def test_resolve_any_for_provider_returns_none():
    """Test resolve_any_for_provider returns None for unused provider."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai", api_key="key"),
            LLMProviderConfig(provider_name="gemini", api_key="key"),
        ],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    assert registry.resolve_any_for_provider("gemini") is None


def test_get_provider_names():
    """Test getting all provider names."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai"),
            LLMProviderConfig(provider_name="gemini"),
        ],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    assert registry.get_provider_names() == ["openai", "gemini"]


def test_get_provider_by_name():
    """Test looking up a provider by name."""
    registry = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(provider_name="openai", api_key="oai-key"),
            LLMProviderConfig(provider_name="gemini", api_key="gem-key"),
        ],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    provider = registry.get_provider_by_name("gemini")
    assert provider is not None
    assert provider.api_key == "gem-key"
    assert registry.get_provider_by_name("nonexistent") is None


def test_validate_requires_query_or_default():
    """Test that validation requires 'query' or 'default' service."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "tagging": LLMServiceConfig(provider="openai", model="gpt-5-nano"),
        },
    )
    with pytest.raises(ValueError, match="'query' or 'default'"):
        registry.validate()


def test_validate_accepts_default_only():
    """Test that validation passes with only a 'default' service."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "default": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    registry.validate()


def test_validate_catches_missing_provider_ref():
    """Test that validation catches service referencing nonexistent provider."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={
            "query": LLMServiceConfig(provider="nonexistent", model="model"),
        },
    )
    with pytest.raises(ValueError, match="not in the providers list"):
        registry.validate()


def test_validate_empty_providers_raises():
    """Test that empty providers list raises."""
    registry = LLMRegistryConfig(
        providers=[],
        services={
            "query": LLMServiceConfig(provider="openai", model="gpt-5.2"),
        },
    )
    with pytest.raises(ValueError, match="non-empty list"):
        registry.validate()


def test_validate_empty_services_raises():
    """Test that empty services dict raises."""
    registry = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="openai", api_key="key")],
        services={},
    )
    with pytest.raises(ValueError, match="non-empty dict"):
        registry.validate()


def test_service_validate_requires_provider():
    """Test LLMServiceConfig validation requires provider."""
    svc = LLMServiceConfig(provider="", model="gpt-5.2")
    with pytest.raises(ValueError, match="provider"):
        svc.validate()


def test_service_validate_requires_model():
    """Test LLMServiceConfig validation requires model."""
    svc = LLMServiceConfig(provider="openai", model="")
    with pytest.raises(ValueError, match="model"):
        svc.validate()


def test_service_validate_temperature_range():
    """Test LLMServiceConfig validation checks temperature range."""
    svc = LLMServiceConfig(provider="openai", model="gpt-5.2", temperature=3.0)
    with pytest.raises(ValueError, match="Temperature"):
        svc.validate()


def test_service_validate_max_tokens_positive():
    """Test LLMServiceConfig validation checks max_tokens positive."""
    svc = LLMServiceConfig(provider="openai", model="gpt-5.2", max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens"):
        svc.validate()


def test_provider_validate_requires_name():
    """Test LLMProviderConfig validation requires provider_name."""
    provider = LLMProviderConfig(provider_name="")
    with pytest.raises(ValueError, match="provider_name"):
        provider.validate()
