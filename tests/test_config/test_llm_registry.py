"""
Tests for LLM registry configuration.
"""

from lsm.config.models import LLMRegistryConfig, LLMProviderConfig, FeatureLLMConfig


def test_feature_selection_order():
    registry = LLMRegistryConfig(
        llms=[
            LLMProviderConfig(
                provider_name="openai",
                api_key="test",
                query=FeatureLLMConfig(model="gpt-4"),
                tagging=FeatureLLMConfig(model="gpt-4o-mini"),
            ),
            LLMProviderConfig(
                provider_name="gemini",
                api_key="test",
                query=FeatureLLMConfig(model="gemini-1.5-pro"),
            ),
            LLMProviderConfig(
                provider_name="claude",
                api_key="test",
                ranking=FeatureLLMConfig(model="claude-haiku-4-5"),
            ),
        ]
    )

    query_config = registry.get_query_config()
    tagging_config = registry.get_tagging_config()
    ranking_config = registry.get_ranking_config()

    assert query_config.provider == "gemini"
    assert query_config.model == "gemini-1.5-pro"
    assert tagging_config.provider == "openai"
    assert tagging_config.model == "gpt-4o-mini"
    assert ranking_config.provider == "claude"
    assert ranking_config.model == "claude-haiku-4-5"


def test_set_feature_selection_clears_other_providers():
    registry = LLMRegistryConfig(
        llms=[
            LLMProviderConfig(
                provider_name="openai",
                api_key="test",
                query=FeatureLLMConfig(model="gpt-4"),
            ),
            LLMProviderConfig(
                provider_name="gemini",
                api_key="test",
                query=FeatureLLMConfig(model="gemini-1.5-pro"),
            ),
        ]
    )

    registry.set_feature_selection("query", "gemini", "gemini-1.5-flash")

    openai = registry.get_provider_by_name("openai")
    gemini = registry.get_provider_by_name("gemini")

    assert openai is not None
    assert gemini is not None
    assert openai.query is None
    assert gemini.query is not None
    assert gemini.query.model == "gemini-1.5-flash"
