"""
Tests for query REPL display utilities.
"""

from lsm.query.commands import display_provider_name, format_feature_label


class TestDisplayProviderName:
    def test_anthropic_becomes_claude(self):
        assert display_provider_name("anthropic") == "claude"

    def test_claude_stays_claude(self):
        assert display_provider_name("claude") == "claude"

    def test_other_providers_unchanged(self):
        assert display_provider_name("openai") == "openai"
        assert display_provider_name("gemini") == "gemini"


class TestFormatFeatureLabel:
    def test_known_features(self):
        assert format_feature_label("query") == "query"
        assert format_feature_label("tagging") == "tag"
        assert format_feature_label("ranking") == "rerank"

    def test_unknown_feature_passthrough(self):
        assert format_feature_label("unknown") == "unknown"


