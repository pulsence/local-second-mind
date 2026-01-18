"""
Tests for query REPL display utilities.
"""

import pytest

from lsm.gui.shell.query.display import (
    display_provider_name,
    format_feature_label,
    stream_output,
)
from lsm.query.session import Candidate, SessionState


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


class TestStreamOutput:
    def test_combines_chunks(self, capsys):
        result = stream_output(["Hello", " ", "world"])
        captured = capsys.readouterr()
        assert "Typing..." in captured.out
        assert "Hello world" in captured.out
        assert result == "Hello world"

    def test_empty_chunks_ignored(self, capsys):
        result = stream_output(["Hello", "", None, "world"])
        assert result == "Helloworld"

    def test_empty_input(self, capsys):
        result = stream_output([])
        captured = capsys.readouterr()
        assert "Typing..." in captured.out
        assert result == ""
