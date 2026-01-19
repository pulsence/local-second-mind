"""
Tests for query REPL command handling.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lsm.config.models import (
    LSMConfig,
    IngestConfig,
    QueryConfig,
    VectorDBConfig,
    LLMRegistryConfig,
    LLMProviderConfig,
    FeatureLLMConfig,
    RemoteProviderConfig,
    RemoteProviderRef,
    RemoteSourcePolicy,
)
from lsm.query.commands import (
    handle_command,
    toggle_remote_provider,
    set_remote_provider_weight,
)
from lsm.query.session import SessionState


def _build_config(tmp_path: Path) -> LSMConfig:
    ingest = IngestConfig(
        roots=[tmp_path],
        manifest=tmp_path / "manifest.json",
    )
    query = QueryConfig()
    llm = LLMRegistryConfig(
        llms=[
            LLMProviderConfig(
                provider_name="openai",
                api_key="test-key",
                query=FeatureLLMConfig(model="gpt-5.2"),
            ),
        ]
    )
    vectordb = VectorDBConfig(persist_dir=tmp_path / ".chroma")
    config = LSMConfig(
        ingest=ingest,
        query=query,
        llm=llm,
        vectordb=vectordb,
        config_path=tmp_path / "config.json",
    )
    return config


def test_unknown_command_shows_help(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    called = {"help": False}

    def _help():
        called["help"] = True

    monkeypatch.setattr("lsm.query.commands.print_help", _help)

    handled = handle_command("/doesnotexist", state, config, Mock(), Mock())
    assert handled is True
    assert called["help"] is True


def test_mode_set_model_knowledge(tmp_path):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    handled = handle_command("/mode set model_knowledge on", state, config, Mock(), Mock())
    assert handled is True
    assert config.get_mode_config().source_policy.model_knowledge.enabled is True


def test_note_custom_filename(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)
    state.last_question = "Test question"
    state.last_answer = "Test answer"
    state.last_local_sources_for_notes = []
    state.last_remote_sources = []

    monkeypatch.setattr("lsm.query.commands.edit_note_in_editor", lambda content: content)

    handled = handle_command("/note custom-note", state, config, Mock(), Mock())
    assert handled is True

    note_path = tmp_path / "notes" / "custom-note.md"
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "Test question" in content
    assert "Test answer" in content


def test_provider_status_command(tmp_path, monkeypatch, capsys):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    provider = Mock()
    provider.health_check.return_value = {
        "status": "available",
        "stats": {"success_count": 1, "failure_count": 0},
    }
    monkeypatch.setattr("lsm.query.commands.create_provider", lambda _: provider)

    handled = handle_command("/provider-status", state, config, Mock(), Mock())
    assert handled is True
    captured = capsys.readouterr()
    assert "PROVIDER HEALTH STATUS" in captured.out


# -----------------------------
# Remote Provider Command Tests
# -----------------------------
def _build_config_with_remote_providers(tmp_path: Path) -> LSMConfig:
    """Build config with remote providers for testing."""
    config = _build_config(tmp_path)
    config.remote_providers = [
        RemoteProviderConfig(name="wikipedia", type="wikipedia", enabled=True, weight=0.7),
        RemoteProviderConfig(name="arxiv", type="arxiv", enabled=True, weight=0.9),
        RemoteProviderConfig(name="brave", type="web_search", enabled=False, weight=1.0),
    ]
    return config


def test_remote_providers_command(tmp_path, capsys):
    """Test /remote-providers command lists providers."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    handled = handle_command("/remote-providers", state, config, Mock(), Mock())
    assert handled is True

    captured = capsys.readouterr()
    assert "REMOTE SOURCE PROVIDERS" in captured.out
    assert "wikipedia" in captured.out
    assert "arxiv" in captured.out
    assert "enabled" in captured.out.lower()


def test_remote_provider_enable_command(tmp_path, capsys):
    """Test /remote-provider enable command."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    # Brave is initially disabled
    assert config.remote_providers[2].enabled is False

    handled = handle_command("/remote-provider enable brave", state, config, Mock(), Mock())
    assert handled is True

    # Brave should now be enabled
    assert config.remote_providers[2].enabled is True

    captured = capsys.readouterr()
    assert "enabled" in captured.out.lower()


def test_remote_provider_disable_command(tmp_path, capsys):
    """Test /remote-provider disable command."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    # Wikipedia is initially enabled
    assert config.remote_providers[0].enabled is True

    handled = handle_command("/remote-provider disable wikipedia", state, config, Mock(), Mock())
    assert handled is True

    # Wikipedia should now be disabled
    assert config.remote_providers[0].enabled is False

    captured = capsys.readouterr()
    assert "disabled" in captured.out.lower()


def test_remote_provider_weight_command(tmp_path, capsys):
    """Test /remote-provider weight command."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    # Initial weight
    assert config.remote_providers[0].weight == 0.7

    handled = handle_command("/remote-provider weight wikipedia 0.95", state, config, Mock(), Mock())
    assert handled is True

    # Weight should be updated
    assert config.remote_providers[0].weight == 0.95

    captured = capsys.readouterr()
    assert "0.95" in captured.out


def test_remote_provider_not_found(tmp_path, capsys):
    """Test /remote-provider with unknown provider name."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    handled = handle_command("/remote-provider enable nonexistent", state, config, Mock(), Mock())
    assert handled is True

    captured = capsys.readouterr()
    assert "not found" in captured.out.lower()


def test_remote_search_usage(tmp_path, capsys):
    """Test /remote-search shows usage when missing arguments."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    handled = handle_command("/remote-search", state, config, Mock(), Mock())
    assert handled is True

    captured = capsys.readouterr()
    assert "Usage:" in captured.out


def test_remote_search_all_usage(tmp_path, capsys):
    """Test /remote-search-all shows usage when missing query."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    handled = handle_command("/remote-search-all", state, config, Mock(), Mock())
    assert handled is True

    captured = capsys.readouterr()
    assert "Usage:" in captured.out


# -----------------------------
# Helper Function Tests
# -----------------------------
def test_toggle_remote_provider():
    """Test toggle_remote_provider helper function."""
    config = Mock()
    config.remote_providers = [
        RemoteProviderConfig(name="test", type="test", enabled=True),
    ]

    # Toggle to disabled
    result = toggle_remote_provider(config, "test", False)
    assert result is True
    assert config.remote_providers[0].enabled is False

    # Toggle back to enabled
    result = toggle_remote_provider(config, "test", True)
    assert result is True
    assert config.remote_providers[0].enabled is True

    # Unknown provider
    result = toggle_remote_provider(config, "unknown", True)
    assert result is False


def test_set_remote_provider_weight():
    """Test set_remote_provider_weight helper function."""
    config = Mock()
    config.remote_providers = [
        RemoteProviderConfig(name="test", type="test", weight=0.5),
    ]

    # Set new weight
    result = set_remote_provider_weight(config, "test", 0.9)
    assert result is True
    assert config.remote_providers[0].weight == 0.9

    # Unknown provider
    result = set_remote_provider_weight(config, "unknown", 0.5)
    assert result is False


# -----------------------------
# RemoteSourcePolicy Tests
# -----------------------------
def test_remote_source_policy_get_provider_names_strings():
    """Test get_provider_names with string list."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=["brave", "wikipedia", "arxiv"]
    )

    names = policy.get_provider_names()
    assert names == ["brave", "wikipedia", "arxiv"]


def test_remote_source_policy_get_provider_names_refs():
    """Test get_provider_names with RemoteProviderRef objects."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=[
            RemoteProviderRef(source="brave", weight=0.6),
            RemoteProviderRef(source="arxiv", weight=0.9),
        ]
    )

    names = policy.get_provider_names()
    assert names == ["brave", "arxiv"]


def test_remote_source_policy_get_provider_names_dicts():
    """Test get_provider_names with dict format (from JSON config)."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=[
            {"source": "brave", "weight": 0.6},
            {"source": "arxiv", "weight": 0.9},
        ]
    )

    names = policy.get_provider_names()
    assert names == ["brave", "arxiv"]


def test_remote_source_policy_get_provider_weight_string():
    """Test get_provider_weight returns None for string provider (uses global weight)."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=["brave", "wikipedia"]
    )

    weight = policy.get_provider_weight("brave")
    assert weight is None  # Should use global weight


def test_remote_source_policy_get_provider_weight_ref():
    """Test get_provider_weight returns override weight from RemoteProviderRef."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=[
            RemoteProviderRef(source="brave", weight=0.6),
            RemoteProviderRef(source="arxiv", weight=0.95),
        ]
    )

    assert policy.get_provider_weight("brave") == 0.6
    assert policy.get_provider_weight("arxiv") == 0.95
    assert policy.get_provider_weight("unknown") is None


def test_remote_source_policy_get_provider_weight_dict():
    """Test get_provider_weight returns override weight from dict format."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=[
            {"source": "brave", "weight": 0.6},
            {"source": "arxiv", "weight": 0.95},
        ]
    )

    assert policy.get_provider_weight("brave") == 0.6
    assert policy.get_provider_weight("arxiv") == 0.95


def test_remote_source_policy_mixed_format():
    """Test get_provider_names and weights with mixed string/ref format."""
    policy = RemoteSourcePolicy(
        enabled=True,
        remote_providers=[
            "wikipedia",  # String - uses global weight
            {"source": "arxiv", "weight": 0.95},  # Dict with override
        ]
    )

    names = policy.get_provider_names()
    assert names == ["wikipedia", "arxiv"]

    assert policy.get_provider_weight("wikipedia") is None  # Global weight
    assert policy.get_provider_weight("arxiv") == 0.95  # Override


def test_remote_provider_ref_validation():
    """Test RemoteProviderRef validation."""
    # Valid ref
    ref = RemoteProviderRef(source="test", weight=0.5)
    ref.validate()  # Should not raise

    # Invalid negative weight
    ref_invalid = RemoteProviderRef(source="test", weight=-0.5)
    with pytest.raises(ValueError, match="non-negative"):
        ref_invalid.validate()
