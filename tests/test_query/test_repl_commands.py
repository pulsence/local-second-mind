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
    CommandResult,
    get_command_handlers,
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


def _run_command(command: str, state: SessionState, config: LSMConfig) -> CommandResult:
    q = command.strip()
    ql = q.lower()
    for handler in get_command_handlers():
        result = handler(q, ql, state, config, Mock(), Mock())
        if result is not None:
            return result
    return CommandResult(output="", handled=False)


def test_unknown_command_shows_help(tmp_path):
    """Test that unknown commands return help text."""
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    result = _run_command("/doesnotexist", state, config)
    assert result.handled is True
    # Unknown commands should return help text
    assert "Commands:" in result.output or "/help" in result.output


def test_mode_set_model_knowledge(tmp_path):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    result = _run_command("/mode set model_knowledge on", state, config)
    assert result.handled is True
    assert config.get_mode_config().source_policy.model_knowledge.enabled is True


def test_note_custom_filename(tmp_path):
    """Test /note command returns action for UI to handle."""
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)
    state.last_question = "Test question"
    state.last_answer = "Test answer"
    state.last_local_sources_for_notes = []
    state.last_remote_sources = []

    result = _run_command("/note custom-note", state, config)
    assert result.handled is True
    # Note command now returns an action for the UI to handle
    assert result.action == "edit_note"
    assert "note_path" in result.action_data
    assert "content" in result.action_data
    assert "Test question" in result.action_data["content"]
    assert "Test answer" in result.action_data["content"]


def test_provider_status_command(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    provider = Mock()
    provider.health_check.return_value = {
        "status": "available",
        "stats": {"success_count": 1, "failure_count": 0},
    }
    monkeypatch.setattr("lsm.query.commands.create_provider", lambda _: provider)

    result = _run_command("/provider-status", state, config)
    assert result.handled is True
    assert "PROVIDER HEALTH STATUS" in result.output


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


def test_remote_providers_command(tmp_path):
    """Test /remote-providers command lists providers."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    result = _run_command("/remote-providers", state, config)
    assert result.handled is True

    assert "REMOTE SOURCE PROVIDERS" in result.output
    assert "wikipedia" in result.output
    assert "arxiv" in result.output
    assert "enabled" in result.output.lower()


def test_remote_provider_enable_command(tmp_path):
    """Test /remote-provider enable command."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    # Brave is initially disabled
    assert config.remote_providers[2].enabled is False

    result = _run_command("/remote-provider enable brave", state, config)
    assert result.handled is True

    # Brave should now be enabled
    assert config.remote_providers[2].enabled is True

    assert "enabled" in result.output.lower()


def test_remote_provider_disable_command(tmp_path):
    """Test /remote-provider disable command."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    # Wikipedia is initially enabled
    assert config.remote_providers[0].enabled is True

    result = _run_command("/remote-provider disable wikipedia", state, config)
    assert result.handled is True

    # Wikipedia should now be disabled
    assert config.remote_providers[0].enabled is False

    assert "disabled" in result.output.lower()


def test_remote_provider_weight_command(tmp_path):
    """Test /remote-provider weight command."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    # Initial weight
    assert config.remote_providers[0].weight == 0.7

    result = _run_command("/remote-provider weight wikipedia 0.95", state, config)
    assert result.handled is True

    # Weight should be updated
    assert config.remote_providers[0].weight == 0.95

    assert "0.95" in result.output


def test_remote_provider_not_found(tmp_path):
    """Test /remote-provider with unknown provider name."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    result = _run_command("/remote-provider enable nonexistent", state, config)
    assert result.handled is True

    assert "not found" in result.output.lower()


def test_remote_search_usage(tmp_path):
    """Test /remote-search shows usage when missing arguments."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    result = _run_command("/remote-search", state, config)
    assert result.handled is True

    assert "Usage:" in result.output


def test_remote_search_all_usage(tmp_path):
    """Test /remote-search-all shows usage when missing query."""
    config = _build_config_with_remote_providers(tmp_path)
    state = SessionState(model=config.llm.get_query_config().model)

    result = _run_command("/remote-search-all", state, config)
    assert result.handled is True

    assert "Usage:" in result.output


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
