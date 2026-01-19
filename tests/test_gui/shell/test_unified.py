"""
Tests for unified interactive shell deprecation.
"""

from pathlib import Path

import pytest

from lsm.gui.shell.unified import UnifiedShell, run_unified_shell
from lsm.config.models import (
    LSMConfig,
    IngestConfig,
    QueryConfig,
    VectorDBConfig,
    LLMRegistryConfig,
    LLMProviderConfig,
    FeatureLLMConfig,
)


@pytest.fixture
def mock_config():
    """Create a mock LSM configuration."""
    return LSMConfig(
        ingest=IngestConfig(
            roots=[Path("/tmp/test")],
            manifest=Path("/tmp/manifest.json"),
        ),
        query=QueryConfig(),
        llm=LLMRegistryConfig(
            llms=[
                LLMProviderConfig(
                    provider_name="openai",
                    api_key="test_key",
                    query=FeatureLLMConfig(model="gpt-5.2"),
                ),
            ]
        ),
        vectordb=VectorDBConfig(
            persist_dir=Path("/tmp/.chroma"),
            collection="test_collection",
        ),
        config_path=Path("/tmp/config.json"),
    )


def test_unified_shell_run_is_disabled(mock_config, capsys):
    """UnifiedShell.run should be disabled in favor of TUI."""
    shell = UnifiedShell(mock_config)
    result = shell.run()

    captured = capsys.readouterr()
    assert result == 2
    assert "Unified shell" in captured.out


def test_run_unified_shell_is_disabled(mock_config, capsys):
    """run_unified_shell should be disabled in favor of TUI."""
    result = run_unified_shell(mock_config)

    captured = capsys.readouterr()
    assert result == 2
    assert "Unified shell" in captured.out
