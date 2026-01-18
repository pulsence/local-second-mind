"""
Tests for unified interactive shell.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from lsm.gui.shell.unified import UnifiedShell
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


class TestUnifiedShell:
    """Tests for UnifiedShell class."""

    def test_shell_initialization(self, mock_config):
        """Test shell initializes with config."""
        shell = UnifiedShell(mock_config)

        assert shell.config == mock_config
        assert shell.current_context is None
        assert shell._ingest_provider is None
        assert shell._query_embedder is None

    def test_print_banner(self, mock_config, capsys):
        """Test shell prints welcome banner."""
        shell = UnifiedShell(mock_config)
        shell.print_banner()

        captured = capsys.readouterr()
        assert "Local Second Mind" in captured.out
        assert "/ingest" in captured.out
        assert "/query" in captured.out
        assert "/exit" in captured.out

    @patch('lsm.vectordb.create_vectordb_provider')
    def test_switch_to_ingest(self, mock_create_provider, mock_config, capsys):
        """Test switching to ingest context."""
        mock_provider = Mock()
        mock_provider.count.return_value = 100
        mock_provider.get_stats.return_value = {"provider": "mock"}
        mock_create_provider.return_value = mock_provider

        shell = UnifiedShell(mock_config)
        shell.switch_to_ingest()

        assert shell.current_context == "ingest"
        assert shell._ingest_provider is not None

        captured = capsys.readouterr()
        assert "INGEST context" in captured.out
        assert "100" in captured.out

    @patch('lsm.vectordb.create_vectordb_provider')
    @patch('lsm.query.retrieval.init_embedder')
    def test_switch_to_query(self, mock_init_embedder, mock_create_provider, mock_config, capsys):
        """Test switching to query context."""
        mock_embedder = Mock()
        mock_provider = Mock()
        mock_provider.count.return_value = 50
        mock_provider.get_stats.return_value = {"provider": "mock"}

        mock_init_embedder.return_value = mock_embedder
        mock_create_provider.return_value = mock_provider

        # Create persist dir
        mock_config.vectordb.persist_dir.mkdir(parents=True, exist_ok=True)

        shell = UnifiedShell(mock_config)
        shell.switch_to_query()

        assert shell.current_context == "query"
        assert shell._query_embedder is not None
        assert shell._query_provider is not None
        assert shell._query_state is not None

        captured = capsys.readouterr()
        assert "QUERY context" in captured.out
        assert "50" in captured.out

    @patch('lsm.vectordb.create_vectordb_provider')
    def test_context_switch_commands(self, mock_create_provider, mock_config):
        """Test context switch commands are handled."""
        mock_provider = Mock()
        mock_provider.count.return_value = 100
        mock_provider.get_stats.return_value = {"provider": "mock"}
        mock_create_provider.return_value = mock_provider

        shell = UnifiedShell(mock_config)
        shell.switch_to_ingest()  # Initialize ingest context first

        # Commands should trigger context switches
        with patch.object(shell, 'switch_to_query') as mock_switch:
            shell.handle_ingest_command("/query")
            # Should call switch_to_query when /query command is issued
            mock_switch.assert_called_once()

    @patch('builtins.input', side_effect=["/exit"])
    def test_shell_exit(self, mock_input, mock_config):
        """Test shell exits on /exit command."""
        shell = UnifiedShell(mock_config)
        result = shell.run()

        assert result == 0

    @patch('builtins.input', side_effect=["/ingest", "/query", "/exit"])
    @patch('lsm.vectordb.create_vectordb_provider')
    @patch('lsm.query.retrieval.init_embedder')
    def test_shell_context_switching(self, mock_init_emb, mock_create_provider, mock_input, mock_config):
        """Test shell can switch between contexts."""
        # Setup mocks
        ingest_provider = Mock(count=Mock(return_value=10), get_stats=Mock(return_value={"provider": "mock"}))
        query_provider = Mock(count=Mock(return_value=20), get_stats=Mock(return_value={"provider": "mock"}))
        mock_create_provider.side_effect = [ingest_provider, query_provider]
        mock_init_emb.return_value = Mock()

        # Create persist dir
        mock_config.vectordb.persist_dir.mkdir(parents=True, exist_ok=True)

        shell = UnifiedShell(mock_config)

        # This will go through: /ingest, /query, /exit
        result = shell.run()

        assert result == 0
        # Should have switched to both contexts
        assert shell._ingest_provider is not None
        assert shell._query_provider is not None


class TestShellHelpers:
    """Test shell helper functions."""

    def test_show_help_no_context(self, mock_config, capsys):
        """Test help when no context is selected."""
        shell = UnifiedShell(mock_config)
        shell.show_help()

        captured = capsys.readouterr()
        assert "Global Commands" in captured.out
        assert "/ingest" in captured.out
        assert "/query" in captured.out

    @patch('lsm.vectordb.create_vectordb_provider')
    @patch('lsm.gui.shell.ingest.display.print_help')
    def test_show_help_ingest_context(self, mock_print_help, mock_create_provider, mock_config):
        """Test help shows ingest commands when in ingest context."""
        mock_create_provider.return_value = Mock(count=Mock(return_value=10), get_stats=Mock(return_value={"provider": "mock"}))

        shell = UnifiedShell(mock_config)
        shell.switch_to_ingest()
        shell.show_help()

        # Should call ingest help
        mock_print_help.assert_called_once()

    @patch('lsm.vectordb.create_vectordb_provider')
    @patch('lsm.query.retrieval.init_embedder')
    @patch('lsm.gui.shell.query.display.print_help')
    def test_show_help_query_context(self, mock_print_help, mock_init_emb, mock_create_provider, mock_config):
        """Test help shows query commands when in query context."""
        mock_init_emb.return_value = Mock()
        mock_create_provider.return_value = Mock(count=Mock(return_value=10), get_stats=Mock(return_value={"provider": "mock"}))
        mock_config.vectordb.persist_dir.mkdir(parents=True, exist_ok=True)

        shell = UnifiedShell(mock_config)
        shell.switch_to_query()
        shell.show_help()

        # Should call query help
        mock_print_help.assert_called_once()
