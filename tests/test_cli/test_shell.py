"""
Tests for unified interactive shell.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from lsm.cli.shell import UnifiedShell
from lsm.config.models import LSMConfig, IngestConfig, QueryConfig, LLMConfig


@pytest.fixture
def mock_config():
    """Create a mock LSM configuration."""
    return LSMConfig(
        ingest=IngestConfig(
            roots=[Path("/tmp/test")],
            manifest=Path("/tmp/manifest.json"),
        ),
        query=QueryConfig(),
        llm=LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test_key"
        ),
        persist_dir=Path("/tmp/.chroma"),
        collection="test_collection",
        embed_model="test-model",
        device="cpu",
        batch_size=32,
    )


class TestUnifiedShell:
    """Tests for UnifiedShell class."""

    def test_shell_initialization(self, mock_config):
        """Test shell initializes with config."""
        shell = UnifiedShell(mock_config)

        assert shell.config == mock_config
        assert shell.current_context is None
        assert shell._ingest_collection is None
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

    @patch('lsm.cli.shell.get_chroma_collection')
    def test_switch_to_ingest(self, mock_get_collection, mock_config, capsys):
        """Test switching to ingest context."""
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_get_collection.return_value = mock_collection

        shell = UnifiedShell(mock_config)
        shell.switch_to_ingest()

        assert shell.current_context == "ingest"
        assert shell._ingest_collection is not None

        captured = capsys.readouterr()
        assert "INGEST context" in captured.out
        assert "100" in captured.out

    @patch('lsm.cli.shell.init_embedder')
    @patch('lsm.cli.shell.init_collection')
    @patch('lsm.cli.shell.OpenAI')
    def test_switch_to_query(self, mock_openai, mock_init_collection, mock_init_embedder, mock_config, capsys):
        """Test switching to query context."""
        mock_embedder = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 50
        mock_client = Mock()

        mock_init_embedder.return_value = mock_embedder
        mock_init_collection.return_value = mock_collection
        mock_openai.return_value = mock_client

        # Create persist dir
        mock_config.persist_dir.mkdir(parents=True, exist_ok=True)

        shell = UnifiedShell(mock_config)
        shell.switch_to_query()

        assert shell.current_context == "query"
        assert shell._query_embedder is not None
        assert shell._query_collection is not None
        assert shell._query_client is not None
        assert shell._query_state is not None

        captured = capsys.readouterr()
        assert "QUERY context" in captured.out
        assert "50" in captured.out

    def test_context_switch_commands(self, mock_config):
        """Test context switch commands are handled."""
        shell = UnifiedShell(mock_config)

        # Initially no context
        assert shell.current_context is None

        # Commands should trigger context switches
        with patch.object(shell, 'switch_to_ingest') as mock_switch:
            shell.handle_ingest_command("/query")
            # Should not call switch_to_ingest because we're checking for /query
            mock_switch.assert_not_called()

    @patch('builtins.input', side_effect=["/exit"])
    def test_shell_exit(self, mock_input, mock_config):
        """Test shell exits on /exit command."""
        shell = UnifiedShell(mock_config)
        result = shell.run()

        assert result == 0

    @patch('builtins.input', side_effect=["/ingest", "/query", "/exit"])
    @patch('lsm.cli.shell.get_chroma_collection')
    @patch('lsm.cli.shell.init_embedder')
    @patch('lsm.cli.shell.init_collection')
    @patch('lsm.cli.shell.OpenAI')
    def test_shell_context_switching(self, mock_openai, mock_init_coll, mock_init_emb, mock_get_coll, mock_input, mock_config):
        """Test shell can switch between contexts."""
        # Setup mocks
        mock_get_coll.return_value = Mock(count=Mock(return_value=10))
        mock_init_emb.return_value = Mock()
        mock_init_coll.return_value = Mock(count=Mock(return_value=20))
        mock_openai.return_value = Mock()

        # Create persist dir
        mock_config.persist_dir.mkdir(parents=True, exist_ok=True)

        shell = UnifiedShell(mock_config)

        # This will go through: /ingest, /query, /exit
        result = shell.run()

        assert result == 0
        # Should have switched to both contexts
        assert shell._ingest_collection is not None
        assert shell._query_collection is not None


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

    @patch('lsm.cli.shell.get_chroma_collection')
    @patch('lsm.ingest.repl.print_help')
    def test_show_help_ingest_context(self, mock_print_help, mock_get_coll, mock_config):
        """Test help shows ingest commands when in ingest context."""
        mock_get_coll.return_value = Mock(count=Mock(return_value=10))

        shell = UnifiedShell(mock_config)
        shell.switch_to_ingest()
        shell.show_help()

        # Should call ingest help
        mock_print_help.assert_called_once()

    @patch('lsm.cli.shell.init_embedder')
    @patch('lsm.cli.shell.init_collection')
    @patch('lsm.cli.shell.OpenAI')
    @patch('lsm.query.repl.print_help')
    def test_show_help_query_context(self, mock_print_help, mock_openai, mock_init_coll, mock_init_emb, mock_config):
        """Test help shows query commands when in query context."""
        mock_init_emb.return_value = Mock()
        mock_init_coll.return_value = Mock(count=Mock(return_value=10))
        mock_openai.return_value = Mock()
        mock_config.persist_dir.mkdir(parents=True, exist_ok=True)

        shell = UnifiedShell(mock_config)
        shell.switch_to_query()
        shell.show_help()

        # Should call query help
        mock_print_help.assert_called_once()
