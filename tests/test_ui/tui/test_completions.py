"""Tests for TUI autocomplete functionality."""

from __future__ import annotations

import pytest

from lsm.ui.tui.completions import (
    get_completions,
    get_commands,
    format_command_help,
    create_completer,
    GLOBAL_COMMANDS,
    INGEST_COMMANDS,
    QUERY_COMMANDS,
    MODE_VALUES,
)


class TestGetCommands:
    """Tests for get_commands function."""

    def test_global_context_returns_global_commands(self):
        """Global context should return only global commands."""
        commands = get_commands("global")
        assert "/exit" in commands
        assert "/quit" in commands
        assert "/help" in commands

    def test_ingest_context_includes_ingest_commands(self):
        """Ingest context should include ingest-specific commands."""
        commands = get_commands("ingest")
        assert "/build" in commands
        assert "/tag" in commands
        assert "/stats" in commands
        assert "/explore" in commands
        assert "/wipe" in commands
        # Also includes global
        assert "/exit" in commands

    def test_query_context_includes_query_commands(self):
        """Query context should include query-specific commands."""
        commands = get_commands("query")
        assert "/mode" in commands
        assert "/show" in commands
        assert "/expand" in commands
        assert "/costs" in commands
        assert "/debug" in commands
        # Also includes global
        assert "/exit" in commands


class TestGetCompletions:
    """Tests for get_completions function."""

    def test_empty_input_returns_all_commands(self):
        """Empty input should return all commands for context."""
        completions = get_completions("", "query")
        assert len(completions) > 0
        assert all(c.startswith("/") for c in completions)

    def test_partial_command_returns_matches(self):
        """Partial command should return matching commands."""
        completions = get_completions("/he", "query")
        assert "/help" in completions

    def test_mode_command_suggests_modes(self):
        """Mode command should suggest mode values."""
        completions = get_completions("/mode ", "query")
        assert "grounded" in completions
        assert "insight" in completions
        assert "hybrid" in completions
        assert "set" in completions

    def test_mode_command_filters_modes(self):
        """Mode command with partial arg should filter modes."""
        completions = get_completions("/mode gr", "query")
        assert "grounded" in completions
        assert "insight" not in completions

    def test_mode_set_suggests_llm_cache_setting(self):
        """Mode set command should include llm_cache setting."""
        completions = get_completions("/mode set ", "query")
        assert "set llm_cache" in completions

    def test_build_command_suggests_options(self):
        """Build command should suggest --force option."""
        completions = get_completions("/build ", "ingest")
        assert "--force" in completions

    def test_build_command_filters_options(self):
        """Build command with partial arg should filter options."""
        completions = get_completions("/build --f", "ingest")
        assert "--force" in completions

    def test_tag_command_suggests_options(self):
        """Tag command should suggest --max option."""
        completions = get_completions("/tag ", "ingest")
        assert "--max" in completions

    def test_show_command_with_candidates_suggests_citations(self):
        """Show command with candidates should suggest citations."""
        candidates = ["candidate1", "candidate2", "candidate3"]
        completions = get_completions("/show S", "query", candidates)
        assert "S1" in completions
        assert "S2" in completions
        assert "S3" in completions

    def test_citation_completion_filters(self):
        """Citation completion should filter by prefix."""
        candidates = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"]
        completions = get_completions("/show S1", "query", candidates)
        assert "S1" in completions
        assert "S10" in completions
        assert "S11" in completions
        assert "S2" not in completions

    def test_export_citations_suggests_formats(self):
        """Export citations command should suggest formats."""
        completions = get_completions("/export-citations ", "query")
        assert "bibtex" in completions
        assert "zotero" in completions

    def test_remote_provider_suggests_actions(self):
        """Remote provider command should suggest current command name."""
        completions = get_completions("/remote-provider ", "query")
        assert "/remote-providers" in completions

    def test_unknown_context_returns_global_only(self):
        """Unknown context should return global commands only."""
        # Using a context that doesn't match ingest or query
        completions = get_completions("", "global")
        assert "/exit" in completions
        # Should not have ingest or query specific commands
        assert "/build" not in completions
        assert "/mode" not in completions


class TestFormatCommandHelp:
    """Tests for format_command_help function."""

    def test_returns_string(self):
        """Should return a string."""
        help_text = format_command_help("query")
        assert isinstance(help_text, str)
        assert len(help_text) > 0

    def test_includes_context_name(self):
        """Help text should include context name."""
        help_text = format_command_help("ingest")
        assert "ingest" in help_text.lower()

    def test_includes_commands(self):
        """Help text should include command names."""
        help_text = format_command_help("query")
        assert "/mode" in help_text
        assert "/help" in help_text


class TestCreateCompleter:
    """Tests for create_completer factory function."""

    def test_creates_callable(self):
        """Should create a callable."""
        completer = create_completer("query")
        assert callable(completer)

    def test_completer_returns_completions(self):
        """Created completer should return completions."""
        completer = create_completer("query")
        completions = completer("/he")
        assert "/help" in completions

    def test_completer_uses_candidates_getter(self):
        """Completer should use candidates getter for citations."""
        candidates = ["c1", "c2", "c3"]
        completer = create_completer("query", lambda: candidates)
        completions = completer("/show S")
        assert "S1" in completions
        assert "S2" in completions
        assert "S3" in completions


class TestCommandDefinitions:
    """Tests for command definitions."""

    def test_global_commands_have_descriptions(self):
        """All global commands should have descriptions."""
        for cmd, desc in GLOBAL_COMMANDS.items():
            assert cmd.startswith("/"), f"Command {cmd} should start with /"
            assert len(desc) > 0, f"Command {cmd} should have description"

    def test_ingest_commands_have_descriptions(self):
        """All ingest commands should have descriptions."""
        for cmd, desc in INGEST_COMMANDS.items():
            assert cmd.startswith("/"), f"Command {cmd} should start with /"
            assert len(desc) > 0, f"Command {cmd} should have description"

    def test_query_commands_have_descriptions(self):
        """All query commands should have descriptions."""
        for cmd, desc in QUERY_COMMANDS.items():
            assert cmd.startswith("/"), f"Command {cmd} should start with /"
            assert len(desc) > 0, f"Command {cmd} should have description"

    def test_mode_values_are_valid(self):
        """Mode values should be the expected modes."""
        assert "grounded" in MODE_VALUES
        assert "insight" in MODE_VALUES
        assert "hybrid" in MODE_VALUES
