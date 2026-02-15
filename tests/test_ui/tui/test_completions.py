"""Tests for TUI autocomplete functionality."""

from __future__ import annotations

import textwrap

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
        assert "/memory" in commands
        assert "/ui" in commands
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

    def test_memory_command_suggests_subcommands(self):
        """Memory command should suggest memory subcommands."""
        completions = get_completions("/memory ", "query")
        assert "candidates" in completions
        assert "promote" in completions
        assert "reject" in completions
        assert "ttl" in completions

    def test_ui_command_suggests_density_subcommand(self):
        """UI command should suggest available UI subcommands."""
        completions = get_completions("/ui ", "query")
        assert "density" in completions

    def test_ui_density_suggests_modes(self):
        """UI density should suggest valid density modes."""
        completions = get_completions("/ui density ", "query")
        assert "auto" in completions
        assert "compact" in completions
        assert "comfortable" in completions

    def test_agent_command_suggests_schedule_subcommands(self):
        """Agent command should suggest scheduler subcommands."""
        completions = get_completions("/agent schedule ", "query")
        assert "add" in completions
        assert "list" in completions
        assert "enable" in completions
        assert "disable" in completions
        assert "remove" in completions
        assert "status" in completions

    def test_agent_command_suggests_interaction_subcommands(self):
        """Agent command should suggest interaction and selection subcommands."""
        completions = get_completions("/agent ", "query")
        assert "list" in completions
        assert "interact" in completions
        assert "approve" in completions
        assert "deny" in completions
        assert "approve-session" in completions
        assert "reply" in completions
        assert "queue" in completions
        assert "select" in completions

    def test_agent_queue_suggests_target_or_message(self):
        """Agent queue should prompt for either target id or message."""
        completions = get_completions("/agent queue ", "query")
        assert "<agent_id>" in completions
        assert "<message>" in completions

    def test_agent_schedule_add_suggests_flags(self):
        """Agent schedule add should suggest supported optional flags."""
        completions = get_completions("/agent schedule add research daily ", "query")
        assert "--params" in completions
        assert "--concurrency_policy" in completions
        assert "--confirmation_mode" in completions

    def test_agent_command_suggests_meta_subcommands(self):
        """Agent meta command should suggest meta subcommands."""
        completions = get_completions("/agent meta ", "query")
        assert "start" in completions
        assert "status" in completions
        assert "log" in completions

    def test_agent_meta_start_suggests_goal_placeholder(self):
        """Agent meta start should prompt for goal argument."""
        completions = get_completions("/agent meta start ", "query")
        assert "<goal>" in completions

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
        assert "/memory" in help_text
        assert "/help" in help_text

    def test_query_help_output_snapshot(self):
        """Query help output should keep stable section ordering and formatting."""
        expected = textwrap.dedent(
            """\
            Available commands (query):

              Navigation:
                /exit                Exit
                /quit                Exit the application
                /help                Show help

              Information:
                /debug               Show retrieval diagnostics
                /costs               Show session cost summary

              Exploration:
                /show                Show cited chunk (requires: S# e.g., S1)

              Operations:
                /load                Pin document for context (requires: path)

              Query:
                /mode                Show or switch query mode
                /show                Show cited chunk (requires: S# e.g., S1)
                /expand              Expand citation (requires: S# e.g., S1)
                /open                Open source file (requires: S#)
                /note                Save last query as note (optional: name)
                /notes               Alias for /note

              Agents:
                /agent               Run or inspect agent workflows
                /memory              Manage agent memory candidates

              UI:
                /ui                  Inspect or change TUI UI settings

              Providers:
                /providers           List available LLM providers
                /provider-status     Show provider health
                /model               Show or set current model
                /models              List available models (optional: provider)
                /vectordb-providers  List vector DB providers
                /vectordb-status     Show vector DB status
                /remote-providers    List remote source providers
                /remote-search       Test remote provider (requires: provider query)
                /remote-search-all   Search all providers (requires: query)

              Export:
                /export-citations    Export citations (optional: format path)
                /budget              Set session budget (requires: set amount)
                /cost-estimate       Estimate query cost (requires: query)

              Filters:
                /set                 Set session filter (requires: filter value)
                /clear               Clear session filter
                /context             Show or set context anchors
            """
        ).strip()
        assert format_command_help("query").strip() == expected

    def test_ingest_help_output_snapshot(self):
        """Ingest help output should keep stable section ordering and formatting."""
        expected = textwrap.dedent(
            """\
            Available commands (ingest):

              Navigation:
                /exit                Exit
                /quit                Exit the application
                /help                Show help

              Information:
                /info                Show collection information
                /stats               Show detailed statistics

              Exploration:
                /explore             Browse indexed files (optional: path filter)
                /show                Show chunks for a file (requires: path)
                /search              Search metadata (requires: query)
                /tags                Show all tags in collection

              Operations:
                /build               Run ingest pipeline (optional: --force)
                /tag                 Run AI tagging (optional: --max N)
                /wipe                Clear collection (requires confirmation)

              Query:
                /show                Show chunks for a file (requires: path)

              Providers:
                /vectordb-providers  List available vector DB providers
                /vectordb-status     Show vector DB provider status
            """
        ).strip()
        assert format_command_help("ingest").strip() == expected


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
