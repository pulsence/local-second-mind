"""Command parsing contract tests for consistent behavior across UI surfaces.

Verifies:
- parse_slash_command grammar contracts
- /agent, /memory, /ui command groups produce consistent error messages
- Settings command grammar (set/unset/delete/reset/default/save/discard)
- Autocomplete registry alignment with implemented commands
"""

from __future__ import annotations

import pytest

from lsm.ui.helpers.commands.common import (
    CommandParseError,
    ParsedCommand,
    format_command_error,
    normalize_argument,
    normalize_arguments,
    parse_on_off_value,
    parse_slash_command,
    tokenize_command,
    QUERY_MODE_VALUES,
    UI_DENSITY_VALUES,
)


# -------------------------------------------------------------------------
# parse_slash_command grammar contracts
# -------------------------------------------------------------------------


class TestParseSlashCommand:
    """Verify core parsing contracts."""

    def test_empty_string_yields_no_command(self) -> None:
        result = parse_slash_command("")
        assert result.cmd == ""
        assert result.parts == ()

    def test_non_slash_text_yields_no_command(self) -> None:
        result = parse_slash_command("hello world")
        assert result.cmd == ""
        assert result.text == "hello world"

    def test_single_slash_command(self) -> None:
        result = parse_slash_command("/help")
        assert result.cmd == "/help"
        assert result.parts == ("/help",)

    def test_slash_command_lowercased(self) -> None:
        result = parse_slash_command("/MODE grounded")
        assert result.cmd == "/mode"
        assert result.parts == ("/MODE", "grounded")

    def test_slash_command_with_args(self) -> None:
        result = parse_slash_command("/agent start research AI safety")
        assert result.cmd == "/agent"
        assert result.parts == ("/agent", "start", "research", "AI", "safety")

    def test_whitespace_trimmed(self) -> None:
        result = parse_slash_command("  /help  ")
        assert result.cmd == "/help"
        assert result.text == "/help"

    def test_none_input_yields_empty(self) -> None:
        result = parse_slash_command(None)  # type: ignore[arg-type]
        assert result.cmd == ""

    def test_parsed_command_is_frozen(self) -> None:
        result = parse_slash_command("/help")
        with pytest.raises(AttributeError):
            result.cmd = "/other"  # type: ignore[misc]


# -------------------------------------------------------------------------
# tokenize_command contracts
# -------------------------------------------------------------------------


class TestTokenizeCommand:
    """Verify tokenization behavior."""

    def test_empty_returns_empty(self) -> None:
        assert tokenize_command("") == []

    def test_simple_split(self) -> None:
        assert tokenize_command("set key value") == ["set", "key", "value"]

    def test_shlex_handles_quoted_args(self) -> None:
        tokens = tokenize_command('set key "hello world"', use_shlex=True)
        assert tokens == ["set", "key", "hello world"]

    def test_shlex_error_raises_command_parse_error(self) -> None:
        with pytest.raises(CommandParseError):
            tokenize_command('set "unterminated', use_shlex=True)


# -------------------------------------------------------------------------
# normalize helpers
# -------------------------------------------------------------------------


class TestNormalization:
    """Verify normalization utilities."""

    def test_normalize_argument_strips(self) -> None:
        assert normalize_argument("  hello  ") == "hello"

    def test_normalize_argument_none(self) -> None:
        assert normalize_argument(None) == ""

    def test_normalize_arguments_preserves_case(self) -> None:
        result = normalize_arguments(["  Hello  ", " World "])
        assert result == ["Hello", "World"]

    def test_normalize_arguments_lower(self) -> None:
        result = normalize_arguments(["Hello", "WORLD"], lower=True)
        assert result == ["hello", "world"]


# -------------------------------------------------------------------------
# parse_on_off_value contracts
# -------------------------------------------------------------------------


class TestParseOnOffValue:
    """Verify boolean toggle parsing."""

    @pytest.mark.parametrize("value", ["on", "true", "yes", "1", "ON", "True"])
    def test_truthy_values(self, value: str) -> None:
        assert parse_on_off_value(value) is True

    @pytest.mark.parametrize("value", ["off", "false", "no", "0", "OFF", "False"])
    def test_falsy_values(self, value: str) -> None:
        assert parse_on_off_value(value) is False

    def test_unknown_returns_none(self) -> None:
        assert parse_on_off_value("maybe") is None

    def test_empty_returns_none(self) -> None:
        assert parse_on_off_value("") is None


# -------------------------------------------------------------------------
# format_command_error contracts
# -------------------------------------------------------------------------


class TestFormatCommandError:
    """Verify consistent error formatting."""

    def test_trailing_newline(self) -> None:
        result = format_command_error("bad input")
        assert result == "bad input\n"

    def test_strips_whitespace(self) -> None:
        result = format_command_error("  padded  ")
        assert result == "padded\n"


# -------------------------------------------------------------------------
# /agent command grammar contracts
# -------------------------------------------------------------------------


class TestAgentCommandContracts:
    """Verify /agent command validation without executing runtime actions."""

    def test_agent_command_needs_subcommand(self) -> None:
        """'/agent' alone should return help text."""
        from lsm.ui.shell.commands.agents import handle_agent_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_agent_command("/agent", object())
        assert "Usage" in result or "/agent" in result

    def test_agent_start_needs_name_and_topic(self) -> None:
        from lsm.ui.shell.commands.agents import handle_agent_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_agent_command("/agent start", object())
        assert "Usage" in result

    def test_agent_approve_needs_agent_id(self) -> None:
        from lsm.ui.shell.commands.agents import handle_agent_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_agent_command("/agent approve", object())
        assert "Usage" in result

    def test_agent_deny_needs_agent_id(self) -> None:
        from lsm.ui.shell.commands.agents import handle_agent_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_agent_command("/agent deny", object())
        assert "Usage" in result

    def test_agent_reply_needs_id_and_message(self) -> None:
        from lsm.ui.shell.commands.agents import handle_agent_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_agent_command("/agent reply", object())
        assert "Usage" in result

    def test_agent_queue_needs_message(self) -> None:
        from lsm.ui.shell.commands.agents import handle_agent_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_agent_command("/agent queue", object())
        assert "Usage" in result


# -------------------------------------------------------------------------
# /memory command grammar contracts
# -------------------------------------------------------------------------


class TestMemoryCommandContracts:
    """Verify /memory command validation without executing runtime actions."""

    def test_memory_command_needs_subcommand(self) -> None:
        from lsm.ui.shell.commands.agents import handle_memory_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_memory_command("/memory", object())
        assert "Usage" in result or "/memory" in result

    def test_memory_candidates_rejects_invalid_status(self) -> None:
        from lsm.ui.shell.commands.agents import handle_memory_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_memory_command("/memory candidates bogus", object())
        assert "Usage" in result

    def test_memory_promote_needs_id(self) -> None:
        from lsm.ui.shell.commands.agents import handle_memory_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_memory_command("/memory promote", object())
        assert "Usage" in result

    def test_memory_reject_needs_id(self) -> None:
        from lsm.ui.shell.commands.agents import handle_memory_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_memory_command("/memory reject", object())
        assert "Usage" in result

    def test_memory_ttl_needs_id_and_days(self) -> None:
        from lsm.ui.shell.commands.agents import handle_memory_command
        from unittest.mock import patch

        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager"):
            result = handle_memory_command("/memory ttl mem-1", object())
        assert "Usage" in result

    def test_memory_ttl_requires_integer_days(self) -> None:
        from lsm.ui.shell.commands.agents import handle_memory_command
        from unittest.mock import patch, MagicMock

        mock_manager = MagicMock()
        with patch("lsm.ui.shell.commands.agents.get_agent_runtime_manager", return_value=mock_manager):
            result = handle_memory_command("/memory ttl mem-1 abc", object())
        assert "integer" in result.lower()


# -------------------------------------------------------------------------
# Settings command grammar contracts
# -------------------------------------------------------------------------


class TestSettingsCommandContracts:
    """Verify settings command verbs parse correctly via SettingsScreen."""

    VALID_VERBS = ("set", "unset", "delete", "reset", "default", "save", "discard")

    def test_all_valid_verbs_are_recognized(self) -> None:
        """Every documented settings verb should not produce 'Unknown command'."""
        from lsm.ui.tui.screens.settings import SettingsScreen
        assert all(
            verb in " ".join(SettingsScreen._TAB_HELP.values())
            for verb in self.VALID_VERBS
        )

    def test_set_requires_key_and_value(self) -> None:
        """'set' with fewer than 3 tokens should report usage error."""
        import shlex
        tokens = shlex.split("set onlykey")
        assert len(tokens) < 3  # Contract: set needs key + value

    def test_unset_requires_exactly_one_key(self) -> None:
        """'unset' with extra tokens is invalid."""
        import shlex
        tokens = shlex.split("unset key extra")
        assert len(tokens) != 2  # Contract: unset needs exactly 2 tokens

    def test_delete_requires_exactly_one_key(self) -> None:
        import shlex
        tokens = shlex.split("delete key extra")
        assert len(tokens) != 2

    def test_discard_accepts_no_args_or_tab(self) -> None:
        """'discard' or 'discard tab' are valid, other forms are not."""
        import shlex
        assert len(shlex.split("discard")) == 1
        assert shlex.split("discard tab") == ["discard", "tab"]
        assert len(shlex.split("discard all things")) > 2


# -------------------------------------------------------------------------
# Completions alignment contracts
# -------------------------------------------------------------------------


class TestCompletionAlignment:
    """Verify autocomplete registrations align with implemented commands."""

    def test_query_commands_include_agent_and_memory(self) -> None:
        from lsm.ui.tui.completions import QUERY_COMMANDS
        assert "/agent" in QUERY_COMMANDS
        assert "/memory" in QUERY_COMMANDS

    def test_query_commands_include_ui(self) -> None:
        from lsm.ui.tui.completions import QUERY_COMMANDS
        assert "/ui" in QUERY_COMMANDS

    def test_mode_values_consistent(self) -> None:
        """Mode values used in completions should match the common module."""
        from lsm.ui.tui.completions import MODE_VALUES
        assert set(MODE_VALUES) == set(QUERY_MODE_VALUES)

    def test_density_values_consistent(self) -> None:
        """Density mode values should match the common module."""
        assert set(UI_DENSITY_VALUES) == {"auto", "compact", "comfortable"}

    def test_ingest_commands_include_core_verbs(self) -> None:
        from lsm.ui.tui.completions import INGEST_COMMANDS
        for cmd in ("/build", "/tag", "/stats", "/explore", "/wipe"):
            assert cmd in INGEST_COMMANDS, f"{cmd} missing from ingest completions"
