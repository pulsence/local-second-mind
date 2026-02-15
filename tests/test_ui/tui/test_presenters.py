"""Tests for TUI presenter modules.

Verifies that extracted formatting functions produce correct output
independently of the TUI widget tree.
"""

from __future__ import annotations

import pytest
from rich.text import Text

from lsm.ui.tui.presenters.agents import (
    format_duration,
    format_interval_label,
    format_stream_log_entry,
    format_stream_drop_notice,
    stream_prefix,
    summarize_stream_args,
    summarize_stream_content,
)


# -------------------------------------------------------------------------
# Agent log formatting
# -------------------------------------------------------------------------


class TestFormatDuration:
    def test_zero(self) -> None:
        assert format_duration(0) == "00:00"

    def test_seconds_only(self) -> None:
        assert format_duration(45) == "00:45"

    def test_minutes_and_seconds(self) -> None:
        assert format_duration(125) == "02:05"

    def test_hours(self) -> None:
        assert format_duration(3661) == "01:01:01"

    def test_negative_clamps_to_zero(self) -> None:
        assert format_duration(-5) == "00:00"


class TestFormatIntervalLabel:
    def test_basic(self) -> None:
        assert format_interval_label(60.0) == "60.0s"

    def test_fractional(self) -> None:
        assert format_interval_label(2.5) == "2.5s"


class TestStreamPrefix:
    def test_llm(self) -> None:
        label, style = stream_prefix("llm")
        assert label == "LLM"
        assert "cyan" in style

    def test_tool(self) -> None:
        label, style = stream_prefix("tool")
        assert label == "TOOL"

    def test_user(self) -> None:
        label, style = stream_prefix("user")
        assert label == "USER"

    def test_agent(self) -> None:
        label, style = stream_prefix("agent")
        assert label == "AGENT"

    def test_unknown(self) -> None:
        label, _ = stream_prefix("other")
        assert label == "EVENT"

    def test_case_insensitive(self) -> None:
        label, _ = stream_prefix("LLM")
        assert label == "LLM"


class TestSummarizeStreamArgs:
    def test_none(self) -> None:
        assert summarize_stream_args(None) == ""

    def test_empty_string(self) -> None:
        assert summarize_stream_args("") == ""

    def test_empty_dict(self) -> None:
        assert summarize_stream_args({}) == ""

    def test_short_string(self) -> None:
        assert summarize_stream_args("hello") == "hello"

    def test_dict(self) -> None:
        result = summarize_stream_args({"key": "val"})
        assert "key" in result and "val" in result

    def test_long_truncated(self) -> None:
        result = summarize_stream_args("x" * 200)
        assert len(result) <= 103
        assert result.endswith("...")


class TestSummarizeStreamContent:
    def test_empty(self) -> None:
        assert summarize_stream_content("") == "(empty)"

    def test_short(self) -> None:
        assert summarize_stream_content("hello") == "hello"

    def test_long_truncated(self) -> None:
        result = summarize_stream_content("x" * 300)
        assert len(result) <= 183
        assert result.endswith("...")

    def test_newlines_stripped(self) -> None:
        result = summarize_stream_content("line1\nline2")
        assert "\n" not in result


class TestFormatStreamLogEntry:
    def test_tool_entry(self) -> None:
        row = {
            "actor": "tool",
            "action": "read_file",
            "action_arguments": "/tmp/test.txt",
            "content": "file contents",
        }
        result = format_stream_log_entry(row)
        assert isinstance(result, Text)
        plain = result.plain
        assert "[TOOL]" in plain
        assert "read_file" in plain

    def test_llm_entry(self) -> None:
        row = {"actor": "llm", "content": "thinking about it", "action": ""}
        result = format_stream_log_entry(row)
        assert "[LLM]" in result.plain

    def test_user_entry(self) -> None:
        row = {"actor": "user", "content": "approved"}
        result = format_stream_log_entry(row)
        assert "[USER]" in result.plain


class TestFormatStreamDropNotice:
    def test_singular(self) -> None:
        result = format_stream_drop_notice(1)
        assert "1 log entry" in result.plain

    def test_plural(self) -> None:
        result = format_stream_drop_notice(5)
        assert "5 log entries" in result.plain
