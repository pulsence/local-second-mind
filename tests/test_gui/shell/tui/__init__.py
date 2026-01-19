"""Deprecated TUI test package (moved to tests/test_gui/tui)."""

import pytest

pytest.skip(
    "TUI tests moved to tests/test_gui/tui; shell package no longer contains TUI tests.",
    allow_module_level=True,
)
