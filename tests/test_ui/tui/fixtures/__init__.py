"""Shared TUI test fixtures and harness utilities.

This package provides reusable test doubles and fixture factories for TUI
screen tests. Import from here instead of duplicating fakes across test files.

Test double naming conventions:

- ``_FakeApp`` - Minimal app double with config, state, and lifecycle stubs
- ``_TestableScreen`` - Screen subclass overriding ``app`` and ``query_one``
- ``_Static`` / ``_Input`` / ``_Select`` - Lightweight widget doubles
- ``_RichLog`` - Collects ``write()`` calls for assertion (agents log)
- ``_TextArea`` - Collects ``insert()`` calls for assertion (query/remote log)

Usage::

    from tests.test_ui.tui.fixtures import create_fake_app, FakeWidget
    from tests.test_ui.tui.fixtures import reset_agent_manager, reset_tui_logging

"""

from __future__ import annotations

from tests.test_ui.tui.fixtures.widgets import (
    FakeStatic,
    FakeInput,
    FakeSelect,
    FakeRichLog,
    FakeTextArea,
    FakeButton,
)
from tests.test_ui.tui.fixtures.app import create_fake_app, create_startup_mock_config
from tests.test_ui.tui.fixtures.global_state import (
    reset_agent_manager,
    reset_tui_logging,
    reset_app_state,
)

__all__ = [
    "FakeStatic",
    "FakeInput",
    "FakeSelect",
    "FakeRichLog",
    "FakeTextArea",
    "FakeButton",
    "create_fake_app",
    "create_startup_mock_config",
    "reset_agent_manager",
    "reset_tui_logging",
    "reset_app_state",
]
