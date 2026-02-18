"""Global state reset fixtures for TUI tests.

Provides fixtures to reset singletons and global state between tests to prevent
state leakage that causes flaky tests.

Usage::

    def test_something(reset_agent_manager, reset_tui_logging):
        # Agent manager is now clean for this test
        ...

"""

from __future__ import annotations

import pytest
from typing import Generator


@pytest.fixture
def reset_agent_manager() -> Generator[None, None, None]:
    """Reset the singleton agent runtime manager between tests.

    This fixture:
    - Shuts down any running agents (cancels pending interactions, joins threads)
    - Clears all active agent state
    - Clears completed runs history
    - Clears log streams
    - Clears session tool approvals
    - Resets selected agent ID
    """
    from lsm.ui.shell.commands.agents import get_agent_runtime_manager

    manager = get_agent_runtime_manager()

    try:
        manager.shutdown(join_timeout_s=1.0)
    except Exception:
        pass

    with manager._lock:
        manager._agents.clear()
        manager._completed_runs.clear()
        manager._completed_order.clear()
        manager._log_streams.clear()
        manager._selected_agent_id = None
        manager._session_tool_approvals.clear()

    yield

    try:
        manager.shutdown(join_timeout_s=1.0)
    except Exception:
        pass

    with manager._lock:
        manager._agents.clear()
        manager._completed_runs.clear()
        manager._completed_order.clear()
        manager._log_streams.clear()
        manager._selected_agent_id = None
        manager._session_tool_approvals.clear()


@pytest.fixture
def reset_tui_logging() -> Generator[None, None, None]:
    """Reset TUI logging state between tests.

    This fixture:
    - Clears the _tui_log_buffer deque
    - Does NOT remove handlers (handlers are app-instance specific)
    """
    import logging

    yield

    from lsm.ui.tui import app as tui_app_module

    if hasattr(tui_app_module, '_app') and tui_app_module._app is not None:
        app = tui_app_module._app
        if hasattr(app, '_tui_log_buffer'):
            app._tui_log_buffer.clear()


@pytest.fixture
def reset_app_state(reset_agent_manager, reset_tui_logging) -> Generator[None, None, None]:
    """Reset all TUI global state between tests.

    Combines reset_agent_manager and reset_tui_logging fixtures.
    Use this when a test needs a completely clean slate.
    """
    yield
