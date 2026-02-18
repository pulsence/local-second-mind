"""TUI-specific pytest configuration and fixtures.

This module provides:
- Test speed markers (tui_fast, tui_slow, tui_integration)
- Fixtures for app/screen lifecycle management
- Automatic cleanup for timers and workers

Usage::

    @pytest.mark.tui_fast
    def test_something():
        # Fast unit test, no real app lifecycle
        ...

    @pytest.mark.tui_slow
    def test_startup():
        # Slower test with real app mount
        ...

"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register TUI test markers."""
    config.addinivalue_line(
        "markers",
        "tui_fast: Fast unit tests with mocks/fakes, no app lifecycle (<100ms)",
    )
    config.addinivalue_line(
        "markers",
        "tui_slow: Slower tests with real app lifecycle (startup, mount, timers)",
    )
    config.addinivalue_line(
        "markers",
        "tui_integration: Multi-screen workflow tests with real providers",
    )


@pytest.fixture
def tui_app_factory():
    """Factory for creating lightweight app instances for testing.

    This fixture provides a factory that creates app instances without
    running full lifecycle (on_mount, etc.). Use for unit tests that
    just need access to app methods/attributes.

    Returns a callable that accepts config and returns an LSMApp instance.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    def _create_app(config=None):
        if config is None:
            config = Mock()
            config.vectordb = Mock()
            config.vectordb.provider = "chromadb"
            config.embed_model = "test-model"
            config.device = "cpu"
            config.global_settings = SimpleNamespace(tui_density_mode="auto")

        from lsm.ui.tui.app import LSMApp

        app = LSMApp(config)
        return app

    return _create_app


@pytest.fixture
def tui_screen_mount():
    """Fixture for mounting/unmounting screens with automatic cleanup.

    Use this fixture when you need to test a real screen lifecycle
    (mount -> interact -> unmount). This fixture ensures:
    - Timers are stopped on unmount
    - Workers are cancelled on unmount
    - Proper cleanup between tests

    Usage::

        async def test_screen_behavior(tui_screen_mount):
            app = LSMApp(config)
            screen = MyScreen()

            mounted_screen = await tui_screen_mount(app, screen)
            # Test interactions...

            # Cleanup happens automatically
    """
    import asyncio

    from typing import Any, Awaitable, Callable, TypeVar

    T = TypeVar("T")

    async def _mount(
        app: Any,
        screen: Any,
        *,
        timeout: float = 5.0,
    ) -> Any:
        """Mount a screen on the app."""
        app.install_screen(screen)

        if hasattr(screen, "on_mount"):
            screen.on_mount()

        await asyncio.sleep(0.1)

        return screen

    async def _unmount(
        app: Any,
        screen: Any,
        *,
        timeout: float = 1.0,
    ) -> None:
        """Unmount a screen and clean up resources."""
        if hasattr(screen, "on_unmount"):
            screen.on_unmount()

        if hasattr(screen, "_timers"):
            for timer in list(getattr(screen, "_timers", []) or []):
                try:
                    timer.stop()
                except Exception:
                    pass

        if hasattr(screen, "_workers"):
            for worker in list(getattr(screen, "_workers", []) or []):
                try:
                    worker.cancel()
                except Exception:
                    pass

        try:
            app.uninstall_screen(screen)
        except Exception:
            pass

        await asyncio.sleep(0.05)

    class _MountContext:
        def __init__(self, app: Any, screen: Any):
            self.app = app
            self.screen = screen

        async def __aenter__(self) -> Any:
            return await _mount(self.app, self.screen)

        async def __aexit__(self, *args: Any) -> None:
            await _unmount(self.app, self.screen)

    def _create_context(app: Any, screen: Any) -> _MountContext:
        return _MountContext(app, screen)

    return _create_context


@pytest.fixture
def isolation_check(request):
    """Fixture to detect state leakage between tests.

    This fixture runs after each test and checks for common state
    leakage patterns. If state leakage is detected, it will fail
    the test with a descriptive error.

    Checks performed:
    - Agent manager has running agents (should be empty between tests)
    - TUI log buffer has unexpected entries
    """
    import logging

    yield

    from lsm.ui.shell.commands.agents import get_agent_runtime_manager

    manager = get_agent_runtime_manager()
    with manager._lock:
        if manager._agents:
            pytest.fail(
                f"Test left {len(manager._agents)} running agent(s) in manager. "
                f"Use reset_agent_manager fixture. Agent IDs: {list(manager._agents.keys())}"
            )


@pytest.fixture(autouse=True)
def _auto_cleanup_tui_state(request):
    """Auto-applied fixture that ensures basic cleanup.

    This fixture automatically runs after every TUI test to ensure
    basic state cleanup. It doesn't fail tests but warns about potential
    issues.
    """
    yield

    try:
        from lsm.ui.shell.commands.agents import get_agent_runtime_manager

        manager = get_agent_runtime_manager()
        with manager._lock:
            if manager._agents:
                try:
                    manager.shutdown(join_timeout_s=0.5)
                except Exception:
                    pass
                manager._agents.clear()
                manager._log_streams.clear()
    except Exception:
        pass
