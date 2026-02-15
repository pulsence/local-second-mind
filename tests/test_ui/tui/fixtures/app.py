"""Reusable fake app factory for TUI screen tests.

Provides ``create_fake_app()`` which returns a ``SimpleNamespace`` with the
minimal attributes that screens expect from ``self.app``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional


def create_fake_app(
    *,
    config: Any = None,
    current_context: str = "query",
    query_state: Any = None,
) -> SimpleNamespace:
    """Create a minimal app double for screen tests.

    Args:
        config: App config object. If None a basic stub is provided.
        current_context: Active TUI context tab name.
        query_state: Optional query state object.

    Returns:
        SimpleNamespace mimicking ``LSMApp`` for test purposes.
    """
    if config is None:
        config = SimpleNamespace(
            query=SimpleNamespace(mode="grounded"),
            llm=SimpleNamespace(
                get_query_config=lambda: SimpleNamespace(
                    model="gpt-test",
                    provider="openai",
                ),
            ),
        )

    return SimpleNamespace(
        config=config,
        current_context=current_context,
        query_state=query_state,
        _tui_log_buffer=[],
    )
