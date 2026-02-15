"""Reusable fake app factory for TUI screen tests.

Provides ``create_fake_app()`` which returns a ``SimpleNamespace`` with the
minimal attributes that screens expect from ``self.app``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock


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


def create_startup_mock_config() -> Mock:
    """Create a Mock config with all attributes accessed by ``LSMApp.__init__``.

    Returns:
        Mock mimicking ``LSMConfig`` for startup tests.
    """
    cfg = Mock()
    cfg.vectordb = Mock()
    cfg.vectordb.provider = "chromadb"
    cfg.embed_model = "test-model"
    cfg.device = "cpu"
    cfg.collection = "test"
    cfg.persist_dir = "/tmp/test"
    cfg.query = Mock()
    cfg.query.mode = "grounded"
    cfg.llm = Mock()
    cfg.llm.get_query_config = Mock(
        return_value=Mock(model="gpt-test"),
    )
    cfg.global_settings = SimpleNamespace(tui_density_mode="auto")
    return cfg
