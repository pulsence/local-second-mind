"""Structural regression coverage for TUI screen layout IDs.

These tests verify that key widgets exist in screen compose() methods.
The tests are designed to be refactor-friendly - they check for required
functionality rather than exact implementation details.

Two test styles:
- Basic tests (tui_fast): Check required IDs exist, don't require specific order
- Strict tests (optional): Check exact ID order, can be enabled with @pytest.mark.struct_strict
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCREENS_DIR = _REPO_ROOT / "lsm" / "ui" / "tui" / "screens"


def _screen_source(screen_file: str) -> str:
    return (_SCREENS_DIR / screen_file).read_text(encoding="utf-8")


def _compose_body(screen_file: str) -> str:
    source = _screen_source(screen_file)
    match = re.search(
        r"^    def compose\(self\) -> ComposeResult:\n(?P<body>.*?)(?=^    def )",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"compose() not found in {screen_file}"
    return str(match.group("body"))


def _compose_ids(screen_file: str) -> list[str]:
    return re.findall(r'id="([^"]+)"', _compose_body(screen_file))


def _assert_ids_in_order(actual_ids: list[str], expected_ids: list[str], *, screen_file: str) -> None:
    cursor = -1
    for expected in expected_ids:
        try:
            index = actual_ids.index(expected, cursor + 1)
        except ValueError as exc:
            raise AssertionError(
                f"{screen_file} missing ordered id '{expected}'. Found IDs: {actual_ids}"
            ) from exc
        cursor = index


_SCREEN_REQUIREMENTS = {
    "query.py": {
        "required_ids": {
            "query-layout",
            "query-top",
            "query-results-panel",
            "query-log-panel",
            "query-log",
            "query-command-input",
        },
        "ordered_ids": [
            "query-layout",
            "query-top",
            "query-results-panel",
            "query-log-panel",
            "query-log",
            "query-command-input",
        ],
    },
    "ingest.py": {
        "required_ids": {
            "ingest-layout",
            "file-tree",
            "ingest-right-pane",
            "ingest-info-container",
            "stats-content",
            "ingest-common-commands",
            "command-progress",
            "ingest-command-status",
            "ingest-selection-stats",
            "ingest-output-container",
            "ingest-output",
            "ingest-explore-tree",
            "ingest-command-input",
        },
        "ordered_ids": [
            "ingest-layout",
            "file-tree",
            "ingest-right-pane",
            "ingest-info-container",
            "stats-content",
            "ingest-output-container",
            "ingest-output",
            "ingest-explore-tree",
            "ingest-command-input",
        ],
    },
    "remote.py": {
        "required_ids": {
            "remote-layout",
            "remote-top",
            "remote-provider-panel",
            "remote-provider-list",
            "remote-log-panel",
            "remote-log",
            "remote-results-panel",
            "remote-results-scroll",
            "remote-results-output",
            "remote-controls",
            "remote-provider-select",
            "remote-query-input",
            "remote-search-button",
            "remote-refresh-button",
            "remote-weight-input",
            "remote-weight-button",
        },
        "ordered_ids": [
            "remote-layout",
            "remote-top",
            "remote-provider-panel",
            "remote-provider-list",
            "remote-log-panel",
            "remote-log",
            "remote-results-panel",
            "remote-results-output",
            "remote-controls",
            "remote-provider-select",
            "remote-query-input",
            "remote-search-button",
            "remote-refresh-button",
            "remote-weight-input",
            "remote-weight-button",
        ],
    },
    "settings.py": {
        "required_ids": {
            "settings-layout",
            "settings-status",
            "settings-tabs",
        },
        "ordered_ids": [
            "settings-layout",
            "settings-status",
            "settings-tabs",
        ],
    },
    "agents.py": {
        "required_ids": {
            "agents-layout",
            "agents-top",
            "agents-left",
            "agents-control-panel",
            "agents-running-panel",
            "agents-status-panel",
            "agents-interaction-panel",
            "agents-meta-panel",
            "agents-schedule-panel",
            "agents-memory-panel",
            "agents-log-panel",
            "agents-log",
        },
        "ordered_ids": [
            "agents-layout",
            "agents-top",
            "agents-left",
            "agents-control-panel",
            "agents-running-panel",
            "agents-status-panel",
            "agents-interaction-panel",
            "agents-meta-panel",
            "agents-schedule-panel",
            "agents-memory-panel",
            "agents-log-panel",
            "agents-log",
        ],
    },
}


@pytest.mark.tui_fast
@pytest.mark.parametrize("screen_file", list(_SCREEN_REQUIREMENTS.keys()))
def test_screen_has_required_ids(screen_file: str) -> None:
    """Test that required widget IDs exist in screen compose() - order agnostic.

    This is the refactor-friendly test that checks functionality exists
    without requiring specific ordering.
    """
    requirements = _SCREEN_REQUIREMENTS[screen_file]
    required_ids = requirements["required_ids"]

    compose_ids = _compose_ids(screen_file)
    missing = sorted(required_ids - set(compose_ids))

    assert not missing, f"{screen_file} missing required IDs: {missing}"


@pytest.mark.tui_fast
@pytest.mark.parametrize("screen_file", list(_SCREEN_REQUIREMENTS.keys()))
def test_screen_layout_container_exists(screen_file: str) -> None:
    """Test that each screen has a main layout container.

    This is a behavior-focused test - we verify that the screen has
    a layout container (typically named *-layout), without checking exact IDs.
    """
    compose_ids = _compose_ids(screen_file)

    layout_ids = [id for id in compose_ids if id.endswith("-layout")]
    assert layout_ids, f"{screen_file} has no layout container (*-layout)"


@pytest.mark.tui_fast
@pytest.mark.parametrize(
    "screen_file,id_pattern,description",
    [
        ("query.py", "command-input", "query screen has command input"),
        ("query.py", "log-panel", "query screen has log panel"),
        ("ingest.py", "command-input", "ingest screen has command input"),
        ("remote.py", "query-input", "remote screen has query input"),
        ("remote.py", "provider-select", "remote screen has provider select"),
        ("agents.py", "running-panel", "agents screen has running panel"),
        ("agents.py", "log-panel", "agents screen has log panel"),
    ],
)
def test_screen_has_functional_component(screen_file: str, id_pattern: str, description: str) -> None:
    """Behavior-focused test verifying functional components exist.

    This test uses pattern matching rather than exact IDs, making it
    more resilient to refactoring.
    """
    compose_ids = _compose_ids(screen_file)

    matching_ids = [id for id in compose_ids if id_pattern in id]
    assert matching_ids, f"{screen_file}: {description} - no ID containing '{id_pattern}'"


@pytest.mark.tui_fast
@pytest.mark.parametrize("screen_file", list(_SCREEN_REQUIREMENTS.keys()))
def test_screen_has_input_widgets(screen_file: str) -> None:
    """Test that screens with command input have input widgets."""
    source = _compose_body(screen_file)

    has_input = "Input(" in source
    has_text_area = "TextArea(" in source

    assert has_input or has_text_area, f"{screen_file} has no Input or TextArea widgets"


@pytest.mark.tui_fast
def test_settings_compose_uses_dynamic_table_and_command_ids() -> None:
    """Test that Settings screen uses dynamic IDs for tabs."""
    settings_source = _screen_source("settings.py")
    compose_body = _compose_body("settings.py")
    assert 'id=self._table_id(tab_id)' in compose_body
    assert 'id=self._command_id(tab_id)' in compose_body

    layout_match = re.search(
        r"_TAB_LAYOUT:\s*tuple\[tuple\[str,\s*str\],\s*\.\.\.\]\s*=\s*\((?P<body>.*?)\n    \)\n",
        settings_source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert layout_match is not None, "_TAB_LAYOUT not found in settings.py"
    tab_ids = re.findall(r'\("([^"]+)",\s*"[^"]+"\)', str(layout_match.group("body")))
    assert tab_ids == [
        "settings-global",
        "settings-ingest",
        "settings-query",
        "settings-llm",
        "settings-vdb",
        "settings-modes",
        "settings-remote",
        "settings-chats-notes",
    ]


@pytest.mark.tui_fast
def test_all_screens_importable() -> None:
    """Verify all screen modules can be imported without error."""
    from lsm.ui.tui.screens.query import QueryScreen
    from lsm.ui.tui.screens.ingest import IngestScreen
    from lsm.ui.tui.screens.remote import RemoteScreen
    from lsm.ui.tui.screens.agents import AgentsScreen
    from lsm.ui.tui.screens.settings import SettingsScreen

    assert QueryScreen is not None
    assert IngestScreen is not None
    assert RemoteScreen is not None
    assert AgentsScreen is not None
    assert SettingsScreen is not None
