"""Structural regression coverage for TUI screen layout IDs."""

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


@pytest.mark.parametrize(
    ("screen_file", "required_ids", "ordered_ids"),
    [
        (
            "query.py",
            {
                "query-layout",
                "query-top",
                "query-results-panel",
                "query-log-panel",
                "query-log",
                "query-command-input",
            },
            [
                "query-layout",
                "query-top",
                "query-results-panel",
                "query-log-panel",
                "query-log",
                "query-command-input",
            ],
        ),
        (
            "ingest.py",
            {
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
            [
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
        ),
        (
            "remote.py",
            {
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
            [
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
        ),
        (
            "settings.py",
            {
                "settings-layout",
                "settings-status",
                "settings-tabs",
            },
            [
                "settings-layout",
                "settings-status",
                "settings-tabs",
            ],
        ),
        (
            "agents.py",
            {
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
            [
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
        ),
    ],
)
def test_screen_layout_ids_are_stable(
    screen_file: str,
    required_ids: set[str],
    ordered_ids: list[str],
) -> None:
    compose_ids = _compose_ids(screen_file)
    missing = sorted(required_ids - set(compose_ids))
    assert not missing, f"{screen_file} missing required IDs: {missing}"
    _assert_ids_in_order(compose_ids, ordered_ids, screen_file=screen_file)


def test_settings_compose_uses_dynamic_table_and_command_ids() -> None:
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
