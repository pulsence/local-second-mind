"""Agents screen layout regression tests for clipping on default terminals."""

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _selector_bodies(css: str, selector: str) -> list[str]:
    pattern = rf"(?ms)^\s*{re.escape(selector)}\s*\{{(.*?)^\s*\}}"
    return re.findall(pattern, css)


def _assert_selector_has(css: str, selector: str, expected_line: str) -> None:
    bodies = _selector_bodies(css, selector)
    assert bodies, f"Selector not found: {selector}"
    assert any(expected_line in body for body in bodies), (
        f"Expected '{expected_line}' in selector '{selector}', but it was not found."
    )


def test_agents_left_column_is_scrollable_to_prevent_panel_clipping() -> None:
    """Agents left column should scroll vertically when panels exceed viewport height."""
    css = _read(_repo_root() / "lsm" / "ui" / "tui" / "styles.tcss")
    _assert_selector_has(css, "#agents-left", "overflow-y: auto;")
    _assert_selector_has(css, "#agents-left", "overflow-x: hidden;")


def test_agents_panels_use_auto_height_not_fractional_splitting() -> None:
    """Control/status/meta/schedule/memory panels should size to content."""
    css = _read(_repo_root() / "lsm" / "ui" / "tui" / "styles.tcss")
    for selector in (
        "#agents-control-panel",
        "#agents-status-panel",
        "#agents-meta-panel",
        "#agents-schedule-panel",
        "#agents-memory-panel",
    ):
        _assert_selector_has(css, selector, "height: auto;")


def test_agents_compose_uses_multiline_button_groups() -> None:
    """Compose should split button groups into rows for narrow terminal widths."""
    source = _read(_repo_root() / "lsm" / "ui" / "tui" / "screens" / "agents.py")
    assert 'with Vertical(id="agents-buttons")' in source
    assert 'with Vertical(id="agents-schedule-buttons")' in source
    assert 'with Vertical(id="agents-memory-buttons")' in source
    assert 'classes="agents-button-row"' in source

