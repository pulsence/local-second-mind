"""Compact layout regression tests for TUI CSS sizing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Skip tests if textual is not available.
pytest.importorskip("textual")


def _styles_text() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    style_path = repo_root / "lsm" / "ui" / "tui" / "styles.tcss"
    return style_path.read_text(encoding="utf-8")


def _selector_bodies(css: str, selector: str) -> list[str]:
    pattern = rf"(?ms)^\s*{re.escape(selector)}\s*\{{(.*?)^\s*\}}"
    return re.findall(pattern, css)


def _assert_selector_has(css: str, selector: str, expected_line: str) -> None:
    bodies = _selector_bodies(css, selector)
    assert bodies, f"Selector not found: {selector}"
    assert any(expected_line in body for body in bodies), (
        f"Expected '{expected_line}' in selector '{selector}', but it was not found."
    )


def test_compact_tab_and_pane_sizes() -> None:
    """Tab controls should use compact heights and padding."""
    css = _styles_text()
    _assert_selector_has(css, "ContentTabs", "height: 2;")
    _assert_selector_has(css, "TabPane", "padding: 0 1;")
    _assert_selector_has(css, "Tab", "padding: 0 2;")
    _assert_selector_has(css, "Tab", "height: 2;")


def test_compact_settings_dimensions() -> None:
    """Settings screen fields should use compact spacing."""
    css = _styles_text()
    _assert_selector_has(css, "#settings-screen", "padding: 1;")
    _assert_selector_has(css, ".settings-panel", "padding: 0 1;")
    _assert_selector_has(css, ".settings-section", "margin-bottom: 1;")
    _assert_selector_has(css, ".settings-section", "padding: 0 1;")
    _assert_selector_has(css, ".settings-actions", "height: auto;")
    _assert_selector_has(css, ".settings-field", "min-height: 2;")
    _assert_selector_has(css, ".settings-field Input", "height: 1;")
    _assert_selector_has(css, ".settings-field Select", "height: 1;")


def test_compact_command_and_bottom_pane_dimensions() -> None:
    """Command input containers should use compact minimum heights."""
    css = _styles_text()
    _assert_selector_has(css, ".bottom-pane", "min-height: 1;")
    _assert_selector_has(css, ".command-input", "min-height: 1;")
    _assert_selector_has(css, "CommandInput", "min-height: 1;")
    _assert_selector_has(css, "#query-command-input", "min-height: 1;")
    _assert_selector_has(css, "#query-command-input Input", "height: 1;")


def test_compact_progress_and_remote_controls() -> None:
    """Progress bar and remote controls should use compact sizing."""
    css = _styles_text()
    _assert_selector_has(css, "ProgressBar", "padding: 0 1;")
    _assert_selector_has(css, "#remote-controls Input", "height: 1;")
    _assert_selector_has(css, "#remote-controls Select", "height: 1;")


def test_global_input_and_select_do_not_force_compact_height() -> None:
    """Global input/select selectors should not force height=1."""
    css = _styles_text()
    input_bodies = _selector_bodies(css, "Input")
    select_bodies = _selector_bodies(css, "Select")

    assert input_bodies, "Global Input selector block is missing."
    assert select_bodies, "Global Select selector block is missing."

    assert not any("height:" in body for body in input_bodies), (
        "Global Input selector should not set height; keep compact rules scoped."
    )
    assert not any("height:" in body for body in select_bodies), (
        "Global Select selector should not set height; keep compact rules scoped."
    )


def test_command_input_default_css_uses_compact_min_height() -> None:
    """Widget default CSS should match compact CommandInput minimum height."""
    from lsm.ui.tui.widgets.input import CommandInput

    assert "min-height: 1;" in CommandInput.DEFAULT_CSS
