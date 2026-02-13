"""Density layout regression tests for TUI CSS sizing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Skip tests if textual is not available.
pytest.importorskip("textual")


def _styles_text() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    styles_dir = repo_root / "lsm" / "ui" / "tui" / "styles"
    return "\n\n".join(
        css_file.read_text(encoding="utf-8")
        for css_file in sorted(styles_dir.glob("*.tcss"))
    )


def _selector_bodies(css: str, selector: str) -> list[str]:
    pattern = rf"(?ms)^\s*{re.escape(selector)}\s*\{{(.*?)^\s*\}}"
    return re.findall(pattern, css)


def _assert_selector_has(css: str, selector: str, expected_line: str) -> None:
    bodies = _selector_bodies(css, selector)
    assert bodies, f"Selector not found: {selector}"
    assert any(expected_line in body for body in bodies), (
        f"Expected '{expected_line}' in selector '{selector}', but it was not found."
    )


def test_comfortable_tab_and_pane_sizes_are_default() -> None:
    """Comfortable mode should be the base/default tab sizing."""
    css = _styles_text()
    _assert_selector_has(css, "ContentTabs", "height: 3;")
    _assert_selector_has(css, "TabPane", "padding: 1 2;")
    _assert_selector_has(css, "Tab", "padding: 1 2;")
    _assert_selector_has(css, "Tab", "height: 3;")


def test_compact_tab_and_pane_sizes_are_scoped() -> None:
    """Compact tab sizing should only apply under density-compact root class."""
    css = _styles_text()
    _assert_selector_has(css, ".density-compact ContentTabs", "height: 2;")
    _assert_selector_has(css, ".density-compact TabPane", "padding: 0 1;")
    _assert_selector_has(css, ".density-compact Tab", "padding: 0 2;")
    _assert_selector_has(css, ".density-compact Tab", "height: 2;")


def test_comfortable_settings_dimensions_are_default() -> None:
    """Settings screen should use comfortable spacing by default."""
    css = _styles_text()
    _assert_selector_has(css, "#settings-screen", "padding: 2;")
    _assert_selector_has(css, ".settings-panel", "padding: 1 2;")
    _assert_selector_has(css, ".settings-section", "margin-bottom: 2;")
    _assert_selector_has(css, ".settings-section", "padding: 1;")
    _assert_selector_has(css, ".settings-field", "min-height: 3;")
    _assert_selector_has(css, ".settings-field Input", "height: 3;")
    _assert_selector_has(css, ".settings-field Select", "height: 3;")


def test_compact_settings_dimensions_are_scoped() -> None:
    """Compact settings spacing should only apply under density-compact."""
    css = _styles_text()
    _assert_selector_has(css, ".density-compact #settings-screen", "padding: 1;")
    _assert_selector_has(css, ".density-compact .settings-panel", "padding: 0 1;")
    _assert_selector_has(css, ".density-compact .settings-section", "margin-bottom: 1;")
    _assert_selector_has(css, ".density-compact .settings-section", "padding: 0 1;")
    _assert_selector_has(css, ".density-compact .settings-field", "min-height: 2;")
    _assert_selector_has(css, ".density-compact .settings-field Input", "height: 1;")
    _assert_selector_has(css, ".density-compact .settings-field Select", "height: 1;")


def test_comfortable_and_compact_command_dimensions() -> None:
    """Command/bottom pane dimensions should have comfortable defaults and compact overrides."""
    css = _styles_text()
    _assert_selector_has(css, ".bottom-pane", "min-height: 3;")
    _assert_selector_has(css, ".command-input", "min-height: 3;")
    _assert_selector_has(css, "CommandInput", "min-height: 3;")
    _assert_selector_has(css, "#query-command-input", "min-height: 3;")
    _assert_selector_has(css, "#query-command-input Input", "height: 3;")

    _assert_selector_has(css, ".density-compact .bottom-pane", "min-height: 1;")
    _assert_selector_has(css, ".density-compact .command-input", "min-height: 1;")
    _assert_selector_has(css, ".density-compact CommandInput", "min-height: 1;")
    _assert_selector_has(css, ".density-compact #query-command-input", "min-height: 1;")
    _assert_selector_has(css, ".density-compact #query-command-input Input", "height: 1;")


def test_progress_and_remote_control_compact_overrides_are_scoped() -> None:
    """Progress and remote controls should only use compact heights under compact class."""
    css = _styles_text()
    _assert_selector_has(css, "ProgressBar", "padding: 1;")
    _assert_selector_has(css, "#remote-controls Input", "height: 3;")
    _assert_selector_has(css, "#remote-controls Select", "height: 3;")

    _assert_selector_has(css, ".density-compact ProgressBar", "padding: 0 1;")
    _assert_selector_has(css, ".density-compact #remote-controls Input", "height: 1;")
    _assert_selector_has(css, ".density-compact #remote-controls Select", "height: 1;")


def test_narrow_breakpoint_rules_exist() -> None:
    """Narrow-terminal layout fallbacks should be present."""
    css = _styles_text()
    _assert_selector_has(css, ".density-narrow #query-top", "layout: vertical;")
    _assert_selector_has(css, ".density-narrow #remote-top", "layout: vertical;")
    _assert_selector_has(css, ".density-narrow #agents-top", "layout: vertical;")


def test_global_input_and_select_do_not_force_compact_height() -> None:
    """Global input/select selectors should not force compact height=1."""
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


def test_command_input_default_css_uses_comfortable_min_height() -> None:
    """Widget default CSS should use comfortable minimum height."""
    from lsm.ui.tui.widgets.input import CommandInput

    assert "min-height: 3;" in CommandInput.DEFAULT_CSS
