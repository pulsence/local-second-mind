"""Agents screen layout regression tests for clipping on default terminals."""

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _styles_text() -> str:
    styles_dir = _repo_root() / "lsm" / "ui" / "tui" / "styles"
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


def test_agents_left_column_is_scrollable_to_prevent_panel_clipping() -> None:
    """Agents left column should scroll vertically when panels exceed viewport height."""
    css = _styles_text()
    _assert_selector_has(css, "#agents-left", "width: 1fr;")
    _assert_selector_has(css, "#agents-left", "min-width: 34;")
    _assert_selector_has(css, "#agents-left", "overflow-y: auto;")
    _assert_selector_has(css, "#agents-left", "overflow-x: hidden;")


def test_agents_panels_use_auto_height_not_fractional_splitting() -> None:
    """Control/status/meta/schedule/memory panels should size to content."""
    css = _styles_text()
    grouped_selector = (
        "#agents-control-panel,\n"
        "#agents-running-panel,\n"
        "#agents-interaction-panel,\n"
        "#agents-status-panel,\n"
        "#agents-meta-panel,\n"
        "#agents-schedule-panel,\n"
        "#agents-memory-panel"
    )
    _assert_selector_has(css, grouped_selector, "height: auto;")


def test_agents_log_panel_uses_two_fraction_width_with_min_constraint() -> None:
    """Log panel should own the larger right column with a minimum width."""
    css = _styles_text()
    _assert_selector_has(css, "#agents-log-panel", "width: 2fr;")
    _assert_selector_has(css, "#agents-log-panel", "min-width: 44;")
    _assert_selector_has(css, "#agents-log-panel", "border: round $primary-darken-2;")


def test_agents_datatables_have_dedicated_styling() -> None:
    """Meta and schedule DataTables should have explicit sizing and border styling."""
    css = _styles_text()
    grouped_selector = (
        "#agents-meta-task-table,\n"
        "#agents-meta-runs-table,\n"
        "#agents-schedule-table"
    )
    _assert_selector_has(css, grouped_selector, "height: 8;")
    _assert_selector_has(css, grouped_selector, "min-height: 6;")
    _assert_selector_has(css, grouped_selector, "border: round $primary-darken-2;")
    _assert_selector_has(css, "#agents-left DataTable", "color: $text;")


def test_agents_compose_uses_multiline_button_groups() -> None:
    """Compose should split button groups into rows for narrow terminal widths."""
    source = _read(_repo_root() / "lsm" / "ui" / "tui" / "screens" / "agents.py")
    assert 'with Vertical(id="agents-buttons")' in source
    assert 'with Vertical(id="agents-schedule-buttons")' in source
    assert 'with Vertical(id="agents-memory-buttons")' in source
    assert 'classes="agents-button-row"' in source
    assert "Refresh every" in source


def test_running_panel_header_uses_flow_layout_without_docked_indicator() -> None:
    """Running panel header should stay auto-sized and avoid docked indicator expansion."""
    css = _styles_text()
    _assert_selector_has(css, ".agents-panel-header", "layout: horizontal;")
    _assert_selector_has(css, ".agents-panel-header", "width: 100%;")
    _assert_selector_has(css, ".agents-panel-header .agents-section-title", "width: 1fr;")
    _assert_selector_has(css, "#agents-interaction-indicator", "margin-left: 1;")

    bodies = _selector_bodies(css, "#agents-interaction-indicator")
    assert bodies, "Selector not found: #agents-interaction-indicator"
    assert all("dock:" not in body for body in bodies), (
        "Interaction indicator must not use docked positioning."
    )


def test_select_overlay_uses_top_layer_to_avoid_table_clipping() -> None:
    """Select overlays should render on top of panel DataTables."""
    css = _styles_text()
    _assert_selector_has(css, "Screen", "layers: base overlay;")
    _assert_selector_has(css, "SelectOverlay", "layer: overlay;")


def test_refresh_interval_selects_have_bottom_margin() -> None:
    """Refresh interval selects should have a small bottom margin for spacing."""
    css = _styles_text()
    _assert_selector_has(
        css,
        "#agents-running-refresh-interval-select,\n#agents-interaction-refresh-interval-select",
        "margin: 0 0 1 0;",
    )
    _assert_selector_has(
        css,
        "#agents-running-refresh-interval-row,\n#agents-interaction-refresh-interval-row",
        "height: auto;",
    )
