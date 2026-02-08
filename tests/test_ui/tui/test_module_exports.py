from __future__ import annotations

import pytest

import lsm.ui.tui.screens as screen_modules
import lsm.ui.tui.widgets as widget_modules


def test_screen_lazy_exports() -> None:
    assert screen_modules.MainScreen is not None
    assert screen_modules.IngestScreen is not None
    assert screen_modules.QueryScreen is not None
    assert screen_modules.SettingsScreen is not None
    assert screen_modules.RemoteScreen is not None
    assert screen_modules.AgentsScreen is not None

    with pytest.raises(AttributeError):
        _ = screen_modules.DoesNotExist


def test_widget_lazy_exports() -> None:
    assert widget_modules.ResultsPanel is not None
    assert widget_modules.ResultItem is not None
    assert widget_modules.CitationSelected is not None
    assert widget_modules.CitationExpanded is not None
    assert widget_modules.CommandInput is not None
    assert widget_modules.CommandSubmitted is not None
    assert widget_modules.StatusBar is not None

    with pytest.raises(AttributeError):
        _ = widget_modules.DoesNotExist
