from __future__ import annotations

import lsm.ui.desktop as desktop
import lsm.ui.web as web


def test_placeholder_modules_export_empty_all() -> None:
    assert desktop.__all__ == []
    assert web.__all__ == []

