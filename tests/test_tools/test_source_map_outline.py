from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools.source_map import SourceMapTool


def test_source_map_returns_outline(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nBody.", encoding="utf-8")

    tool = SourceMapTool()
    payload = json.loads(
        tool.execute(
            {
                "evidence": [
                    {"source_path": str(path), "snippet": "Title", "score": 0.9},
                    {"source_path": str(path), "snippet": "Body", "score": 0.8},
                ],
                "max_depth": 1,
            }
        )
    )

    entry = payload[str(path)]
    assert entry["outline"]
    assert "top_snippets" not in entry
