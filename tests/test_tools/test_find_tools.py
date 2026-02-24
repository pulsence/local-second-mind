from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools.find_file import FindFileTool
from lsm.agents.tools.find_section import FindSectionTool


def test_find_file_matches_name_and_returns_outline(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "alpha.md").write_text("# Alpha\n\nText.", encoding="utf-8")
    target = root / "target.md"
    target.write_text("# Target\n\nBody.", encoding="utf-8")

    tool = FindFileTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(root),
                "name_pattern": "target",
                "max_depth": 1,
            }
        )
    )

    assert len(payload) == 1
    assert payload[0]["path"] == str(target.resolve())
    outline = payload[0]["outline"]
    assert outline
    assert all(node["depth"] <= 1 for node in outline)


def test_find_file_matches_content_pattern(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "one.txt").write_text("Find the needle here.", encoding="utf-8")
    (root / "two.txt").write_text("No match.", encoding="utf-8")

    tool = FindFileTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(root),
                "content_pattern": "needle",
            }
        )
    )

    assert len(payload) == 1
    assert payload[0]["path"].endswith("one.txt")


def test_find_section_returns_content_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nParagraph text.\n", encoding="utf-8")

    tool = FindSectionTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(path),
                "section": "Title",
            }
        )
    )

    assert len(payload) == 1
    entry = payload[0]
    assert "Title" in entry["content"]
    assert "line_hashes" in entry
    assert entry["start_hash"]
    assert entry["end_hash"]
    assert all(len(item["hash"]) == 8 for item in entry["line_hashes"])


def test_find_section_filters_by_node_type(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    path.write_text(
        "class Sample:\n    pass\n\n\n"
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )

    tool = FindSectionTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(path),
                "section": "helper",
                "node_type": "function",
            }
        )
    )

    assert len(payload) == 1
    assert payload[0]["node"]["node_type"] == "function"
