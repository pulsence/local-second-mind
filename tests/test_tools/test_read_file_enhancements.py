from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools.read_file import ReadFileTool


def test_read_file_section_targeted(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nParagraph.", encoding="utf-8")

    tool = ReadFileTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(path),
                "section": "Title",
            }
        )
    )

    assert "Title" in payload["content"]
    assert "Paragraph." in payload["content"]


def test_read_file_includes_hashes(tmp_path: Path) -> None:
    path = tmp_path / "hashes.txt"
    path.write_text("alpha\nbeta", encoding="utf-8")

    tool = ReadFileTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(path),
                "include_hashes": True,
            }
        )
    )

    hashes = payload["line_hashes"]
    assert len(hashes) == 2
    assert all(len(item["hash"]) == 8 for item in hashes)


def test_read_file_max_depth_filters_outline(tmp_path: Path) -> None:
    path = tmp_path / "outline.md"
    path.write_text("# Heading\n\n## Subheading\n\nBody", encoding="utf-8")

    tool = ReadFileTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(path),
                "max_depth": 1,
            }
        )
    )

    outline = payload["outline"]
    assert outline
    assert all(node["depth"] <= 1 for node in outline)
