from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools.edit_file import EditFileTool
from lsm.utils.file_graph import compute_line_hashes, get_file_graph


def _line_hashes_for(text: str) -> list[str]:
    lines = text.split("\n")
    return compute_line_hashes(lines)


def test_compute_line_hashes_returns_short_hashes() -> None:
    hashes = compute_line_hashes(["alpha", "beta"])
    assert len(hashes) == 2
    assert all(len(value) == 8 for value in hashes)
    assert hashes[0] != hashes[1]


def test_edit_file_replaces_range_and_returns_graph(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nOld line\n", encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    hashes = _line_hashes_for(text)
    target_idx = text.split("\n").index("Old line")

    tool = EditFileTool()
    result = json.loads(
        tool.execute(
            {
                "path": str(path),
                "start_hash": hashes[target_idx],
                "end_hash": hashes[target_idx],
                "new_content": "New line",
            }
        )
    )

    assert result["status"] == "ok"
    assert "graph" in result
    updated = path.read_text(encoding="utf-8")
    assert "New line" in updated
    assert "Old line" not in updated

    refreshed = get_file_graph(path)
    assert result["graph"]["content_hash"] == refreshed.content_hash


def test_edit_file_hash_mismatch_returns_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    hashes = _line_hashes_for(path.read_text(encoding="utf-8"))

    tool = EditFileTool()
    payload = json.loads(
        tool.execute(
            {
                "path": str(path),
                "start_hash": "deadbeef",
                "end_hash": hashes[1],
                "new_content": "X",
                "start_line": 1,
                "end_line": 2,
            }
        )
    )

    assert payload["status"] == "error"
    assert payload["error"] == "hash_mismatch"
    details = payload["details"]
    assert "actual_hashes" in details
    assert "context" in details
    assert "suggestions" in details


def test_edit_file_requires_disambiguation_on_collisions(tmp_path: Path) -> None:
    path = tmp_path / "collision.txt"
    path.write_text("same\nsame\n", encoding="utf-8")
    hashes = _line_hashes_for(path.read_text(encoding="utf-8"))

    tool = EditFileTool()
    collision = json.loads(
        tool.execute(
            {
                "path": str(path),
                "start_hash": hashes[0],
                "end_hash": hashes[0],
                "new_content": "unique",
            }
        )
    )
    assert collision["status"] == "error"
    assert collision["error"] == "hash_collision"

    resolved = json.loads(
        tool.execute(
            {
                "path": str(path),
                "start_hash": hashes[0],
                "end_hash": hashes[0],
                "start_line": 1,
                "end_line": 1,
                "new_content": "unique",
            }
        )
    )
    assert resolved["status"] == "ok"
    assert path.read_text(encoding="utf-8").split("\n")[0] == "unique"
