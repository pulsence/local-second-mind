from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools.file_metadata import FileMetadataTool
from lsm.agents.tools.read_file import ReadFileTool
from lsm.utils.file_graph import get_file_graph


def test_read_file_returns_section_by_node_id(tmp_path: Path) -> None:
    html = "<html><body><h1>Intro</h1><p>Para text.</p></body></html>"
    path = tmp_path / "sample.html"
    path.write_text(html, encoding="utf-8")

    graph = get_file_graph(path)
    heading = next(node for node in graph.nodes if node.node_type == "heading")

    tool = ReadFileTool()
    output = tool.execute({"path": str(path), "node_id": heading.id})

    assert output.strip() == "Intro\n\nPara text."


def test_file_metadata_can_include_graph(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nBody.", encoding="utf-8")

    tool = FileMetadataTool()
    payload = json.loads(tool.execute({"paths": [str(path)], "include_graph": True}))

    assert payload[0]["path"] == str(path.resolve())
    assert "graph" in payload[0]
    assert payload[0]["graph"]["nodes"]


def test_line_hash_is_stable_for_same_content(tmp_path: Path) -> None:
    path = tmp_path / "stable.md"
    path.write_text("# Title\n\nParagraph.", encoding="utf-8")

    graph_a = get_file_graph(path)
    graph_b = get_file_graph(path)

    heading_a = next(node for node in graph_a.nodes if node.node_type == "heading")
    heading_b = next(node for node in graph_b.nodes if node.node_type == "heading")

    assert heading_a.line_hash == heading_b.line_hash
