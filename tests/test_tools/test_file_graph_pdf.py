from __future__ import annotations

from pathlib import Path

from lsm.ingest.models import PageSegment
from lsm.utils.file_graph import get_file_graph


def _find_node(graph, node_type: str, name: str):
    for node in graph.nodes:
        if node.node_type == node_type and node.name == name:
            return node
    raise AssertionError(f"Missing node {node_type}:{name}")


def test_pdf_graph_pages_and_sections(tmp_path: Path, monkeypatch) -> None:
    segments = [
        PageSegment(text="# Intro\n\nPara 1.", page_number=1),
        PageSegment(text="## Section\n\nPara 2.", page_number=2),
    ]
    combined = "\n\n".join(seg.text for seg in segments)

    def fake_parse_pdf(path, enable_ocr=False, skip_errors=True):
        _ = path, enable_ocr, skip_errors
        return combined, {}, segments

    import lsm.ingest.parsers as parsers

    monkeypatch.setattr(parsers, "parse_pdf", fake_parse_pdf)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"pdf-test-graph")

    graph = get_file_graph(pdf_path)

    root = _find_node(graph, "document", "sample.pdf")
    page1 = _find_node(graph, "page", "Page 1")
    page2 = _find_node(graph, "page", "Page 2")
    heading_intro = _find_node(graph, "heading", "Intro")
    heading_section = _find_node(graph, "heading", "Section")

    assert page1.parent_id == root.id
    assert page2.parent_id == root.id

    assert page1.start_line == 1
    assert page1.end_line == 3
    assert page2.start_line == 5
    assert page2.end_line == 7

    assert heading_intro.parent_id == page1.id
    assert heading_section.parent_id == page2.id

    assert heading_intro.end_line == 3
    assert heading_section.end_line == 7
