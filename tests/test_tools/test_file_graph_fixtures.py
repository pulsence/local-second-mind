from __future__ import annotations

import copy
import json
from pathlib import Path

from lsm.ingest.models import PageSegment
from lsm.utils.file_graph import get_file_graph


FIXTURES_DIR = Path("tests/fixtures/file_graph")


def _normalize_graph(graph: dict) -> dict:
    payload = copy.deepcopy(graph)
    payload["path"] = "<path>"
    for node in payload.get("nodes", []):
        if node.get("node_type") == "document":
            meta = node.get("metadata") or {}
            if "path" in meta:
                meta["path"] = "<path>"
            node["metadata"] = meta
    return payload


def _load_expected(name: str) -> dict:
    data = json.loads((FIXTURES_DIR / f"{name}.graph.json").read_text(encoding="utf-8"))
    return _normalize_graph(data)


def test_fixture_graph_code() -> None:
    graph = get_file_graph(FIXTURES_DIR / "code_sample.py").to_dict()
    assert _normalize_graph(graph) == _load_expected("code_sample")


def test_fixture_graph_text() -> None:
    graph = get_file_graph(FIXTURES_DIR / "text_sample.md").to_dict()
    assert _normalize_graph(graph) == _load_expected("text_sample")


def test_fixture_graph_html() -> None:
    graph = get_file_graph(FIXTURES_DIR / "html_sample.html").to_dict()
    assert _normalize_graph(graph) == _load_expected("html_sample")


def test_fixture_graph_pdf(monkeypatch) -> None:
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

    graph = get_file_graph(FIXTURES_DIR / "pdf_sample.pdf").to_dict()
    assert _normalize_graph(graph) == _load_expected("pdf_sample")
