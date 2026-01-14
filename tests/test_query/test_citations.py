"""
Tests for citation export utilities.
"""

from pathlib import Path

from lsm.query.citations import (
    export_citations_from_sources,
    parse_note_sources,
    export_citations_from_note,
)


def test_export_citations_bibtex(tmp_path):
    sources = [
        {
            "source_path": "/path/to/doc.txt",
            "source_name": "doc.txt",
            "title": "Test Document",
            "author": "Test Author",
            "ingested_at": "2025-01-01T00:00:00Z",
        }
    ]
    out_path = export_citations_from_sources(sources, fmt="bibtex", output_path=tmp_path / "refs.bib")
    content = out_path.read_text(encoding="utf-8")
    assert "@misc" in content
    assert "Test Document" in content
    assert "Test Author" in content


def test_export_citations_zotero(tmp_path):
    sources = [
        {
            "source_path": "https://example.com/article",
            "source_name": "article",
            "title": "Web Article",
            "author": "Example Author",
            "ingested_at": "2025-01-01T00:00:00Z",
        }
    ]
    out_path = export_citations_from_sources(sources, fmt="zotero", output_path=tmp_path / "refs.json")
    content = out_path.read_text(encoding="utf-8")
    assert "\"itemType\"" in content
    assert "Web Article" in content
    assert "example.com" in content


def test_parse_note_sources():
    note_text = """
## Local Sources

### Source 1: doc.txt
**Title:** Test Document
**Author:** Test Author
**Path:** `/path/to/doc.txt`
"""
    sources = parse_note_sources(note_text)
    assert len(sources) == 1
    assert sources[0]["source_path"] == "/path/to/doc.txt"
    assert sources[0]["title"] == "Test Document"
    assert sources[0]["author"] == "Test Author"


def test_export_citations_from_note(tmp_path):
    note_path = tmp_path / "note.md"
    note_path.write_text(
        "### Source 1: doc.txt\n**Path:** `/path/to/doc.txt`\n",
        encoding="utf-8"
    )
    out_path = export_citations_from_note(note_path, fmt="bibtex")
    assert out_path.exists()
    assert out_path.suffix == ".bib"
