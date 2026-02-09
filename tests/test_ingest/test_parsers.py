"""
Tests for lsm.ingest.parsers module using realistic fixture corpus files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lsm.ingest.parsers import parse_file, parse_html, parse_md, parse_txt


def _document_path(synthetic_data_root: Path, filename: str) -> Path:
    return synthetic_data_root / "documents" / filename


class TestParseTxt:
    """Tests for parsing text-like content from realistic fixture files."""

    def test_parse_txt_research_paper_contains_expected_sections(
        self,
        synthetic_data_root: Path,
    ) -> None:
        fp = _document_path(synthetic_data_root, "research_paper.md")
        text, metadata = parse_txt(fp)

        assert isinstance(text, str)
        assert metadata == {}
        assert "Retrieval-Augmented Knowledge Work in Local-First Systems" in text
        assert "## 3. Methodology" in text
        assert "def run_pipeline" in text

    def test_parse_txt_unicode_content_preserves_non_ascii(
        self,
        synthetic_data_root: Path,
    ) -> None:
        fp = _document_path(synthetic_data_root, "unicode_content.txt")
        text, metadata = parse_txt(fp)

        assert isinstance(text, str)
        assert metadata == {}
        assert "Greek:" in text
        assert "Chinese:" in text
        assert "Arabic:" in text
        assert any(ord(ch) > 127 for ch in text)

    def test_parse_txt_whitespace_only_file(self, synthetic_data_root: Path) -> None:
        fp = _document_path(synthetic_data_root, "empty_with_whitespace.txt")
        text, metadata = parse_txt(fp)

        assert isinstance(text, str)
        assert metadata == {}
        assert text.strip() == ""

    def test_parse_txt_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_txt(tmp_path / "missing.txt")


class TestParseMarkdown:
    """Tests for parse_md on realistic markdown content."""

    def test_parse_md_research_document(self, synthetic_data_root: Path) -> None:
        fp = _document_path(synthetic_data_root, "research_paper.md")
        text, metadata = parse_md(fp)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)
        assert metadata == {}
        assert "## 1. Introduction" in text
        assert "### 3.2 Pipeline Definition" in text
        assert "```python" in text


class TestParseHtml:
    """Tests for HTML parsing with metadata and structure preservation."""

    def test_parse_html_technical_manual(self, synthetic_data_root: Path) -> None:
        fp = _document_path(synthetic_data_root, "technical_manual.html")
        text, metadata = parse_html(fp)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)
        assert metadata.get("title") == "Knowledge System Operations Manual"
        assert "# Knowledge System Operations Manual" in text
        assert "## Operational Workflow" in text
        assert "### Step 1" in text
        assert "<script" not in text.lower()

    def test_parse_html_removes_script_style_noscript(self, tmp_path: Path) -> None:
        fp = tmp_path / "with_script.html"
        fp.write_text(
            """
<html>
  <head><title>Script Test</title><style>.hidden{display:none}</style></head>
  <body>
    <h1>Visible Heading</h1>
    <p>Visible paragraph.</p>
    <script>window.secret = "should_not_appear";</script>
    <noscript>This should not appear either.</noscript>
  </body>
</html>
""",
            encoding="utf-8",
        )

        text, metadata = parse_html(fp)

        assert metadata.get("title") == "Script Test"
        assert "Visible Heading" in text
        assert "Visible paragraph." in text
        assert "should_not_appear" not in text
        assert "This should not appear either." not in text


class TestParseFileDispatch:
    """Tests the parse_file extension dispatch and tuple contracts."""

    def test_parse_file_for_markdown_and_html(self, synthetic_data_root: Path) -> None:
        md_text, md_meta, md_pages = parse_file(
            _document_path(synthetic_data_root, "research_paper.md")
        )
        html_text, html_meta, html_pages = parse_file(
            _document_path(synthetic_data_root, "technical_manual.html")
        )

        assert isinstance(md_text, str)
        assert isinstance(md_meta, dict)
        assert md_pages is None
        assert "Methodology" in md_text

        assert isinstance(html_text, str)
        assert isinstance(html_meta, dict)
        assert html_pages is None
        assert "Operational Workflow" in html_text

    def test_parse_file_unknown_extension_falls_back_to_text(self, tmp_path: Path) -> None:
        fp = tmp_path / "custom.ext"
        fp.write_text("Fallback parser content.", encoding="utf-8")

        text, metadata, page_segments = parse_file(fp)

        assert text == "Fallback parser content."
        assert metadata == {}
        assert page_segments is None

    def test_parse_binary_file_as_text_best_effort(self, tmp_path: Path) -> None:
        fp = tmp_path / "binary.dat"
        fp.write_bytes(b"\x00\x01\x02sample\xff\xfe")

        text, metadata = parse_txt(fp)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)
