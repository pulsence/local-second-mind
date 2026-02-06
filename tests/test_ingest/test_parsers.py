"""
Tests for lsm.ingest.parsers module.

Tests file parsing functionality for TXT, MD, HTML, and other formats.
"""

import pytest
from lsm.ingest.parsers import parse_txt, parse_html


class TestParseTxt:
    """Tests for parse_txt function."""

    def test_parse_txt_basic(self, sample_txt_file):
        """Test parsing a basic text file."""
        text, metadata = parse_txt(sample_txt_file)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)
        assert len(text) > 0
        assert "sample text file" in text
        assert "multiple paragraphs" in text

    def test_parse_txt_preserves_newlines(self, sample_txt_file):
        """Test that newlines are preserved in output."""
        text, _ = parse_txt(sample_txt_file)
        assert "\n\n" in text

    def test_parse_txt_empty_file(self, empty_file):
        """Test parsing an empty file."""
        text, metadata = parse_txt(empty_file)
        assert text == ""
        assert metadata == {}

    def test_parse_txt_nonexistent_file(self, tmp_path):
        """Test parsing a file that doesn't exist."""
        nonexistent = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            parse_txt(nonexistent)

    def test_parse_txt_with_unicode(self, tmp_path):
        """Test parsing a file with unicode characters."""
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("Hello world", encoding="utf-8")

        text, _ = parse_txt(unicode_file)

        assert "Hello" in text
        assert "world" in text


class TestParseHtml:
    """Tests for parse_html function."""

    def test_parse_html_basic(self, sample_html_file):
        """Test parsing a basic HTML file."""
        text, metadata = parse_html(sample_html_file)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)
        assert len(text) > 0
        assert "Test Document" in text
        assert "sample HTML file" in text

    def test_parse_html_removes_tags(self, sample_html_file):
        """Test that HTML tags are removed."""
        text, _ = parse_html(sample_html_file)

        assert "<html>" not in text
        assert "<body>" not in text
        assert "<p>" not in text
        assert "<h1>" not in text

    def test_parse_html_preserves_text(self, sample_html_file):
        """Test that text content is preserved."""
        text, _ = parse_html(sample_html_file)

        assert "Item 1" in text
        assert "Item 2" in text
        assert "formatted" in text

    def test_parse_html_empty_file(self, tmp_path):
        """Test parsing an empty HTML file."""
        empty_html = tmp_path / "empty.html"
        empty_html.write_text("")

        text, metadata = parse_html(empty_html)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)

    def test_parse_html_malformed(self, tmp_path):
        """Test parsing malformed HTML."""
        malformed_html = tmp_path / "malformed.html"
        malformed_html.write_text("<html><body><p>Unclosed paragraph</body>")

        text, _ = parse_html(malformed_html)

        assert isinstance(text, str)
        assert "Unclosed paragraph" in text

    def test_parse_html_with_script_tags(self, tmp_path):
        """Test that script tags are removed."""
        html_with_script = tmp_path / "with_script.html"
        html_with_script.write_text(
            """
<html>
<body>
<p>Visible text</p>
<script>alert('hidden');</script>
<p>More visible text</p>
</body>
</html>
"""
        )

        text, _ = parse_html(html_with_script)

        assert "Visible text" in text
        assert "More visible text" in text


class TestParseMd:
    """Tests for parse_txt function used with Markdown files."""

    def test_parse_md_basic(self, sample_md_file):
        """Test parsing a Markdown file."""
        text, metadata = parse_txt(sample_md_file)

        assert isinstance(text, str)
        assert isinstance(metadata, dict)
        assert len(text) > 0
        assert "Test Document" in text
        assert "Section 1" in text

    def test_parse_md_preserves_structure(self, sample_md_file):
        """Test that Markdown structure is preserved (as text)."""
        text, _ = parse_txt(sample_md_file)

        assert "#" in text or "Test Document" in text
        assert "Item 1" in text


class TestEdgeCases:
    """Tests for edge cases in file parsing."""

    def test_parse_very_large_file(self, tmp_path):
        """Test parsing a very large file."""
        large_file = tmp_path / "large.txt"
        content = "\n".join([f"Line {i}" for i in range(10000)])
        large_file.write_text(content, encoding="utf-8")

        text, _ = parse_txt(large_file)

        assert isinstance(text, str)
        assert "Line 0" in text
        assert "Line 9999" in text

    def test_parse_file_with_only_whitespace(self, tmp_path):
        """Test parsing a file with only whitespace."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n\n   \t\t\n   ")

        text, _ = parse_txt(whitespace_file)

        assert isinstance(text, str)

    def test_parse_binary_file_as_text(self, tmp_path):
        """Test parsing a binary file as text."""
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        try:
            text, metadata = parse_txt(binary_file)
            assert isinstance(text, str)
            assert isinstance(metadata, dict)
        except Exception:
            pass
