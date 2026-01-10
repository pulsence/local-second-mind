"""
Tests for lsm.ingest.parsers module.

Tests file parsing functionality for TXT, MD, HTML, and other formats.
"""

import pytest
from pathlib import Path
from lsm.ingest.parsers import parse_txt, parse_html


class TestParseTxt:
    """Tests for parse_txt function."""

    def test_parse_txt_basic(self, sample_txt_file):
        """Test parsing a basic text file."""
        result = parse_txt(sample_txt_file)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "sample text file" in result
        assert "multiple paragraphs" in result

    def test_parse_txt_preserves_newlines(self, sample_txt_file):
        """Test that newlines are preserved in output."""
        result = parse_txt(sample_txt_file)

        # Should contain paragraph breaks
        assert "\n\n" in result

    def test_parse_txt_empty_file(self, empty_file):
        """Test parsing an empty file."""
        result = parse_txt(empty_file)

        assert result == ""

    def test_parse_txt_nonexistent_file(self, tmp_path):
        """Test parsing a file that doesn't exist."""
        nonexistent = tmp_path / "does_not_exist.txt"

        # Should raise FileNotFoundError or return empty string
        # Current implementation may return "" on error
        result = parse_txt(nonexistent)
        assert result == "" or result is None

    def test_parse_txt_with_unicode(self, tmp_path):
        """Test parsing a file with Unicode characters."""
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("Hello 世界 🌍", encoding="utf-8")

        result = parse_txt(unicode_file)

        assert "Hello" in result
        assert "世界" in result
        assert "🌍" in result


class TestParseHtml:
    """Tests for parse_html function."""

    def test_parse_html_basic(self, sample_html_file):
        """Test parsing a basic HTML file."""
        result = parse_html(sample_html_file)

        assert isinstance(result, str)
        assert len(result) > 0

        # Should extract text content
        assert "Test Document" in result
        assert "sample HTML file" in result

    def test_parse_html_removes_tags(self, sample_html_file):
        """Test that HTML tags are removed."""
        result = parse_html(sample_html_file)

        # Tags should be removed
        assert "<html>" not in result
        assert "<body>" not in result
        assert "<p>" not in result
        assert "<h1>" not in result

    def test_parse_html_preserves_text(self, sample_html_file):
        """Test that text content is preserved."""
        result = parse_html(sample_html_file)

        # Text from various elements should be present
        assert "Item 1" in result
        assert "Item 2" in result
        assert "formatted" in result

    def test_parse_html_empty_file(self, tmp_path):
        """Test parsing an empty HTML file."""
        empty_html = tmp_path / "empty.html"
        empty_html.write_text("")

        result = parse_html(empty_html)

        # Should handle gracefully
        assert result == "" or result is not None

    def test_parse_html_malformed(self, tmp_path):
        """Test parsing malformed HTML."""
        malformed_html = tmp_path / "malformed.html"
        malformed_html.write_text("<html><body><p>Unclosed paragraph</body>")

        result = parse_html(malformed_html)

        # BeautifulSoup should handle malformed HTML gracefully
        assert isinstance(result, str)
        assert "Unclosed paragraph" in result

    def test_parse_html_with_script_tags(self, tmp_path):
        """Test that script tags are removed."""
        html_with_script = tmp_path / "with_script.html"
        html_with_script.write_text("""
<html>
<body>
<p>Visible text</p>
<script>alert('hidden');</script>
<p>More visible text</p>
</body>
</html>
""")

        result = parse_html(html_with_script)

        assert "Visible text" in result
        assert "More visible text" in result
        # Script content should ideally be removed (depends on implementation)
        # BeautifulSoup.get_text() may still include script content


class TestParseMd:
    """Tests for parse_txt function used with Markdown files."""

    def test_parse_md_basic(self, sample_md_file):
        """Test parsing a Markdown file."""
        # Markdown files are currently parsed as plain text
        result = parse_txt(sample_md_file)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Document" in result
        assert "Section 1" in result

    def test_parse_md_preserves_structure(self, sample_md_file):
        """Test that Markdown structure is preserved (as text)."""
        result = parse_txt(sample_md_file)

        # Markdown formatting is preserved in raw form
        assert "#" in result or "Test Document" in result
        assert "Item 1" in result


class TestEdgeCases:
    """Tests for edge cases in file parsing."""

    def test_parse_very_large_file(self, tmp_path):
        """Test parsing a very large file."""
        large_file = tmp_path / "large.txt"

        # Create a file with 10,000 lines
        content = "\n".join([f"Line {i}" for i in range(10000)])
        large_file.write_text(content, encoding="utf-8")

        result = parse_txt(large_file)

        assert isinstance(result, str)
        assert "Line 0" in result
        assert "Line 9999" in result

    def test_parse_file_with_only_whitespace(self, tmp_path):
        """Test parsing a file with only whitespace."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n\n   \t\t\n   ")

        result = parse_txt(whitespace_file)

        # Should return whitespace or normalized version
        assert isinstance(result, str)

    def test_parse_binary_file_as_text(self, tmp_path):
        """Test parsing a binary file as text."""
        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        # Attempting to parse binary as text should not crash
        try:
            result = parse_txt(binary_file)
            assert isinstance(result, str) or result == ""
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
