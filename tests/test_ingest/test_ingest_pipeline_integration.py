"""
Integration tests for ingest parsing and pipeline behavior.

Covers metadata extraction, chunking positions, OCR detection, and pipeline controls.
"""

import pytest
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from lsm.ingest.parsers import (
    parse_docx,
    parse_html,
    extract_markdown_frontmatter,
)
from lsm.ingest.chunking import chunk_text
from lsm.ingest.pipeline import parse_and_chunk_job
from lsm.ingest.models import ParseResult


class TestMetadataExtraction:
    """Test metadata extraction from various file types."""

    def test_markdown_frontmatter_extraction(self):
        """Test YAML frontmatter extraction from Markdown."""
        text = """---
title: Test Document
author: John Doe
date: 2026-01-11
tags:
  - testing
  - markdown
---

# Main Content

This is the body of the document.
"""
        metadata, body = extract_markdown_frontmatter(text)

        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "John Doe"
        # YAML parses dates as datetime.date objects
        import datetime
        assert metadata["date"] == datetime.date(2026, 1, 11) or metadata["date"] == "2026-01-11"
        assert "tags" in metadata
        assert len(metadata["tags"]) == 2
        assert "---" not in body  # Frontmatter removed
        assert "# Main Content" in body

    def test_markdown_without_frontmatter(self):
        """Test Markdown without frontmatter."""
        text = "# Just a heading\n\nNo frontmatter here."
        metadata, body = extract_markdown_frontmatter(text)

        assert metadata == {}
        assert body == text

    def test_html_metadata_extraction(self):
        """Test metadata extraction from HTML."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="author" content="Jane Smith">
    <meta name="description" content="A test page">
</head>
<body>
    <p>Content here</p>
</body>
</html>
"""
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_path = Path(f.name)

        try:
            text, metadata = parse_html(temp_path)

            assert "title" in metadata
            assert metadata["title"] == "Test Page"
            assert "author" in metadata
            assert metadata["author"] == "Jane Smith"
            assert "description" in metadata
            assert metadata["description"] == "A test page"
            assert "Content here" in text
        finally:
            temp_path.unlink()


class TestChunkPositionTracking:
    """Test chunk position tracking."""

    def test_basic_position_tracking(self):
        """Test that positions are tracked correctly."""
        text = "This is a test. " * 100  # Create text that needs chunking

        chunks, positions = chunk_text(text, chunk_size=100, overlap=20, track_positions=True)

        assert len(chunks) == len(positions)
        assert len(chunks) > 1  # Should be multiple chunks

        # Check first chunk position
        assert positions[0]["chunk_index"] == 0
        assert positions[0]["start_char"] == 0
        assert positions[0]["end_char"] > 0
        assert positions[0]["length"] == len(chunks[0])

        # Verify positions are sequential
        for i in range(len(positions) - 1):
            # Next chunk should start before current chunk ends (due to overlap)
            assert positions[i + 1]["start_char"] < positions[i]["end_char"]

    def test_position_accuracy(self):
        """Test position accuracy for reconstruction."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10

        chunks, positions = chunk_text(text, chunk_size=50, overlap=10, track_positions=True)

        # Verify we can extract chunk text using positions
        from lsm.ingest.utils import normalize_whitespace
        normalized = normalize_whitespace(text)

        for chunk, pos in zip(chunks, positions):
            # Extract text using positions
            extracted = normalized[pos["start_char"]:pos["end_char"]]
            # Should match the chunk (after stripping, which chunker does)
            assert extracted.strip() == chunk.strip()

    def test_chunking_without_positions(self):
        """Test chunk_text with position tracking disabled."""
        text = "Simple text for chunking."

        chunks, positions = chunk_text(text, track_positions=False)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
        assert positions == []


class TestParseAndChunkJob:
    """Test the integrated parse_and_chunk_job function."""

    def test_parse_and_chunk_with_metadata(self):
        """Test that parse_and_chunk_job preserves metadata."""
        # Create a temporary markdown file with frontmatter
        import tempfile
        content = """---
title: Integration Test
author: Test Suite
---

# Test Document

This is test content that should be chunked.
""" * 10  # Make it long enough to chunk

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            result = parse_and_chunk_job(
                fp=temp_path,
                source_path=str(temp_path),
                mtime_ns=temp_path.stat().st_mtime_ns,
                size=temp_path.stat().st_size,
                fhash="test_hash",
                had_prev=False,
                enable_ocr=False,
                chunk_size=200,
                chunk_overlap=50,
            )

            # Check result is successful
            assert result.ok is True
            assert len(result.chunks) > 0

            # Check metadata was extracted
            assert result.metadata is not None
            assert result.metadata.get("title") == "Integration Test"
            assert result.metadata.get("author") == "Test Suite"

            # Check positions were tracked
            assert result.chunk_positions is not None
            assert len(result.chunk_positions) == len(result.chunks)

            # Verify each position has required fields
            for pos in result.chunk_positions:
                assert "chunk_index" in pos
                assert "start_char" in pos
                assert "end_char" in pos
                assert "length" in pos

        finally:
            temp_path.unlink()

    def test_parse_and_chunk_handles_errors(self):
        """Test error handling in parse_and_chunk_job."""
        # Try to parse a non-existent file
        fake_path = Path("/nonexistent/file.txt")

        result = parse_and_chunk_job(
            fp=fake_path,
            source_path=str(fake_path),
            mtime_ns=0,
            size=0,
            fhash="fake",
            had_prev=False,
        )

        assert result.ok is False
        assert result.err is not None

    def test_parse_and_chunk_respects_stop_event(self, tmp_path):
        """Test that stop_event short-circuits parsing."""
        stop_event = threading.Event()
        stop_event.set()

        fake_path = tmp_path / "file.txt"
        fake_path.write_text("Test content", encoding="utf-8")

        result = parse_and_chunk_job(
            fp=fake_path,
            source_path=str(fake_path),
            mtime_ns=fake_path.stat().st_mtime_ns,
            size=fake_path.stat().st_size,
            fhash="fake",
            had_prev=False,
            stop_event=stop_event,
        )

        assert result.ok is False
        assert result.err == "stopped"


class TestOCRSupport:
    """Test OCR functionality (if pytesseract is available)."""

    def test_ocr_detection_logic(self):
        """Test that OCR detection works correctly."""
        from lsm.ingest.parsers import is_page_image_based

        # Mock a page with lots of text
        mock_page_text = Mock()
        mock_page_text.get_text.return_value = "This is a lot of text. " * 100

        assert is_page_image_based(mock_page_text) is False

        # Mock a page with little text (image-based)
        mock_page_image = Mock()
        mock_page_image.get_text.return_value = "12"  # Just page number

        assert is_page_image_based(mock_page_image) is True

    @pytest.mark.skipif(
        not hasattr(__import__('lsm.ingest.parsers', fromlist=['OCR_AVAILABLE']), 'OCR_AVAILABLE') or
        not __import__('lsm.ingest.parsers', fromlist=['OCR_AVAILABLE']).OCR_AVAILABLE,
        reason="OCR not available (pytesseract not installed)"
    )
    def test_ocr_available(self):
        """Test that OCR is available when pytesseract is installed."""
        from lsm.ingest.parsers import OCR_AVAILABLE
        assert OCR_AVAILABLE is True


class TestDocxParsing:
    """Test DOCX parsing failure handling."""

    def test_parse_docx_invalid_file(self, tmp_path):
        """Invalid DOCX should not raise and should return empty content."""
        invalid_docx = tmp_path / "invalid.docx"
        invalid_docx.write_text("not a docx file", encoding="utf-8")

        text, metadata = parse_docx(invalid_docx)

        assert text == ""
        assert metadata == {}


class TestEndToEndPipeline:
    """Test end-to-end pipeline integration."""

    @patch("lsm.ingest.pipeline.SentenceTransformer")
    @patch("lsm.ingest.pipeline.get_chroma_collection")
    def test_pipeline_preserves_metadata(self, mock_get_collection, mock_transformer):
        """Test that metadata flows through entire pipeline."""
        # This is a simplified test - full pipeline testing requires more setup
        # Just verify the data structures support metadata

        from lsm.ingest.models import ParseResult, WriteJob

        # Create a ParseResult with metadata
        pr = ParseResult(
            source_path="/test/doc.md",
            fp=Path("/test/doc.md"),
            mtime_ns=123456789,
            size=1000,
            file_hash="abc123",
            chunks=["chunk1", "chunk2"],
            ext=".md",
            had_prev=False,
            ok=True,
            metadata={"author": "Test", "title": "Doc"},
            chunk_positions=[
                {"chunk_index": 0, "start_char": 0, "end_char": 100, "length": 100},
                {"chunk_index": 1, "start_char": 80, "end_char": 180, "length": 100},
            ]
        )

        # Create WriteJob from ParseResult
        wj = WriteJob(
            source_path=pr.source_path,
            fp=pr.fp,
            mtime_ns=pr.mtime_ns,
            size=pr.size,
            file_hash=pr.file_hash,
            ext=pr.ext,
            chunks=pr.chunks,
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            had_prev=pr.had_prev,
            metadata=pr.metadata,
            chunk_positions=pr.chunk_positions,
        )

        # Verify metadata preserved
        assert wj.metadata == pr.metadata
        assert wj.chunk_positions == pr.chunk_positions
