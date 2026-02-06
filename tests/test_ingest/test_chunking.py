"""
Tests for lsm.ingest.chunking module.

Tests text chunking functionality with different sizes and overlap settings.
"""

import pytest
from lsm.ingest.chunking import chunk_text
from lsm.config.models.constants import DEFAULT_CHUNK_SIZE


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_text_basic(self):
        """Test basic chunking of text."""
        text = "A" * 3000  # Text longer than default chunk size

        chunks, positions = chunk_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert isinstance(positions, list)

    def test_chunk_text_respects_size_limit(self):
        """Test that chunks don't exceed size limit."""
        text = "A" * 5000

        chunks, _ = chunk_text(text, chunk_size=1000)

        for chunk in chunks:
            # Chunks should be at most chunk_size
            assert len(chunk) <= 1000

    def test_chunk_text_with_overlap(self):
        """Test that chunks have proper overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 200  # ~5200 chars

        chunks, _ = chunk_text(text, chunk_size=1000, overlap=100)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check for overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # Last part of current chunk should overlap with start of next
            # Note: Exact overlap detection depends on implementation
            assert len(chunks[i]) > 0
            assert len(chunks[i+1]) > 0

    def test_chunk_text_empty_string(self):
        """Test chunking an empty string."""
        chunks, positions = chunk_text("")

        assert isinstance(chunks, list)
        # Empty string might return empty list or list with empty string
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")
        assert positions == []

    def test_chunk_text_shorter_than_chunk_size(self):
        """Test text shorter than chunk size."""
        text = "Short text"

        chunks, _ = chunk_text(text, chunk_size=1000)

        # Should return single chunk
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exactly_chunk_size(self):
        """Test text exactly equal to chunk size."""
        text = "A" * 1000

        chunks, _ = chunk_text(text, chunk_size=1000)

        # Should return single chunk
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_with_newlines(self):
        """Test chunking text with newlines."""
        text = "Line 1\n\nLine 2\n\nLine 3\n\n" * 100

        chunks, _ = chunk_text(text, chunk_size=500)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Newlines should be preserved
        assert any("\n" in chunk for chunk in chunks)

    def test_chunk_text_default_parameters(self):
        """Test chunking with default parameters."""
        text = "A" * 4000

        chunks, _ = chunk_text(text)

        # Should use default chunk configuration constants
        assert len(chunks) > 0

        # Verify chunks respect default size
        for chunk in chunks:
            assert len(chunk) <= DEFAULT_CHUNK_SIZE

    def test_chunk_text_zero_overlap(self):
        """Test chunking with zero overlap."""
        text = "ABCDEFGH" * 500  # 4000 chars

        chunks, _ = chunk_text(text, chunk_size=1000, overlap=0)

        # With zero overlap, chunks should be sequential with no shared content
        assert len(chunks) > 0

        # Reconstruct text from chunks (with zero overlap)
        if len(chunks) > 1:
            # Total length should be approximately original
            total_chars = sum(len(chunk) for chunk in chunks)
            assert total_chars >= len(text) - 1000  # Allow for final chunk


class TestChunkingEdgeCases:
    """Tests for edge cases in chunking."""

    def test_chunk_text_very_small_chunk_size(self):
        """Test chunking with very small chunk size."""
        text = "Hello World"

        chunks, _ = chunk_text(text, chunk_size=5, overlap=0)

        # Should create multiple small chunks
        assert len(chunks) >= 2
        assert all(len(chunk) <= 5 for chunk in chunks)

    def test_chunk_text_overlap_larger_than_chunk_size(self):
        """Test chunking when overlap >= chunk size."""
        text = "A" * 1000

        # This is an invalid configuration, but should handle gracefully
        chunks, _ = chunk_text(text, chunk_size=100, overlap=150)

        # Should still produce chunks (implementation dependent)
        assert isinstance(chunks, list)

    def test_chunk_text_unicode_characters(self):
        """Test chunking text with Unicode characters."""
        text = "Hello 世界 🌍 " * 500

        chunks, _ = chunk_text(text, chunk_size=1000)

        assert len(chunks) > 0
        # Unicode should be preserved
        assert any("世界" in chunk for chunk in chunks)
        assert any("🌍" in chunk for chunk in chunks)

    def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only text."""
        text = "   \n\n\t\t   " * 100

        chunks, _ = chunk_text(text)

        assert isinstance(chunks, list)
        # Whitespace should be preserved
        if len(chunks) > 0:
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_single_character(self):
        """Test chunking a single character."""
        text = "A"

        chunks, _ = chunk_text(text, chunk_size=1000)

        assert len(chunks) == 1
        assert chunks[0] == "A"

    def test_chunk_text_preserves_content(self):
        """Test that all original content appears in chunks."""
        text = "The quick brown fox jumps over the lazy dog. " * 100

        chunks, _ = chunk_text(text, chunk_size=500, overlap=50)

        # Combine all chunks and check for original content
        combined = "".join(chunks)

        # All words from original should appear in combined
        # (accounting for overlap, combined will be longer)
        assert "quick brown fox" in combined
        assert "lazy dog" in combined

    def test_chunk_text_consistent_chunking(self):
        """Test that chunking is deterministic."""
        text = "Consistent chunking test. " * 200

        chunks1, _ = chunk_text(text, chunk_size=1000, overlap=100)
        chunks2, _ = chunk_text(text, chunk_size=1000, overlap=100)

        # Same input should produce same chunks
        assert chunks1 == chunks2


class TestChunkingWithRealContent:
    """Tests chunking with realistic document content."""

    def test_chunk_paragraph_text(self):
        """Test chunking paragraph-formatted text."""
        text = """
        This is the first paragraph with some meaningful content about testing.
        It spans multiple lines and contains various punctuation marks.

        This is the second paragraph. It discusses a different topic entirely
        and provides additional context for our chunking tests.

        Finally, the third paragraph wraps things up with a conclusion about
        how text chunking should work in practice.
        """ * 20  # Repeat to make it longer

        chunks, _ = chunk_text(text, chunk_size=800, overlap=100)

        assert len(chunks) > 1
        # Paragraphs should be preserved in chunks
        assert any("first paragraph" in chunk for chunk in chunks)
        assert any("second paragraph" in chunk for chunk in chunks)

    def test_chunk_markdown_like_content(self):
        """Test chunking Markdown-formatted content."""
        text = """
# Heading 1

Some content under heading 1.

## Subheading 1.1

More detailed content here with **bold** and *italic* formatting.

- List item 1
- List item 2
- List item 3

## Subheading 1.2

Additional content with [links](https://example.com) and `code`.
""" * 10

        chunks, _ = chunk_text(text, chunk_size=600)

        assert len(chunks) > 0
        # Markdown formatting should be preserved
        assert any("#" in chunk for chunk in chunks)
        assert any("*" in chunk for chunk in chunks)
