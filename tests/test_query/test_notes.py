"""
Tests for notes writing system.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from lsm.query.notes import (
    slugify,
    generate_timestamp,
    get_note_filename,
    format_local_sources,
    format_remote_sources,
    generate_note_content,
    write_note,
    edit_note_in_editor,
    resolve_notes_dir,
)
from lsm.config.loader import build_config_from_raw


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_slugification(self):
        """Test basic text slugification."""
        assert slugify("Hello World") == "hello-world"
        assert slugify("Test Question?") == "test-question"
        assert slugify("Multiple   Spaces") == "multiple-spaces"

    def test_special_characters(self):
        """Test slugifying special characters."""
        assert slugify("What's the answer?") == "what-s-the-answer"
        assert slugify("Test: A Question") == "test-a-question"
        assert slugify("Foo/Bar\\Baz") == "foo-bar-baz"

    def test_max_length(self):
        """Test slugify respects max length."""
        long_text = "This is a very long question that exceeds the maximum length limit"
        result = slugify(long_text, max_length=20)

        assert len(result) <= 20
        assert not result.endswith("-")

    def test_leading_trailing_hyphens(self):
        """Test slugify removes leading/trailing hyphens."""
        assert slugify("  Test  ") == "test"
        assert slugify("---Test---") == "test"


class TestTimestampGeneration:
    """Tests for timestamp generation."""

    def test_generate_timestamp_format(self):
        """Test timestamp format is correct."""
        timestamp = generate_timestamp()

        # Should be in format: YYYYMMDD-HHMMSS
        assert len(timestamp) == 15  # 8 + 1 + 6
        assert timestamp[8] == "-"
        assert timestamp[:8].isdigit()
        assert timestamp[9:].isdigit()

    def test_generate_timestamp_unique(self):
        """Test timestamps are unique (or at least different when called rapidly)."""
        import time

        timestamp1 = generate_timestamp()
        time.sleep(0.01)  # Small delay
        timestamp2 = generate_timestamp()

        # They should be different or same if called in same second
        # This is a weak test but ensures format is consistent
        assert len(timestamp1) == len(timestamp2)


class TestNoteFilenames:
    """Tests for note filename generation."""

    def test_timestamp_format(self):
        """Test timestamp format generates correct filename."""
        filename = get_note_filename("test question", format="timestamp")

        assert filename.endswith(".md")
        assert len(filename) == 18  # 15 chars + .md

    def test_query_slug_format(self):
        """Test query slug format."""
        filename = get_note_filename("What is RAG?", format="query_slug")

        assert filename.startswith("what-is-rag-")
        assert filename.endswith(".md")
        assert len(filename.split("-")) >= 4  # slug + date parts

    def test_unknown_format_defaults_to_timestamp(self):
        """Test unknown format falls back to timestamp."""
        filename = get_note_filename("test", format="invalid_format")

        # Should default to timestamp
        assert filename.endswith(".md")
        assert len(filename) == 18


class TestSourceFormatting:
    """Tests for source formatting functions."""

    def test_format_local_sources_empty(self):
        """Test formatting empty local sources."""
        result = format_local_sources([])

        assert "No local sources" in result

    def test_format_local_sources_single(self):
        """Test formatting single local source."""
        sources = [
            {
                "text": "This is a test chunk of text.",
                "meta": {
                    "source_path": "/path/to/doc.txt",
                    "chunk_index": 0,
                    "title": "Test Document",
                    "author": "Test Author"
                },
                "distance": 0.2
            }
        ]

        result = format_local_sources(sources)

        assert "Source 1" in result
        assert "doc.txt" in result
        assert "**Relevance:** 0.80" in result  # 1.0 - 0.2
        assert "Test Document" in result
        assert "Test Author" in result
        assert "This is a test chunk" in result

    def test_format_local_sources_wikilinks_and_tags(self):
        """Test wikilink and tag rendering for local sources."""
        sources = [
            {
                "text": "Tagged text",
                "meta": {
                    "source_path": "/path/to/My Note.md",
                    "chunk_index": 1,
                    "ai_tags": "[\"alpha\", \"beta\"]",
                    "user_tags": "[\"gamma\"]",
                },
                "distance": 0.4
            }
        ]

        result = format_local_sources(
            sources,
            use_wikilinks=True,
            include_tags=True,
        )

        assert "[[My Note]]" in result
        assert "#alpha" in result
        assert "#beta" in result
        assert "#gamma" in result

    def test_format_local_sources_truncates_long_text(self):
        """Test long text is truncated."""
        long_text = "A" * 600
        sources = [
            {
                "text": long_text,
                "meta": {"source_path": "/test.txt", "chunk_index": 0},
                "distance": 0.1
            }
        ]

        result = format_local_sources(sources)

        # Should be truncated to ~500 chars
        assert "..." in result
        assert len(result) < len(long_text) + 500

    def test_format_remote_sources_empty(self):
        """Test formatting empty remote sources."""
        result = format_remote_sources([])

        assert "No remote sources" in result

    def test_format_remote_sources_single(self):
        """Test formatting single remote source."""
        sources = [
            {
                "title": "Test Article",
                "url": "https://example.com/article",
                "snippet": "This is a test snippet.",
                "score": 0.9
            }
        ]

        result = format_remote_sources(sources)

        assert "Remote Source 1" in result
        assert "Test Article" in result
        assert "https://example.com/article" in result
        assert "This is a test snippet" in result
        assert "0.90" in result


class TestNoteContentGeneration:
    """Tests for note content generation."""

    def test_generate_basic_note(self):
        """Test generating basic note with query and answer."""
        content = generate_note_content(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            mode="grounded"
        )

        assert "# Query Session" in content
        assert "What is RAG?" in content
        assert "RAG stands for" in content
        assert "grounded" in content
        assert "## Query" in content
        assert "## Answer" in content

    def test_generate_note_with_local_sources(self):
        """Test generating note with local sources."""
        local_sources = [
            {
                "text": "Test text",
                "meta": {"source_path": "/test.txt", "chunk_index": 0},
                "distance": 0.1
            }
        ]

        content = generate_note_content(
            query="Test question",
            answer="Test answer",
            local_sources=local_sources,
            mode="grounded"
        )

        assert "## Local Sources" in content
        assert "Source 1" in content
        assert "test.txt" in content

    def test_generate_note_with_backlinks_and_tags(self):
        """Test backlinks and tags in note content."""
        local_sources = [
            {
                "text": "Test text",
                "meta": {
                    "source_path": "/path/to/My Note.md",
                    "chunk_index": 0,
                    "ai_tags": "[\"alpha\"]",
                },
                "distance": 0.1
            }
        ]

        content = generate_note_content(
            query="Test question",
            answer="Test answer",
            local_sources=local_sources,
            mode="grounded",
            use_wikilinks=True,
            include_backlinks=True,
            include_tags=True,
        )

        assert "**Tags:**" in content
        assert "#alpha" in content
        assert "## Backlinks" in content
        assert "[[My Note]]" in content

    def test_generate_note_with_remote_sources(self):
        """Test generating note with remote sources."""
        remote_sources = [
            {
                "title": "Web Article",
                "url": "https://example.com",
                "snippet": "Article snippet",
                "score": 0.9
            }
        ]

        content = generate_note_content(
            query="Test question",
            answer="Test answer",
            remote_sources=remote_sources,
            mode="hybrid"
        )

        assert "## Remote Sources" in content
        assert "Web Article" in content
        assert "https://example.com" in content

    def test_generate_note_includes_timestamp(self):
        """Test note includes timestamp."""
        content = generate_note_content(
            query="Test",
            answer="Answer",
            mode="grounded"
        )

        assert "**Date:**" in content
        # Should have year in timestamp
        from datetime import datetime
        current_year = str(datetime.now().year)
        assert current_year in content


class TestNoteWriting:
    """Tests for note writing to disk."""

    def test_write_note_creates_directory(self, tmp_path):
        """Test write_note creates notes directory."""
        notes_dir = tmp_path / "notes"

        note_path = write_note(
            notes_dir=notes_dir,
            query="Test question",
            answer="Test answer",
            mode="grounded"
        )

        assert notes_dir.exists()
        assert note_path.exists()
        assert note_path.parent == notes_dir

    def test_write_note_content(self, tmp_path):
        """Test written note has correct content."""
        notes_dir = tmp_path / "notes"

        note_path = write_note(
            notes_dir=notes_dir,
            query="What is X?",
            answer="X is Y.",
            mode="grounded"
        )

        content = note_path.read_text(encoding="utf-8")

        assert "What is X?" in content
        assert "X is Y." in content
        assert "grounded" in content

    def test_write_note_with_sources(self, tmp_path):
        """Test writing note with sources."""
        notes_dir = tmp_path / "notes"
        local_sources = [
            {
                "text": "Source text",
                "meta": {"source_path": "/test.txt", "chunk_index": 0},
                "distance": 0.2
            }
        ]

        note_path = write_note(
            notes_dir=notes_dir,
            query="Test",
            answer="Answer",
            local_sources=local_sources,
            mode="grounded"
        )

        content = note_path.read_text(encoding="utf-8")
        assert "Source 1" in content
        assert "test.txt" in content

    def test_write_note_filename_format(self, tmp_path):
        """Test note filename format."""
        notes_dir = tmp_path / "notes"

        # Timestamp format
        note_path1 = write_note(
            notes_dir=notes_dir,
            query="Test 1",
            answer="Answer 1",
            filename_format="timestamp"
        )
        assert len(note_path1.stem) == 15  # YYYYMMDD-HHMMSS

        # Query slug format
        note_path2 = write_note(
            notes_dir=notes_dir,
            query="Test Question Two",
            answer="Answer 2",
            filename_format="query_slug"
        )
        assert "test-question-two" in note_path2.stem


class TestNoteEditing:
    """Tests for note editing in editor."""

    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    @patch('pathlib.Path.read_text')
    @patch('pathlib.Path.unlink')
    def test_edit_note_in_editor_success(self, mock_unlink, mock_read, mock_temp, mock_run):
        """Test editing note in editor successfully."""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test.md"
        mock_temp.return_value.__enter__.return_value = mock_file

        # Mock reading edited content
        mock_read.return_value = "Edited content"

        # Mock editor environment
        with patch.dict('os.environ', {'EDITOR': 'nano'}):
            result = edit_note_in_editor("Original content")

        assert result == "Edited content"
        mock_run.assert_called_once()
        mock_file.write.assert_called_with("Original content")

    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_edit_note_editor_fallback(self, mock_temp, mock_run):
        """Test editor fallback when EDITOR not set."""
        mock_file = Mock()
        mock_file.name = "/tmp/test.md"
        mock_temp.return_value.__enter__.return_value = mock_file

        with patch.dict('os.environ', {}, clear=True):
            with patch('platform.system', return_value='Windows'):
                with patch('pathlib.Path.read_text', return_value="content"):
                    result = edit_note_in_editor("test")

                    # Should use notepad on Windows
                    mock_run.assert_called_once()
                    call_args = str(mock_run.call_args)
                    assert "notepad" in call_args

    @patch('subprocess.run', side_effect=Exception("Editor failed"))
    @patch('tempfile.NamedTemporaryFile')
    def test_edit_note_editor_failure(self, mock_temp, mock_run):
        """Test handling editor failure."""
        mock_file = Mock()
        mock_file.name = "/tmp/test.md"
        mock_temp.return_value.__enter__.return_value = mock_file

        with pytest.raises(OSError, match="Could not open editor"):
            edit_note_in_editor("test content")


class TestResolveNotesDir:
    """Tests for resolving notes directories from config."""

    def _config(self, tmp_path: Path):
        raw = {
            "global": {
                "global_folder": str(tmp_path / "global"),
            },
            "ingest": {
                "roots": [str(tmp_path / "docs")],
            },
            "llms": {
                "providers": [
                    {"provider_name": "openai", "api_key": "test-key"}
                ],
                "services": {
                    "query": {"provider": "openai", "model": "gpt-5.2"}
                },
            },
            "vectordb": {
                "provider": "chromadb",
                "persist_dir": str(tmp_path / ".chroma"),
                "collection": "test_collection",
            },
            "query": {"mode": "grounded"},
        }
        return build_config_from_raw(raw, tmp_path / "config.json")

    def test_default_notes_dir_uses_global_notes_folder(self, tmp_path: Path):
        config = self._config(tmp_path)
        resolved = resolve_notes_dir(config, "notes")
        assert resolved == (config.global_folder / "Notes").resolve()

    def test_custom_relative_notes_dir_uses_config_parent(self, tmp_path: Path):
        config = self._config(tmp_path)
        resolved = resolve_notes_dir(config, "custom_notes")
        assert resolved == (config.config_path.parent / "custom_notes").resolve()
