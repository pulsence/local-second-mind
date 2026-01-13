"""
Tests for ingest statistics module.
"""

import pytest
from unittest.mock import Mock, MagicMock

from lsm.ingest.stats import (
    get_collection_info,
    get_collection_stats,
    analyze_metadata,
    format_stats_report,
    search_metadata,
)


class TestCollectionInfo:
    """Test collection info retrieval."""

    def test_get_collection_info(self):
        """Test getting basic collection info."""
        mock_collection = Mock()
        mock_collection.name = "test_kb"
        mock_collection.id = "abc-123"
        mock_collection.count.return_value = 1000
        mock_collection.metadata = {"version": "1.0"}

        info = get_collection_info(mock_collection)

        assert info["name"] == "test_kb"
        assert info["id"] == "abc-123"
        assert info["count"] == 1000
        assert "metadata" in info
        assert info["metadata"]["version"] == "1.0"


class TestMetadataAnalysis:
    """Test metadata analysis."""

    def test_analyze_metadata_basic(self):
        """Test basic metadata analysis."""
        metadatas = [
            {
                "source_path": "/docs/file1.md",
                "ext": ".md",
                "author": "Alice",
                "title": "Doc 1",
                "chunk_index": 0,
            },
            {
                "source_path": "/docs/file1.md",
                "ext": ".md",
                "author": "Alice",
                "title": "Doc 1",
                "chunk_index": 1,
            },
            {
                "source_path": "/docs/file2.pdf",
                "ext": ".pdf",
                "author": "Bob",
                "title": "Doc 2",
                "chunk_index": 0,
            },
        ]

        stats = analyze_metadata(metadatas, total_count=3)

        assert stats["unique_files"] == 2
        assert stats["file_types"][".md"] == 2
        assert stats["file_types"][".pdf"] == 1
        assert stats["unique_authors"] == 2
        assert "Alice" in stats["authors"]
        assert "Bob" in stats["authors"]

    def test_analyze_metadata_chunks_per_file(self):
        """Test chunks per file statistics."""
        metadatas = [
            {"source_path": "/docs/file1.md", "ext": ".md", "chunk_index": 0},
            {"source_path": "/docs/file1.md", "ext": ".md", "chunk_index": 1},
            {"source_path": "/docs/file1.md", "ext": ".md", "chunk_index": 2},
            {"source_path": "/docs/file2.md", "ext": ".md", "chunk_index": 0},
        ]

        stats = analyze_metadata(metadatas, total_count=4)

        assert stats["avg_chunks_per_file"] == 2.0  # (3 + 1) / 2
        assert stats["max_chunks_per_file"] == 3
        assert stats["min_chunks_per_file"] == 1

    def test_analyze_metadata_with_dates(self):
        """Test metadata with ingestion dates."""
        metadatas = [
            {
                "source_path": "/docs/file1.md",
                "ext": ".md",
                "ingested_at": "2026-01-10T10:00:00",
                "chunk_index": 0,
            },
            {
                "source_path": "/docs/file2.md",
                "ext": ".md",
                "ingested_at": "2026-01-11T15:30:00",
                "chunk_index": 0,
            },
        ]

        stats = analyze_metadata(metadatas, total_count=2)

        assert "first_ingested" in stats
        assert "last_ingested" in stats
        assert stats["first_ingested"] == "2026-01-10T10:00:00"
        assert stats["last_ingested"] == "2026-01-11T15:30:00"


class TestStatsReport:
    """Test statistics report formatting."""

    def test_format_stats_report(self):
        """Test formatting a stats report."""
        stats = {
            "total_chunks": 1000,
            "unique_files": 50,
            "avg_chunks_per_file": 20.0,
            "max_chunks_per_file": 100,
            "min_chunks_per_file": 5,
            "file_types": {
                ".md": 500,
                ".pdf": 300,
                ".txt": 200,
            },
        }

        report = format_stats_report(stats)

        assert "COLLECTION STATISTICS" in report
        assert "1,000" in report  # Total chunks formatted
        assert "50" in report  # Unique files
        assert "20.0" in report  # Avg chunks
        assert ".md" in report
        assert ".pdf" in report
        assert ".txt" in report

    def test_format_stats_report_with_authors(self):
        """Test report with author information."""
        stats = {
            "total_chunks": 100,
            "unique_files": 10,
            "unique_authors": 3,
            "authors": ["Alice", "Bob", "Charlie"],
        }

        report = format_stats_report(stats)

        assert "AUTHORS" in report
        assert "Alice" in report
        assert "Bob" in report
        assert "Charlie" in report


class TestSearch:
    """Test metadata search."""

    def test_search_metadata_by_query(self):
        """Test searching by query string."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_path": "/docs/python_tutorial.md"},
                {"source_path": "/docs/python_guide.md"},
                {"source_path": "/guides/java_basics.md"},
            ]
        }

        results = search_metadata(mock_collection, query="python")

        assert len(results) == 2
        assert all("python" in r["source_path"] for r in results)

    def test_search_metadata_by_extension(self):
        """Test searching by file extension."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_path": "/docs/file1.pdf", "ext": ".pdf"},
                {"source_path": "/docs/file2.pdf", "ext": ".pdf"},
            ]
        }

        results = search_metadata(mock_collection, ext=".pdf", limit=20)

        # Verify where clause was passed correctly
        mock_collection.get.assert_called_once()
        call_args = mock_collection.get.call_args
        assert "where" in call_args[1]
        assert call_args[1]["where"]["ext"]["$eq"] == ".pdf"

    def test_search_metadata_empty_results(self):
        """Test search with no results."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"metadatas": []}

        results = search_metadata(mock_collection, query="nonexistent")

        assert results == []


class TestGetCollectionStats:
    """Test full collection statistics gathering."""

    def test_get_collection_stats_empty(self):
        """Test stats for empty collection."""
        mock_collection = Mock()
        mock_collection.count.return_value = 0

        stats = get_collection_stats(mock_collection)

        assert stats["total_chunks"] == 0
        assert "message" in stats
        assert "empty" in stats["message"].lower()

    def test_get_collection_stats_with_data(self):
        """Test stats for populated collection."""
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_path": "/docs/file1.md", "ext": ".md", "chunk_index": 0},
                {"source_path": "/docs/file2.md", "ext": ".md", "chunk_index": 0},
            ]
        }

        stats = get_collection_stats(mock_collection, limit=100)

        assert stats["total_chunks"] == 100
        assert stats["analyzed_chunks"] == 2
        assert "unique_files" in stats
        assert stats["unique_files"] == 2

    def test_get_collection_stats_progress_callback(self):
        """Test progress callback is invoked during full scans."""
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_collection.get.side_effect = [
            {
                "metadatas": [
                    {"source_path": "/docs/file1.md", "ext": ".md", "chunk_index": 0},
                    {"source_path": "/docs/file2.md", "ext": ".md", "chunk_index": 0},
                    {"source_path": "/docs/file3.md", "ext": ".md", "chunk_index": 0},
                ]
            },
            {
                "metadatas": [
                    {"source_path": "/docs/file4.md", "ext": ".md", "chunk_index": 0},
                    {"source_path": "/docs/file5.md", "ext": ".md", "chunk_index": 0},
                ]
            },
        ]

        calls = []

        def progress(analyzed: int) -> None:
            calls.append(analyzed)

        stats = get_collection_stats(mock_collection, limit=None, batch_size=3, progress_callback=progress)

        assert stats["total_chunks"] == 5
        assert calls == [3]
