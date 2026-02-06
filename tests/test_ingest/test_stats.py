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
    load_error_report,
    summarize_error_report,
    get_file_chunks,
    iter_collection_metadatas,
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

    def test_get_collection_info_without_metadata(self):
        """Metadata access failures should not fail info retrieval."""
        mock_collection = Mock()
        mock_collection.name = "test_kb"
        mock_collection.id = "abc-123"
        mock_collection.count.return_value = 10
        type(mock_collection).metadata = property(lambda _self: (_ for _ in ()).throw(RuntimeError("no metadata")))

        info = get_collection_info(mock_collection)
        assert info["count"] == 10
        assert "metadata" not in info


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

    def test_format_stats_report_with_parse_errors_and_timeline(self):
        """Report should include parse error details and timeline data."""
        stats = {
            "total_chunks": 20,
            "first_ingested": "2026-02-01",
            "last_ingested": "2026-02-02",
            "parse_errors": {
                "failed_documents": 2,
                "page_errors": 1,
                "generated_at": "2026-02-03",
                "path": "/tmp/report.json",
                "sample_failed_documents": ["/very/long/path/" + "a" * 60 + ".pdf"],
            },
        }
        report = format_stats_report(stats)
        assert "INGESTION TIMELINE" in report
        assert "PARSE ERRORS" in report
        assert "Failed documents: 2" in report
        assert "Page errors:      1" in report


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

    def test_search_metadata_error_returns_empty(self):
        """Exceptions should return an empty result list."""
        mock_collection = Mock()
        mock_collection.get.side_effect = RuntimeError("boom")
        assert search_metadata(mock_collection, query="x") == []


class TestErrorReports:
    """Test error report load and summary helpers."""

    def test_load_error_report_missing_and_invalid(self, tmp_path):
        missing = tmp_path / "missing.json"
        assert load_error_report(missing) is None

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_error_report(bad) is None

    def test_load_and_summarize_error_report(self, tmp_path):
        report_path = tmp_path / "report.json"
        report_path.write_text(
            """
            {
              "generated_at": "2026-02-06",
              "failed_documents": [{"source_path": "/a.pdf"}, {"source_path": "/b.pdf"}],
              "page_errors": [{"source_path": "/a.pdf", "page": 2}]
            }
            """,
            encoding="utf-8",
        )
        report = load_error_report(report_path)
        assert report is not None
        summary = summarize_error_report(report, report_path)
        assert summary["failed_documents"] == 2
        assert summary["page_errors"] == 1
        assert summary["generated_at"] == "2026-02-06"
        assert summary["sample_failed_documents"] == ["/a.pdf", "/b.pdf"]


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

    def test_get_collection_stats_with_error_report_and_no_metadata(self, tmp_path):
        """No metadata should return message; error report still included when available."""
        report_path = tmp_path / "errors.json"
        report_path.write_text(
            '{"generated_at":"2026-02-06","failed_documents":[{"source_path":"x"}],"page_errors":[]}',
            encoding="utf-8",
        )
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_collection.get.return_value = {"metadatas": []}

        stats = get_collection_stats(mock_collection, limit=2, error_report_path=report_path)
        assert stats["total_chunks"] == 5
        assert "No metadata available" in stats["message"]

    def test_get_collection_stats_handles_exception(self):
        """Unexpected failures should surface in error payload."""
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_collection.get.side_effect = RuntimeError("broken")
        stats = get_collection_stats(mock_collection, limit=1)
        assert stats["total_chunks"] == 5
        assert "broken" in stats["error"]


class TestChunkAccess:
    """Test chunk-level retrieval helpers."""

    def test_get_file_chunks_sorts_by_index(self):
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "documents": ["chunk-b", "chunk-a"],
            "metadatas": [
                {"source_path": "/x.md", "chunk_index": 2},
                {"source_path": "/x.md", "chunk_index": 1},
            ],
        }
        chunks = get_file_chunks(mock_collection, "/x.md")
        assert [c["chunk_index"] for c in chunks] == [1, 2]
        assert chunks[0]["text"] == "chunk-a"

    def test_get_file_chunks_error(self):
        mock_collection = Mock()
        mock_collection.get.side_effect = RuntimeError("oops")
        assert get_file_chunks(mock_collection, "/x.md") == []


class TestIterMetadata:
    """Test metadata iteration fallback behavior."""

    def test_iter_collection_metadatas_typeerror_fallback(self):
        mock_collection = Mock()
        mock_collection.get.side_effect = [
            TypeError("offset unsupported"),
            {"metadatas": [{"source_path": "/a.md", "ext": ".md"}]},
        ]
        items = list(iter_collection_metadatas(mock_collection, batch_size=10))
        assert len(items) == 1
        assert items[0]["source_path"] == "/a.md"
