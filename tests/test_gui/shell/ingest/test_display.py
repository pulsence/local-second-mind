"""
Tests for ingest REPL display utilities.
"""

import pytest

from lsm.gui.shell.ingest.display import (
    normalize_query_path,
    parse_explore_query,
    new_tree_node,
    build_tree,
    compute_common_parts,
)


class TestNormalizeQueryPath:
    def test_strips_whitespace(self):
        assert normalize_query_path("  test  ") == "test"

    def test_lowercases(self):
        assert normalize_query_path("Test/Path") == "test\\path"  # On Windows

    def test_strips_leading_trailing_separators(self):
        result = normalize_query_path("/test/path/")
        assert not result.startswith("\\") and not result.startswith("/")


class TestParseExploreQuery:
    def test_empty_query_returns_defaults(self):
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query(None)
        assert path_filter is None
        assert ext_filter is None
        assert pattern is None
        assert display_root == "All files"
        assert full_path is False

    def test_extension_filter_with_dot(self):
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query(".pdf")
        assert ext_filter == ".pdf"
        assert path_filter is None
        assert pattern is None

    def test_extension_filter_with_prefix(self):
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query("ext:md")
        assert ext_filter == ".md"

    def test_wildcard_pattern(self):
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query("*.txt")
        assert pattern == "*.txt"
        assert ext_filter is None

    def test_path_filter(self):
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query("docs/notes")
        assert path_filter is not None
        assert "docs" in path_filter.lower()

    def test_full_path_flag(self):
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query("--full-path test")
        assert full_path is True


class TestNewTreeNode:
    def test_creates_valid_node(self):
        node = new_tree_node("test")
        assert node["name"] == "test"
        assert node["children"] == {}
        assert node["files"] == {}
        assert node["file_count"] == 0
        assert node["chunk_count"] == 0


class TestBuildTree:
    def test_empty_file_stats(self):
        root = build_tree({}, None, ())
        assert root["file_count"] == 0
        assert root["chunk_count"] == 0

    def test_single_file(self):
        file_stats = {
            "/docs/test.md": {"ext": ".md", "chunk_count": 5}
        }
        root = build_tree(file_stats, None, ())
        assert root["file_count"] == 1
        assert root["chunk_count"] == 5

    def test_with_common_parts(self):
        file_stats = {
            "/home/user/docs/file1.md": {"ext": ".md", "chunk_count": 3},
            "/home/user/docs/file2.md": {"ext": ".md", "chunk_count": 2},
        }
        common_parts = ("home", "user", "docs")
        root = build_tree(file_stats, None, common_parts)
        assert root["file_count"] == 2
        assert root["chunk_count"] == 5


class TestComputeCommonParts:
    def test_empty_paths(self):
        result = compute_common_parts({})
        assert result == ()

    def test_single_path(self):
        result = compute_common_parts({"/docs/file.md": {"chunk_count": 1}})
        assert len(result) > 0

    def test_common_prefix(self):
        paths = {
            "/home/user/docs/a.md": {"chunk_count": 1},
            "/home/user/docs/b.md": {"chunk_count": 1},
        }
        result = compute_common_parts(paths)
        # Should have a common prefix
        assert len(result) > 0
