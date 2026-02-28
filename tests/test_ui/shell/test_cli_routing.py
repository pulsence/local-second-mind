"""Tests for CLI routing changes."""

import pytest

from lsm.__main__ import build_parser


def test_ingest_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest"])


def test_ingest_build_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "ingest",
            "build",
            "--force",
            "--force-reingest-changed-config",
            "--force-file-pattern",
            "*.md",
        ]
    )

    assert args.command == "ingest"
    assert args.ingest_command == "build"
    assert args.force is True
    assert args.force_reingest_changed_config is True
    assert args.force_file_pattern == "*.md"


def test_ingest_tag_parses():
    parser = build_parser()
    args = parser.parse_args(["ingest", "tag", "--max", "10"])

    assert args.command == "ingest"
    assert args.ingest_command == "tag"
    assert args.max == 10


def test_ingest_wipe_requires_confirm_flag():
    parser = build_parser()
    args = parser.parse_args(["ingest", "wipe"])

    assert args.command == "ingest"
    assert args.ingest_command == "wipe"
    assert args.confirm is False


def test_db_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["db"])


def test_db_prune_parses():
    parser = build_parser()
    args = parser.parse_args(["db", "prune", "--max-versions", "3", "--older-than-days", "10"])

    assert args.command == "db"
    assert args.db_command == "prune"
    assert args.max_versions == 3
    assert args.older_than_days == 10


def test_db_complete_parses():
    parser = build_parser()
    args = parser.parse_args(["db", "complete", "--force-file-pattern", "*.pdf"])

    assert args.command == "db"
    assert args.db_command == "complete"
    assert args.force_file_pattern == "*.pdf"


def test_migrate_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "migrate",
            "--from",
            "sqlite",
            "--to",
            "postgresql",
            "--source-path",
            ".lsm",
            "--target-connection-string",
            "postgresql://user:pass@localhost:5432/lsm",
            "--batch-size",
            "500",
        ]
    )

    assert args.command == "migrate"
    assert args.migration_source == "sqlite"
    assert args.migration_target == "postgresql"
    assert args.source_path == ".lsm"
    assert args.target_connection_string.startswith("postgresql://")
    assert args.batch_size == 500


def test_migrate_v07_to_v08_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "migrate",
            "--from",
            "v0.7",
            "--to",
            "v0.8",
            "--source-dir",
            "/tmp/legacy",
        ]
    )
    assert args.command == "migrate"
    assert args.migration_source == "v0.7"
    assert args.migration_target == "v0.8"
    assert args.source_dir == "/tmp/legacy"


def test_query_rejects_interactive_flag():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["query", "--interactive"])
