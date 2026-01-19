"""Tests for CLI routing changes."""

import pytest

from lsm.__main__ import build_parser


def test_ingest_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest"])


def test_ingest_build_parses():
    parser = build_parser()
    args = parser.parse_args(["ingest", "build", "--force"])

    assert args.command == "ingest"
    assert args.ingest_command == "build"
    assert args.force is True


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


def test_query_rejects_interactive_flag():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["query", "--interactive"])
