from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_FIXTURES = (
    "mock_openai_client",
    "mock_embedder",
    "mock_chroma_collection",
    "mock_vectordb_provider",
    "progress_callback_mock",
)
FAKE_BASED_INTEGRATION_TESTS = (
    "tests/test_ingest/test_api.py",
    "tests/test_ingest/test_integration.py",
    "tests/test_query/test_integration.py",
    "tests/test_query/test_integration_progress.py",
)


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_legacy_shared_mock_fixtures_removed_from_conftest() -> None:
    conftest_text = _read("tests/conftest.py")
    for fixture_name in LEGACY_FIXTURES:
        assert f"def {fixture_name}" not in conftest_text


def test_core_integration_suites_do_not_use_unittest_mock_or_mocker() -> None:
    for rel_path in FAKE_BASED_INTEGRATION_TESTS:
        content = _read(rel_path)
        assert "from unittest.mock" not in content, rel_path
        assert "import unittest.mock" not in content, rel_path
        assert "mocker" not in content, rel_path
