from __future__ import annotations

from types import SimpleNamespace

import pytest

from lsm.ui import utils
from lsm.query.session import SessionState


class _Result:
    def __init__(self, title: str, url: str, snippet: str, score: float, metadata=None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score
        self.metadata = metadata or {}


class _Provider:
    def __init__(self, results):
        self._results = results

    def search(self, query: str, max_results: int = 5):
        return self._results[:max_results]


def test_display_provider_name_and_feature_label() -> None:
    assert utils.display_provider_name("anthropic") == "claude"
    assert utils.display_provider_name("claude") == "claude"
    assert utils.display_provider_name("openai") == "openai"
    assert utils.format_feature_label("tagging") == "tag"
    assert utils.format_feature_label("ranking") == "rerank"
    assert utils.format_feature_label("custom") == "custom"


def test_ingest_banner_and_help() -> None:
    banner = utils.get_ingest_banner()
    help_text = utils.get_ingest_help()
    assert "Ingest Management" in banner
    assert "/build [--force]" in banner
    assert "INGEST COMMANDS" in help_text
    assert "/wipe" in help_text


def test_format_ingest_tree_and_truncation() -> None:
    root = {
        "name": "root",
        "file_count": 2,
        "chunk_count": 3,
        "children": {
            "docs": {
                "name": "docs",
                "file_count": 1,
                "chunk_count": 2,
                "children": {},
                "files": {"a.md": 2},
            }
        },
        "files": {"b.txt": 1},
    }
    rendered = utils.format_ingest_tree(root, "base")
    assert "base/ (2 files, 3 chunks)" in rendered
    assert "docs/" in rendered
    assert "a.md (2 chunks)" in rendered
    assert "b.txt (1 chunks)" in rendered

    truncated = utils.format_ingest_tree(root, "base", max_entries=1)
    assert "Output truncated." in truncated


def test_open_file_missing_or_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    assert utils.open_file("") is False

    monkeypatch.setattr(utils.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(utils.sys, "platform", "linux")

    def _boom(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(utils.subprocess, "run", _boom)
    assert utils.open_file("/tmp/file") is False


def test_open_file_platform_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.os.path, "exists", lambda _p: True)

    calls = []

    def _fake_run(args, check=False):
        calls.append(tuple(args))
        return None

    monkeypatch.setattr(utils.subprocess, "run", _fake_run)
    monkeypatch.setattr(utils.sys, "platform", "darwin")
    assert utils.open_file("/tmp/file") is True
    assert calls[-1][0] == "open"

    monkeypatch.setattr(utils.sys, "platform", "linux")
    assert utils.open_file("/tmp/file") is True
    assert calls[-1][0] == "xdg-open"


def test_open_file_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(utils.sys, "platform", "win32")
    if not hasattr(utils.os, "startfile"):
        monkeypatch.setattr(utils.os, "startfile", lambda _p: None, raising=False)
    else:
        monkeypatch.setattr(utils.os, "startfile", lambda _p: None)

    assert utils.open_file("C:\\temp\\x.txt") is True


def test_run_remote_search_provider_not_found() -> None:
    config = SimpleNamespace(remote_providers=[])
    output = utils.run_remote_search("missing", "query", config)
    assert "Provider not found" in output


def test_run_remote_search_success_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_config = SimpleNamespace(
        name="wiki",
        type="wikipedia",
        weight=1.0,
        api_key=None,
        endpoint=None,
        max_results=None,
        language=None,
        user_agent=None,
        timeout=None,
        min_interval_seconds=None,
        section_limit=None,
        snippet_max_chars=None,
        include_disambiguation=None,
    )
    config = SimpleNamespace(remote_providers=[provider_config])
    monkeypatch.setattr(utils, "create_remote_provider", lambda _t, _cfg: _Provider([]))

    output = utils.run_remote_search("wiki", "python", config)
    assert "No results found" in output


def test_run_remote_search_success_with_results(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_config = SimpleNamespace(
        name="wiki",
        type="wikipedia",
        weight=1.0,
        api_key=None,
        endpoint=None,
        max_results=None,
        language=None,
        user_agent=None,
        timeout=None,
        min_interval_seconds=None,
        section_limit=None,
        snippet_max_chars=None,
        include_disambiguation=None,
    )
    config = SimpleNamespace(remote_providers=[provider_config])
    long_snippet = "x" * 250
    monkeypatch.setattr(
        utils,
        "create_remote_provider",
        lambda _t, _cfg: _Provider([_Result("Title", "https://e.com", long_snippet, 0.9)]),
    )

    output = utils.run_remote_search("wiki", "python", config)
    assert "Found 1 results" in output
    assert "URL: https://e.com" in output
    assert "..." in output


def test_run_remote_search_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_config = SimpleNamespace(
        name="wiki",
        type="wikipedia",
        weight=1.0,
        api_key=None,
        endpoint=None,
        max_results=None,
        language=None,
        user_agent=None,
        timeout=None,
        min_interval_seconds=None,
        section_limit=None,
        snippet_max_chars=None,
        include_disambiguation=None,
    )
    config = SimpleNamespace(remote_providers=[provider_config])

    def _boom(_t, _cfg):
        raise RuntimeError("bad provider")

    monkeypatch.setattr(utils, "create_remote_provider", _boom)
    output = utils.run_remote_search("wiki", "python", config)
    assert "Search failed: bad provider" in output


def test_run_remote_search_all_no_providers() -> None:
    state = SessionState()
    config = SimpleNamespace(get_active_remote_providers=lambda: [])
    output = utils.run_remote_search_all("q", config, state)
    assert "No remote providers configured." in output


def test_run_remote_search_all_dedupes_and_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    p1 = SimpleNamespace(
        name="a",
        type="wikipedia",
        weight=1.0,
        api_key=None,
        endpoint=None,
        max_results=5,
        language=None,
        user_agent=None,
        timeout=None,
        min_interval_seconds=None,
        section_limit=None,
        snippet_max_chars=None,
        include_disambiguation=None,
    )
    p2 = SimpleNamespace(
        name="b",
        type="wikipedia",
        weight=0.5,
        api_key=None,
        endpoint=None,
        max_results=5,
        language=None,
        user_agent=None,
        timeout=None,
        min_interval_seconds=None,
        section_limit=None,
        snippet_max_chars=None,
        include_disambiguation=None,
    )
    config = SimpleNamespace(get_active_remote_providers=lambda: [p1, p2])
    state = SessionState()

    providers = {
        "a": _Provider([_Result("A1", "https://same", "s", 0.9)]),
        "b": _Provider([_Result("B1", "https://same", "s2", 0.8), _Result("B2", "https://u2", "s3", 0.7)]),
    }

    def _factory(_type, cfg):
        return providers[cfg["type"] if cfg["type"] in providers else cfg.get("name", "a")]

    # Map by provider type is insufficient for two providers; select by call order.
    calls = {"idx": 0}

    def _factory_ordered(_type, _cfg):
        calls["idx"] += 1
        return providers["a"] if calls["idx"] == 1 else providers["b"]

    monkeypatch.setattr(utils, "create_remote_provider", _factory_ordered)
    output = utils.run_remote_search_all("python", config, state)

    assert "Total: 2 unique results" in output
    assert len(state.last_remote_sources) == 2


def test_format_source_list_groups_labels() -> None:
    formatted = utils.format_source_list(
        [
            {"label": "S1", "source_path": "/docs/readme.md", "source_name": "readme.md"},
            {"label": "S2", "source_path": "/docs/readme.md", "source_name": "readme.md"},
            {"label": "S3", "source_path": "/docs/guide.md", "source_name": "guide.md"},
        ]
    )
    assert "Sources:" in formatted
    assert "[S1] [S2]" in formatted
    assert "guide.md" in formatted


def test_format_source_list_empty() -> None:
    assert utils.format_source_list([]) == ""
