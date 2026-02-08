from __future__ import annotations

from types import SimpleNamespace

import pytest

import lsm.ui.tui.commands.ingest as ingest_cmd


class _Provider:
    def __init__(self, name: str = "chromadb", count_value: int = 0):
        self.name = name
        self._count = count_value

    def count(self) -> int:
        return self._count

    def get_stats(self) -> dict:
        return {"provider": self.name, "status": "ok"}

    def is_available(self) -> bool:
        return True

    def health_check(self) -> dict:
        return {"provider": self.name, "status": "ok"}


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        vectordb=SimpleNamespace(provider="chromadb", collection="kb"),
        llm=SimpleNamespace(
            get_tagging_config=lambda: SimpleNamespace(model="gpt-test", provider="openai")
        ),
    )


def test_source_matches_pattern() -> None:
    assert ingest_cmd._source_matches_pattern("/a/b/file.md", "*.md") is True
    assert ingest_cmd._source_matches_pattern("/a/b/file.md", "file.*") is True
    assert ingest_cmd._source_matches_pattern("/a/b/file.md", "*.pdf") is False


def test_parse_tag_args_error_branch() -> None:
    max_chunks, error = ingest_cmd.parse_tag_args("--max nope")
    assert max_chunks is None
    assert "Invalid --max argument" in error


def test_build_and_tag_prompts() -> None:
    cfg = _config()
    assert "WARNING" in ingest_cmd.get_build_confirmation_prompt(True)
    assert ingest_cmd.get_build_confirmation_prompt(False) == ""
    tag_prompt = ingest_cmd.get_tag_confirmation_prompt(cfg, None)
    assert "AI CHUNK TAGGING" in tag_prompt
    assert "Max chunks: unlimited" in tag_prompt


def test_get_wipe_warning_with_count_and_error() -> None:
    provider = _Provider(count_value=12)
    text = ingest_cmd.get_wipe_warning(provider)
    assert "12 chunks" in text

    class _BoomProvider:
        def count(self):
            raise RuntimeError("bad")

    err_text = ingest_cmd.get_wipe_warning(_BoomProvider())
    assert "Error: bad" in err_text


def test_run_build_success_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    fake_result = SimpleNamespace(
        total_files=10,
        completed_files=8,
        skipped_files=2,
        chunks_added=100,
        elapsed_seconds=1.2,
        errors=[],
    )
    monkeypatch.setattr(ingest_cmd, "run_ingest", lambda **kwargs: fake_result)
    text = ingest_cmd.run_build(cfg, force=False)
    assert "Ingest completed successfully" in text
    assert "Total files: 10" in text

    def _boom(**kwargs):
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(ingest_cmd, "run_ingest", _boom)
    err = ingest_cmd.run_build(cfg, force=False)
    assert "Error during ingest: ingest failed" in err


def test_run_wipe_success_empty_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    monkeypatch.setattr(ingest_cmd, "wipe_collection", lambda _cfg: 5)
    assert "Deleted 5 chunks" in ingest_cmd.run_wipe(cfg)

    monkeypatch.setattr(ingest_cmd, "wipe_collection", lambda _cfg: 0)
    assert "already empty" in ingest_cmd.run_wipe(cfg)

    def _boom(_cfg):
        raise RuntimeError("wipe failed")

    monkeypatch.setattr(ingest_cmd, "wipe_collection", _boom)
    assert "Error during wipe: wipe failed" in ingest_cmd.run_wipe(cfg)


def test_run_tag_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    provider = _Provider()
    monkeypatch.setattr(ingest_cmd, "tag_chunks", lambda **kwargs: (7, 1))
    text = ingest_cmd.run_tag(provider, cfg, 10)
    assert "TAGGING COMPLETE" in text
    assert "Successfully tagged: 7 chunks" in text

    def _boom(*args, **kwargs):
        raise RuntimeError("tag failed")

    monkeypatch.setattr(ingest_cmd, "tag_chunks", _boom)
    err = ingest_cmd.run_tag(provider, cfg, 10)
    assert "Error during tagging: tag failed" in err


def test_format_info(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    provider = _Provider(name="chromadb")
    monkeypatch.setattr(
        ingest_cmd,
        "api_get_collection_info",
        lambda _cfg: SimpleNamespace(name="kb", chunk_count=4, provider="chromadb"),
    )
    text = ingest_cmd.format_info(provider, cfg)
    assert "COLLECTION INFO" in text
    assert "Chunks:   4" in text


def test_format_stats_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    provider = _Provider(count_value=10)
    stats = SimpleNamespace(
        chunk_count=10,
        unique_files=2,
        file_types={".md": 6, ".txt": 4},
        top_files=[
            {"source_path": "/a/very/long/path/doc.md", "chunk_count": 6},
            {"source_path": "/a/other.txt", "chunk_count": 4},
        ],
    )
    monkeypatch.setattr(ingest_cmd, "api_get_collection_stats", lambda *args, **kwargs: stats)
    updates = []
    text = ingest_cmd.format_stats(provider, cfg, progress_callback=lambda a, t: updates.append((a, t)))
    assert "COLLECTION STATISTICS" in text
    assert "FILE TYPES" in text
    assert updates[-1] == (10, 10)

    def _boom(*args, **kwargs):
        raise RuntimeError("stats failed")

    monkeypatch.setattr(ingest_cmd, "api_get_collection_stats", _boom)
    assert "Error: stats failed" in ingest_cmd.format_stats(provider, cfg)


def test_format_explore_no_files_and_with_files(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _Provider(count_value=3)
    monkeypatch.setattr(
        ingest_cmd,
        "parse_explore_query",
        lambda q: ("", None, None, "root", False),
    )

    monkeypatch.setattr(ingest_cmd, "iter_collection_metadatas", lambda *args, **kwargs: iter(()))
    no_files = ingest_cmd.format_explore(provider)
    assert "No files found." in no_files

    metas = iter(
        [
            {"source_path": "/docs/a.md", "ext": ".md"},
            {"source_path": "/docs/a.md", "ext": ".md"},
            {"source_path": "/docs/b.txt", "ext": ".txt"},
        ]
    )
    monkeypatch.setattr(ingest_cmd, "iter_collection_metadatas", lambda *args, **kwargs: metas)
    monkeypatch.setattr(ingest_cmd, "compute_common_parts", lambda fs: ())
    monkeypatch.setattr(ingest_cmd, "build_tree", lambda fs, pf, cp: {"file_count": 2, "chunk_count": 3, "children": {}, "files": {}})
    monkeypatch.setattr(ingest_cmd, "format_ingest_tree", lambda tree, label: f"{label} tree")
    yes_files = ingest_cmd.format_explore(provider)
    assert "root tree" in yes_files


def test_format_show_and_search_and_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _Provider()

    assert ingest_cmd.format_show(provider, "") == "Usage: /show <file_path>"
    monkeypatch.setattr(ingest_cmd, "get_file_chunks", lambda c, p: [])
    assert "No chunks found" in ingest_cmd.format_show(provider, "/x")

    monkeypatch.setattr(
        ingest_cmd,
        "get_file_chunks",
        lambda c, p: [{"chunk_index": 0, "text": "x" * 600, "author": "A", "title": "T"}],
    )
    shown = ingest_cmd.format_show(provider, "/x")
    assert "Chunk 1" in shown
    assert "..." in shown

    assert ingest_cmd.format_search(provider, "") == "Usage: /search <query>"
    monkeypatch.setattr(ingest_cmd, "search_metadata", lambda *args, **kwargs: [])
    assert "No results found." in ingest_cmd.format_search(provider, "q")
    monkeypatch.setattr(
        ingest_cmd,
        "search_metadata",
        lambda *args, **kwargs: [{"source_path": "/docs/a.md", "chunk_index": 1, "ext": ".md"} for _ in range(25)],
    )
    search_text = ingest_cmd.format_search(provider, "q")
    assert "Found 25 results" in search_text
    assert "and 5 more results" in search_text

    monkeypatch.setattr(ingest_cmd, "get_all_tags", lambda c: {"ai_tags": ["a"], "user_tags": ["u"]})
    tags = ingest_cmd.format_tags(provider)
    assert "AI-Generated Tags" in tags
    assert "User Tags" in tags


def test_format_vectordb_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    monkeypatch.setattr(ingest_cmd, "list_available_providers", lambda: [])
    assert "No vector DB providers registered." in ingest_cmd.format_vectordb_providers(cfg)

    monkeypatch.setattr(ingest_cmd, "list_available_providers", lambda: ["chromadb", "postgresql"])
    monkeypatch.setattr(ingest_cmd, "create_vectordb_provider", lambda _cfg: _Provider(name="chromadb"))
    providers_text = ingest_cmd.format_vectordb_providers(cfg)
    assert "AVAILABLE VECTOR DB PROVIDERS" in providers_text
    assert "ACTIVE" in providers_text

    monkeypatch.setattr(ingest_cmd, "create_vectordb_provider", lambda _cfg: _Provider(name="chromadb"))
    status = ingest_cmd.format_vectordb_status(cfg)
    assert "VECTOR DB STATUS" in status
    assert "Status:   ok" in status

    def _boom(_cfg):
        raise RuntimeError("status failed")

    monkeypatch.setattr(ingest_cmd, "create_vectordb_provider", _boom)
    err = ingest_cmd.format_vectordb_status(cfg)
    assert "Error: status failed" in err


@pytest.mark.parametrize(
    ("line", "expected_action", "expected_exit", "contains"),
    [
        ("/exit", None, True, "Goodbye"),
        ("/help", None, False, "HELP_TEXT"),
        ("/info", None, False, "INFO"),
        ("/stats", None, False, "STATS"),
        ("/explore test", None, False, "EXPLORE"),
        ("/show /a", None, False, "SHOW"),
        ("/search q", None, False, "SEARCH"),
        ("/build", "build_run", False, ""),
        ("/build --force", "build_confirm", False, "WARNING"),
        ("/tag --max 5", "tag_confirm", False, "TAG"),
        ("/tags", None, False, "TAGS"),
        ("/vectordb-providers", None, False, "PROVIDERS"),
        ("/vectordb-status", None, False, "STATUS"),
        ("/wipe", "wipe_confirm", False, "WIPE"),
        ("/unknown", None, False, "Unknown command"),
    ],
)
def test_handle_command_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    line: str,
    expected_action: str | None,
    expected_exit: bool,
    contains: str,
) -> None:
    cfg = _config()
    provider = _Provider()
    monkeypatch.setattr(ingest_cmd, "get_ingest_help", lambda: "HELP_TEXT")
    monkeypatch.setattr(ingest_cmd, "format_info", lambda p, c: "INFO")
    monkeypatch.setattr(ingest_cmd, "format_stats", lambda p, c, progress_callback=None: "STATS")
    monkeypatch.setattr(ingest_cmd, "format_explore", lambda p, q=None, progress_callback=None: "EXPLORE")
    monkeypatch.setattr(ingest_cmd, "format_show", lambda p, fp: "SHOW")
    monkeypatch.setattr(ingest_cmd, "format_search", lambda p, q: "SEARCH")
    monkeypatch.setattr(ingest_cmd, "get_tag_confirmation_prompt", lambda c, m: "TAG")
    monkeypatch.setattr(ingest_cmd, "format_tags", lambda p: "TAGS")
    monkeypatch.setattr(ingest_cmd, "format_vectordb_providers", lambda c: "PROVIDERS")
    monkeypatch.setattr(ingest_cmd, "format_vectordb_status", lambda c: "STATUS")
    monkeypatch.setattr(ingest_cmd, "get_wipe_warning", lambda p: "WIPE")

    result = ingest_cmd.handle_command(line, provider, cfg)
    assert result.action == expected_action
    assert result.should_exit is expected_exit
    if contains:
        assert contains in result.output
