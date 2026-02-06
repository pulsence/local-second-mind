from lsm.ingest.fs import iter_files


def test_iter_files_filters_by_extension(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    wanted = root / "a.txt"
    ignored = root / "b.md"
    wanted.write_text("a", encoding="utf-8")
    ignored.write_text("b", encoding="utf-8")

    files = list(iter_files([root], {".txt"}, set()))
    assert wanted in files
    assert ignored not in files


def test_iter_files_skips_excluded_directories(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    excluded = root / ".git"
    excluded.mkdir()
    blocked_file = excluded / "secret.txt"
    blocked_file.write_text("blocked", encoding="utf-8")
    allowed_file = root / "ok.txt"
    allowed_file.write_text("ok", encoding="utf-8")

    files = list(iter_files([root], {".txt"}, {".git"}))
    assert allowed_file in files
    assert blocked_file not in files


def test_iter_files_ignores_missing_roots(tmp_path) -> None:
    missing_root = tmp_path / "missing"
    files = list(iter_files([missing_root], {".txt"}, set()))
    assert files == []


def test_iter_files_handles_case_insensitive_extensions(tmp_path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    upper = root / "UPPER.TXT"
    upper.write_text("content", encoding="utf-8")

    files = list(iter_files([root], {".txt"}, set()))
    assert upper in files
