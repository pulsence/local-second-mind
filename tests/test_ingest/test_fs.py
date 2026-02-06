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

