from pathlib import Path

from lsm.ingest.models import ParseResult


def test_parse_result_smoke() -> None:
    result = ParseResult(
        source_path="x",
        fp=Path("x.txt"),
        mtime_ns=1,
        size=1,
        file_hash="h",
        chunks=["c"],
        ext=".txt",
        had_prev=False,
        ok=True,
    )
    assert result.ok is True

