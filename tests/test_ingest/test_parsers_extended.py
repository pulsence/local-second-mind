from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lsm.ingest.parsers as parsers


class _FakePage:
    def __init__(self, text: str = "text", get_text_exc: Exception | None = None, pix=None):
        self._text = text
        self._get_text_exc = get_text_exc
        self._pix = pix or SimpleNamespace(width=1, height=1, samples=b"\x00\x00\x00")

    def get_text(self) -> str:
        if self._get_text_exc:
            raise self._get_text_exc
        return self._text

    def get_pixmap(self, dpi: int = 300):
        return self._pix


class _FakeDoc:
    def __init__(self, pages, metadata=None, load_page_exc_at: int | None = None):
        self._pages = pages
        self.page_count = len(pages)
        self.metadata = metadata or {}
        self._load_page_exc_at = load_page_exc_at

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def load_page(self, page_num: int):
        if self._load_page_exc_at is not None and page_num == self._load_page_exc_at:
            raise RuntimeError("load fail")
        return self._pages[page_num]


def test_extract_markdown_frontmatter_success() -> None:
    text = "---\nauthor: Alice\ntitle: Note\n---\nBody"
    metadata, body = parsers.extract_markdown_frontmatter(text)
    assert metadata["author"] == "Alice"
    assert metadata["title"] == "Note"
    assert body == "Body"


def test_extract_markdown_frontmatter_invalid_yaml() -> None:
    text = "---\n: not valid\n---\nBody"
    metadata, body = parsers.extract_markdown_frontmatter(text)
    assert metadata == {}
    assert "Body" in body


def test_parse_md_extracts_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("---\nauthor: Alice\n---\nHello", encoding="utf-8")
    text, metadata = parsers.parse_md(p)
    assert text == "Hello"
    assert metadata["author"] == "Alice"


def test_parse_md_latin1_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "latin.md"
    p.write_text("x", encoding="utf-8")

    calls = {"count": 0}
    original = Path.read_text

    def _fake_read_text(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    text, metadata = parsers.parse_md(p)
    assert text == "x"
    assert metadata == {}


def test_is_page_image_based_threshold() -> None:
    assert parsers.is_page_image_based(_FakePage("tiny"), min_text_threshold=10) is True
    assert parsers.is_page_image_based(_FakePage("enough text"), min_text_threshold=5) is False


def test_ocr_page_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsers, "OCR_AVAILABLE", False)
    assert parsers.ocr_page(_FakePage()) == ""


def test_ocr_page_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsers, "OCR_AVAILABLE", True)
    monkeypatch.setattr(parsers, "Image", SimpleNamespace(frombytes=lambda *_args, **_kwargs: "img"))
    monkeypatch.setattr(parsers, "pytesseract", SimpleNamespace(image_to_string=lambda _img: "ocr text"))
    assert parsers.ocr_page(_FakePage()) == "ocr text"


def test_ocr_page_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsers, "OCR_AVAILABLE", True)
    monkeypatch.setattr(parsers, "Image", SimpleNamespace(frombytes=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad image"))))
    monkeypatch.setattr(parsers, "pytesseract", SimpleNamespace(image_to_string=lambda _img: "x"))
    assert parsers.ocr_page(_FakePage()) == ""


# ------------------------------------------------------------------
# MuPDF repair strategy tests (task 3.4.3)
# ------------------------------------------------------------------


class _GarbageDoc:
    """Fake fitz.Document supporting tobytes for garbage-collection tests."""

    def __init__(self, pages=None, tobytes_exc=None):
        self._pages = pages or []
        self.page_count = len(self._pages)
        self.metadata = {}
        self._tobytes_exc = tobytes_exc
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def load_page(self, page_num: int):
        return self._pages[page_num]

    def tobytes(self, garbage=0, deflate=False, clean=False):
        if self._tobytes_exc:
            raise self._tobytes_exc
        return b"%PDF-cleaned"

    def close(self):
        self._closed = True


def test_open_pdf_with_repair_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strategy 2 garbage-collection repair path succeeds."""
    calls: list[dict] = []
    sentinel = _FakeDoc(pages=[_FakePage("ok")])
    gc_doc = _GarbageDoc()

    def _fake_open(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        n = len(calls)
        if n == 1:
            raise RuntimeError("syntax error: xref")
        if n == 2:
            # Strategy 2a: open stream for garbage collection
            return gc_doc
        if n == 3:
            # Strategy 2b: reopen cleaned bytes
            return sentinel
        raise AssertionError(f"Unexpected call #{n}")

    monkeypatch.setattr(parsers.fitz, "open", _fake_open)
    p = tmp_path / "a.pdf"
    p.write_bytes(b"pdf")
    assert parsers._open_pdf_with_repair(p) is sentinel
    assert len(calls) == 3


def test_open_pdf_with_repair_non_repairable_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(parsers.fitz, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("permission denied")))
    p = tmp_path / "a.pdf"
    p.write_bytes(b"pdf")
    with pytest.raises(RuntimeError, match="permission denied"):
        parsers._open_pdf_with_repair(p)


def test_repair_zlib_garbage_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strategy 2 uses garbage collection to rebuild xref and recompress streams."""
    calls: list[dict] = []
    sentinel = _FakeDoc(pages=[_FakePage("ok")])
    gc_doc = _GarbageDoc()

    def _fake_open(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        n = len(calls)
        if n == 1:
            raise RuntimeError("zlib error in stream")
        if n == 2:
            return gc_doc
        if n == 3:
            return sentinel
        raise AssertionError(f"Unexpected call #{n}")

    monkeypatch.setattr(parsers.fitz, "open", _fake_open)
    p = tmp_path / "corrupt.pdf"
    p.write_bytes(b"%PDF-corrupt")

    result = parsers._open_pdf_with_repair(p)
    assert result is sentinel
    assert len(calls) == 3
    assert "stream" in calls[1]["kwargs"]
    assert calls[2]["kwargs"].get("stream") == b"%PDF-cleaned"


def test_repair_expanded_markers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """New error markers trigger repair path."""
    for marker in ("trailer not found", "corrupt object", "malformed pdf", "bad startxref"):
        calls = {"n": 0}
        sentinel = object()

        def _make_fake_open(marker_msg):
            def _fake_open(*_args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError(marker_msg)
                if calls["n"] == 2:
                    return _GarbageDoc()
                return sentinel
            return _fake_open

        calls["n"] = 0
        monkeypatch.setattr(parsers.fitz, "open", _make_fake_open(marker))
        p = tmp_path / f"test_{marker.replace(' ', '_')}.pdf"
        p.write_bytes(b"%PDF-test")
        result = parsers._open_pdf_with_repair(p)
        assert result is sentinel, f"Repair not triggered for marker: {marker}"


def test_repair_all_strategies_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When all repair strategies fail, the last exception is raised."""
    def _always_fail(*_args, **_kwargs):
        raise RuntimeError("zlib error: total failure")

    monkeypatch.setattr(parsers.fitz, "open", _always_fail)
    p = tmp_path / "hopeless.pdf"
    p.write_bytes(b"%PDF-bad")

    with pytest.raises(RuntimeError, match="zlib error"):
        parsers._open_pdf_with_repair(p)


def test_repair_garbage_fails_fallback_to_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If garbage collection fails, fall back to plain stream open (Strategy 3)."""
    calls = {"n": 0}
    sentinel = object()

    def _fake_open(*_args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("zlib error in pdf")
        if calls["n"] == 2:
            return _GarbageDoc(tobytes_exc=RuntimeError("tobytes failed"))
        if calls["n"] == 3:
            return sentinel
        raise AssertionError(f"Unexpected call #{calls['n']}")

    monkeypatch.setattr(parsers.fitz, "open", _fake_open)
    p = tmp_path / "partial.pdf"
    p.write_bytes(b"%PDF-partial")

    result = parsers._open_pdf_with_repair(p)
    assert result is sentinel
    assert calls["n"] == 3


def test_parse_pdf_skip_errors_on_open_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """parse_pdf with skip_errors=True returns empty result when open completely fails."""
    def _always_fail(_path):
        raise RuntimeError("zlib error: unrecoverable")

    monkeypatch.setattr(parsers, "_open_pdf_with_repair", _always_fail)
    text, metadata, page_segs = parsers.parse_pdf(tmp_path / "bad.pdf", skip_errors=True)
    assert text == ""
    assert "error" in metadata
    assert "zlib error" in metadata["error"]
    assert page_segs is None


def test_parse_pdf_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = _FakeDoc(
        pages=[_FakePage("page1"), _FakePage("page2")],
        metadata={"author": "A", "title": "T"},
    )
    monkeypatch.setattr(parsers, "_open_pdf_with_repair", lambda _p: doc)
    text, metadata, page_segs = parsers.parse_pdf(tmp_path / "x.pdf")
    assert "page1" in text and "page2" in text
    assert metadata["author"] == "A"
    assert metadata["title"] == "T"
    assert page_segs is not None
    assert len(page_segs) == 2
    assert page_segs[0].page_number == 1
    assert page_segs[1].page_number == 2


def test_parse_pdf_page_errors_collected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = _FakeDoc(
        pages=[_FakePage("ok"), _FakePage("bad", get_text_exc=RuntimeError("extract fail"))],
    )
    monkeypatch.setattr(parsers, "_open_pdf_with_repair", lambda _p: doc)
    monkeypatch.setattr(parsers, "is_page_image_based", lambda _p: False)
    text, metadata, _ = parsers.parse_pdf(tmp_path / "x.pdf", skip_errors=True)
    assert "ok" in text
    assert "_parse_errors" in metadata


def test_parse_pdf_ocr_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = _FakeDoc(pages=[_FakePage("")])
    monkeypatch.setattr(parsers, "_open_pdf_with_repair", lambda _p: doc)
    monkeypatch.setattr(parsers, "ocr_page", lambda _p: "ocr result")
    text, metadata, page_segs = parsers.parse_pdf(tmp_path / "x.pdf", enable_ocr=True)
    assert text == "ocr result"
    assert metadata.get("_parse_errors") is None
    assert page_segs is not None
    assert page_segs[0].page_number == 1


def test_parse_pdf_fail_and_skip_errors_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = _FakeDoc(pages=[_FakePage("x", get_text_exc=RuntimeError("boom"))])
    monkeypatch.setattr(parsers, "_open_pdf_with_repair", lambda _p: doc)
    with pytest.raises(RuntimeError, match="boom"):
        parsers.parse_pdf(tmp_path / "x.pdf", skip_errors=False)


def test_parse_pdf_open_exception_returns_error_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(parsers, "_open_pdf_with_repair", lambda _p: (_ for _ in ()).throw(RuntimeError("open fail")))
    text, metadata, page_segs = parsers.parse_pdf(tmp_path / "x.pdf", skip_errors=True)
    assert text == ""
    assert "open fail" in metadata["error"]
    assert page_segs is None


def test_parse_docx_success_and_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_doc = SimpleNamespace(
        core_properties=SimpleNamespace(author="A", title="T", subject=None, keywords=None, created=None, modified=None),
        paragraphs=[SimpleNamespace(text="p1"), SimpleNamespace(text=""), SimpleNamespace(text="p2")],
    )
    monkeypatch.setattr(parsers, "Document", lambda _p: fake_doc)
    monkeypatch.setattr(parsers, "_docx_has_page_break_before", lambda _para: False)
    text, metadata, page_segs = parsers.parse_docx(tmp_path / "x.docx")
    assert text == "p1\np2"
    assert metadata["author"] == "A"
    # No page breaks detected → page_segs is None
    assert page_segs is None

    monkeypatch.setattr(parsers, "Document", lambda _p: (_ for _ in ()).throw(RuntimeError("bad docx")))
    text2, metadata2, page_segs2 = parsers.parse_docx(tmp_path / "x.docx")
    assert text2 == ""
    assert metadata2 == {}
    assert page_segs2 is None


def test_parse_html_extracts_metadata(tmp_path: Path) -> None:
    p = tmp_path / "a.html"
    p.write_text(
        "<html><head><title>T</title><meta name='author' content='A'><meta name='description' content='D'></head>"
        "<body><script>x=1</script><p>Body</p></body></html>",
        encoding="utf-8",
    )
    text, metadata = parsers.parse_html(p)
    assert "Body" in text
    assert metadata["title"] == "T"
    assert metadata["author"] == "A"
    assert metadata["description"] == "D"


def test_parse_file_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(parsers, "parse_txt", lambda _p: ("txt", {}))
    monkeypatch.setattr(parsers, "parse_md", lambda _p: ("md", {}))
    monkeypatch.setattr(parsers, "parse_pdf", lambda _p, enable_ocr=False, skip_errors=True: ("pdf", {}, None))
    monkeypatch.setattr(parsers, "parse_docx", lambda _p: ("docx", {}, None))
    monkeypatch.setattr(parsers, "parse_html", lambda _p: ("html", {}))

    assert parsers.parse_file(tmp_path / "a.txt")[0] == "txt"
    assert parsers.parse_file(tmp_path / "a.rst")[0] == "txt"
    assert parsers.parse_file(tmp_path / "a.md")[0] == "md"
    assert parsers.parse_file(tmp_path / "a.pdf")[0] == "pdf"
    assert parsers.parse_file(tmp_path / "a.docx")[0] == "docx"
    assert parsers.parse_file(tmp_path / "a.html")[0] == "html"
    assert parsers.parse_file(tmp_path / "a.htm")[0] == "html"
    assert parsers.parse_file(tmp_path / "a.bin")[0] == "txt"

