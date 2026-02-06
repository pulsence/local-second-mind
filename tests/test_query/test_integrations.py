from __future__ import annotations

from pathlib import Path

from lsm.query.integrations import logseq, obsidian


def test_logseq_helpers(tmp_path: Path) -> None:
    assert logseq.format_wikilink("Note") == "[[Note]]"
    assert logseq.normalize_tag("  Deep Work  ") == "#deep-work"

    note_path = logseq.export_note("content", tmp_path / "graph", "a.md")
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "content"


def test_obsidian_helpers(tmp_path: Path) -> None:
    assert obsidian.format_wikilink("Note") == "[[Note]]"
    assert obsidian.normalize_tag("  Meta Ethics  ") == "#meta-ethics"

    note_path = obsidian.export_note("content", tmp_path / "vault", "b.md")
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "content"

