"""
Tests for lsm.ingest.chunking using realistic corpus documents.
"""

from __future__ import annotations

from pathlib import Path

from lsm.ingest.chunking import chunk_text
from lsm.ingest.parsers import parse_html, parse_md, parse_txt


def _document_path(synthetic_data_root: Path, filename: str) -> Path:
    return synthetic_data_root / "documents" / filename


class TestChunkTextCore:
    """Core chunking behavior and offset tracking."""

    def test_chunk_text_empty_returns_no_chunks(self) -> None:
        chunks, positions = chunk_text("")
        assert chunks == []
        assert positions == []

    def test_chunk_text_short_text_returns_single_chunk(self) -> None:
        text = "Short text that should stay in one chunk."
        chunks, positions = chunk_text(text, chunk_size=1000, overlap=100)

        assert len(chunks) == 1
        assert chunks[0] == text
        assert len(positions) == 1
        assert positions[0]["start_char"] == 0
        assert positions[0]["end_char"] == len(text)

    def test_chunk_text_positions_are_monotonic(self) -> None:
        text = ("Chunking should preserve deterministic offsets. " * 120).strip()
        chunks, positions = chunk_text(text, chunk_size=500, overlap=80)

        assert len(chunks) == len(positions)
        for pos in positions:
            assert pos["start_char"] < pos["end_char"]
            assert pos["length"] == pos["end_char"] - pos["start_char"]
        for prev, curr in zip(positions, positions[1:]):
            assert curr["start_char"] >= prev["start_char"]


class TestChunkingWithRealCorpus:
    """Chunking behavior on realistic fixture documents."""

    def test_chunk_research_paper_with_overlap(self, synthetic_data_root: Path) -> None:
        text, _ = parse_md(_document_path(synthetic_data_root, "research_paper.md"))
        chunks, positions = chunk_text(text, chunk_size=900, overlap=150)

        assert len(chunks) > 3
        assert len(chunks) == len(positions)
        assert all(len(chunk) <= 900 for chunk in chunks)
        assert any("Methodology" in chunk for chunk in chunks)

        for prev, curr in zip(positions, positions[1:]):
            # With overlap enabled, the next start should advance but not skip
            # farther than one full chunk window.
            assert curr["start_char"] > prev["start_char"]
            assert curr["start_char"] - prev["start_char"] <= 900

    def test_chunk_large_document_stress(self, synthetic_data_root: Path) -> None:
        text, _ = parse_md(_document_path(synthetic_data_root, "large_document.md"))
        chunks, positions = chunk_text(text, chunk_size=700, overlap=120)

        assert len(chunks) >= 20
        assert len(chunks) == len(positions)
        assert positions[-1]["end_char"] > 5000

    def test_chunk_text_is_deterministic_for_real_document(
        self,
        synthetic_data_root: Path,
    ) -> None:
        text, _ = parse_md(_document_path(synthetic_data_root, "research_paper.md"))

        chunks1, positions1 = chunk_text(text, chunk_size=850, overlap=120)
        chunks2, positions2 = chunk_text(text, chunk_size=850, overlap=120)

        assert chunks1 == chunks2
        assert positions1 == positions2

    def test_chunking_html_after_parse(self, synthetic_data_root: Path) -> None:
        text, metadata = parse_html(_document_path(synthetic_data_root, "technical_manual.html"))
        chunks, _ = chunk_text(text, chunk_size=800, overlap=100)

        assert metadata.get("title") == "Knowledge System Operations Manual"
        assert len(chunks) > 2
        assert any("Knowledge System Operations Manual" in chunk for chunk in chunks)
        assert any("Operational Workflow" in chunk for chunk in chunks)

    def test_chunking_unicode_document_preserves_non_ascii(self, synthetic_data_root: Path) -> None:
        text, _ = parse_txt(_document_path(synthetic_data_root, "unicode_content.txt"))
        chunks, _ = chunk_text(text, chunk_size=120, overlap=20)

        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "Greek:" in combined and "Arabic:" in combined
        assert any(ord(ch) > 127 for ch in combined)
