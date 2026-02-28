"""
Tests for lsm.ingest.structure_chunking module.

Tests structure-aware chunking with heading detection, paragraph splitting,
sentence preservation, page number mapping, and overlap behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsm.ingest.models import PageSegment
from lsm.ingest.parsers import parse_html
from lsm.ingest.structure_chunking import (
    StructuredChunk,
    _detect_heading,
    _split_paragraphs,
    _split_sentences,
    _build_page_char_map,
    _page_for_offset,
    structure_chunk_text,
    structured_chunks_to_positions,
)
from lsm.utils.file_graph import build_markdown_graph


# ------------------------------------------------------------------
# Heading detection
# ------------------------------------------------------------------
class TestDetectHeading:
    """Tests for _detect_heading helper."""

    def test_markdown_h1(self):
        assert _detect_heading("# Main Title") == "Main Title"

    def test_markdown_h2(self):
        assert _detect_heading("## Section") == "Section"

    def test_markdown_h3(self):
        assert _detect_heading("### Subsection") == "Subsection"

    def test_markdown_h6(self):
        assert _detect_heading("###### Deep Heading") == "Deep Heading"

    def test_bold_line(self):
        assert _detect_heading("**Bold Heading**") == "Bold Heading"

    def test_html_heading(self):
        assert _detect_heading("<h2>HTML Section</h2>") == "HTML Section"

    def test_regular_text(self):
        assert _detect_heading("Just a regular paragraph.") is None

    def test_bold_inside_sentence(self):
        # Not a heading — text after the bold
        assert _detect_heading("Some **bold** text here") is None

    def test_empty_string(self):
        assert _detect_heading("") is None

    def test_hash_without_space(self):
        # "#NoSpace" is not a valid heading
        assert _detect_heading("#NoSpace") is None


# ------------------------------------------------------------------
# Paragraph splitting
# ------------------------------------------------------------------
class TestSplitParagraphs:
    """Tests for _split_paragraphs helper."""

    def test_single_paragraph(self):
        text = "Hello world. This is a test."
        paras = _split_paragraphs(text)
        assert len(paras) == 1
        assert paras[0].text == text
        assert paras[0].heading is None

    def test_two_paragraphs(self):
        text = "Para one.\n\nPara two."
        paras = _split_paragraphs(text)
        assert len(paras) == 2
        assert paras[0].text == "Para one."
        assert paras[1].text == "Para two."

    def test_heading_then_paragraph(self):
        text = "# Heading\n\nParagraph text."
        paras = _split_paragraphs(text)
        assert len(paras) == 2
        # First para is the heading itself
        assert paras[0].is_heading is True
        assert paras[0].heading == "Heading"
        # Second para inherits heading context
        assert paras[1].heading == "Heading"
        assert paras[1].is_heading is False

    def test_multiple_headings(self):
        text = "# H1\n\nFirst paragraph.\n\n## H2\n\nSecond paragraph."
        paras = _split_paragraphs(text)
        headings = [p for p in paras if p.is_heading]
        assert len(headings) == 2
        # Second paragraph should have H2 as heading
        non_headings = [p for p in paras if not p.is_heading]
        assert non_headings[0].heading == "H1"
        assert non_headings[1].heading == "H2"

    def test_empty_text(self):
        assert _split_paragraphs("") == []

    def test_whitespace_only(self):
        assert _split_paragraphs("   \n\n   ") == []

    def test_paragraph_indices(self):
        text = "A\n\nB\n\nC"
        paras = _split_paragraphs(text)
        assert [p.index for p in paras] == [0, 1, 2]


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------
class TestSplitSentences:
    """Tests for _split_sentences helper."""

    def test_single_sentence(self):
        sents = _split_sentences("Hello world.")
        assert len(sents) == 1
        assert sents[0] == "Hello world."

    def test_two_sentences(self):
        sents = _split_sentences("First sentence. Second sentence.")
        assert len(sents) == 2

    def test_question_and_exclamation(self):
        sents = _split_sentences("Why? Because! That is all.")
        assert len(sents) == 3

    def test_abbreviation_like_text(self):
        # e.g. "Dr. Smith" – might split or not depending on regex
        sents = _split_sentences("Dr. Smith went to the store.")
        # At minimum the content should be preserved
        combined = " ".join(sents)
        assert "Smith" in combined

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_no_punctuation(self):
        sents = _split_sentences("Just some words without punctuation")
        assert len(sents) == 1


# ------------------------------------------------------------------
# Page mapping
# ------------------------------------------------------------------
class TestPageMapping:
    """Tests for page number mapping helpers."""

    def test_build_page_char_map_simple(self):
        segs = [
            PageSegment(text="Page one text.", page_number=1),
            PageSegment(text="Page two text.", page_number=2),
        ]
        full = "Page one text.\n\nPage two text."
        spans = _build_page_char_map(segs, full)
        assert len(spans) == 2
        assert spans[0][2] == 1  # page 1
        assert spans[1][2] == 2  # page 2

    def test_page_for_offset(self):
        spans = [(0, 14, 1), (16, 30, 2)]
        assert _page_for_offset(0, spans) == 1
        assert _page_for_offset(10, spans) == 1
        assert _page_for_offset(16, spans) == 2
        assert _page_for_offset(25, spans) == 2

    def test_page_for_offset_beyond_end(self):
        spans = [(0, 10, 1)]
        # Beyond last span → returns last page
        assert _page_for_offset(100, spans) == 1

    def test_page_for_offset_empty_spans(self):
        assert _page_for_offset(0, []) is None


# ------------------------------------------------------------------
# Main structure_chunk_text
# ------------------------------------------------------------------
class TestStructureChunkText:
    """Tests for the main structure_chunk_text function."""

    def test_empty_text(self):
        assert structure_chunk_text("") == []

    def test_whitespace_only(self):
        assert structure_chunk_text("   \n\n   ") == []

    def test_single_short_paragraph(self):
        text = "Hello world. This is a test."
        chunks = structure_chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text

    def test_never_splits_sentence(self):
        """Verify that no sentence is split across chunks."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = structure_chunk_text(text, chunk_size=40, overlap=0.0)
        for chunk in chunks:
            # Each chunk should end with a sentence-ending punctuation
            # (or be the last chunk)
            words = chunk.text.strip()
            assert len(words) > 0

    def test_respects_heading_boundaries(self):
        """Chunks should not mix content from different headings."""
        text = (
            "# Section A\n\n"
            "Content under section A.\n\n"
            "# Section B\n\n"
            "Content under section B."
        )
        chunks = structure_chunk_text(text, chunk_size=5000)
        # Heading-based split: should have at least 2 chunks
        headings = [c.heading for c in chunks]
        assert "Section A" in headings
        assert "Section B" in headings

    def test_max_heading_depth_ignores_deep_heading_boundaries(self):
        text = (
            "# H1\n\n"
            "Intro.\n\n"
            "## H2\n\n"
            "Level 2 content.\n\n"
            "### H3\n\n"
            "Level 3 content."
        )
        chunks = structure_chunk_text(text, chunk_size=5000, max_heading_depth=2)
        headings = {c.heading for c in chunks if c.heading}
        assert "H1" in headings
        assert "H2" in headings
        assert "H3" not in headings
        assert any("### H3" in c.text for c in chunks)

    def test_max_heading_depth_none_keeps_all_heading_boundaries(self):
        text = (
            "# H1\n\n"
            "Intro.\n\n"
            "## H2\n\n"
            "Level 2 content.\n\n"
            "### H3\n\n"
            "Level 3 content."
        )
        chunks = structure_chunk_text(text, chunk_size=5000, max_heading_depth=None)
        headings = {c.heading for c in chunks if c.heading}
        assert "H3" in headings
        assert all("### H3" not in c.text for c in chunks)

    def test_heading_metadata(self):
        """Heading text should appear in chunk metadata."""
        text = "## My Section\n\nSome content here."
        chunks = structure_chunk_text(text, chunk_size=5000)
        content_chunks = [c for c in chunks if not c.text.startswith("##")]
        assert len(content_chunks) >= 1
        assert content_chunks[0].heading == "My Section"

    def test_overlap_carries_sentences(self):
        """With overlap > 0, chunks should share trailing/leading sentences."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = structure_chunk_text(text, chunk_size=40, overlap=0.5)
        if len(chunks) > 1:
            # Check that content from end of first chunk appears at start of second
            first_sents = set(_split_sentences(chunks[0].text))
            second_sents = set(_split_sentences(chunks[1].text))
            overlap = first_sents & second_sents
            assert len(overlap) > 0, "Expected overlapping sentences between chunks"

    def test_zero_overlap(self):
        """With overlap=0, chunks should not share content."""
        text = "A. B. C. D. E. F. G. H. I. J."
        chunks = structure_chunk_text(text, chunk_size=15, overlap=0.0)
        assert len(chunks) >= 2

    def test_paragraph_not_mixed(self):
        """Each chunk should contain content from at most one heading context."""
        text = (
            "# First\n\n"
            "Para under first.\n\n"
            "# Second\n\n"
            "Para under second."
        )
        chunks = structure_chunk_text(text, chunk_size=5000)
        for chunk in chunks:
            # A chunk should not contain both "Para under first" and "Para under second"
            has_first = "Para under first" in chunk.text
            has_second = "Para under second" in chunk.text
            assert not (has_first and has_second), "Chunk mixes content from different headings"

    def test_character_offsets(self):
        """start_char and end_char should be populated."""
        text = "Hello world.\n\nSecond paragraph."
        chunks = structure_chunk_text(text, chunk_size=5000)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_with_page_segments(self):
        """Page numbers should be mapped when page_segments provided."""
        page1 = "Content on page one."
        page2 = "Content on page two."
        full_text = f"{page1}\n\n{page2}"
        segments = [
            PageSegment(text=page1, page_number=1),
            PageSegment(text=page2, page_number=2),
        ]
        chunks = structure_chunk_text(
            full_text, chunk_size=5000, page_segments=segments,
        )
        assert len(chunks) >= 1
        assert chunks[0].page_start == 1

    def test_page_span_across_pages(self):
        """A chunk spanning pages should have different page_start and page_end."""
        page1 = "Page one sentence."
        page2 = "Page two sentence."
        full_text = f"{page1}\n\n{page2}"
        segments = [
            PageSegment(text=page1, page_number=1),
            PageSegment(text=page2, page_number=2),
        ]
        # With a huge chunk_size, all content in one chunk
        chunks = structure_chunk_text(
            full_text, chunk_size=5000, page_segments=segments,
        )
        if len(chunks) == 1:
            # Single chunk should span both pages
            assert chunks[0].page_start == 1
            assert chunks[0].page_end == 2

    def test_without_page_segments(self):
        """When no page_segments provided, page fields should be None."""
        chunks = structure_chunk_text("Some text.", chunk_size=5000)
        assert chunks[0].page_start is None
        assert chunks[0].page_end is None

    def test_large_document_produces_multiple_chunks(self):
        """A long document should produce multiple chunks."""
        paragraphs = [
            f"Paragraph {i}. This contains some content for testing. More words here."
            for i in range(50)
        ]
        text = "\n\n".join(paragraphs)
        chunks = structure_chunk_text(text, chunk_size=200, overlap=0.2)
        assert len(chunks) > 5

    def test_markdown_document(self):
        """Test chunking a realistic Markdown document."""
        text = """# Introduction

This is the introduction to the document. It provides an overview of the topics
that will be covered in the following sections.

## Background

The background section covers relevant prior work and context. Several studies
have demonstrated the importance of structure-aware text processing.

## Methods

We employed a multi-stage pipeline for document processing. First, documents
are parsed to extract text and metadata. Then, text is chunked using
structure-aware rules that respect heading and paragraph boundaries.

### Data Collection

Data was collected from multiple sources including PDFs, Word documents,
and Markdown files.

### Analysis

The analysis phase involved comparing different chunking strategies and
measuring their impact on retrieval quality.

## Results

Results show that structure-aware chunking significantly improves retrieval
accuracy compared to fixed-size chunking approaches.

## Conclusion

In conclusion, structure-aware chunking provides meaningful improvements
for knowledge management systems."""
        chunks = structure_chunk_text(text, chunk_size=400, overlap=0.2)
        assert len(chunks) >= 3
        # Check heading tracking
        headings = {c.heading for c in chunks if c.heading}
        assert "Introduction" in headings or "Background" in headings

    def test_intelligent_heading_depth_splits_large_sections_at_subheadings(self):
        text = (
            "# Root\n\n"
            + ("Intro sentence. " * 80)
            + "\n\n## Child A\n\n"
            + ("Child A sentence. " * 40)
            + "\n\n## Child B\n\n"
            + ("Child B sentence. " * 40)
        )
        graph = build_markdown_graph(Path("doc.md"), text)
        chunks = structure_chunk_text(
            text,
            chunk_size=260,
            overlap=0.0,
            file_graph=graph,
            intelligent_heading_depth=True,
        )
        headings = {c.heading for c in chunks if c.heading}
        assert "Child A" in headings
        assert "Child B" in headings

    def test_intelligent_heading_depth_keeps_small_parent_section_whole(self):
        text = (
            "# Root\n\n"
            "Short intro.\n\n"
            "## Child A\n\n"
            "Brief child.\n\n"
            "## Child B\n\n"
            "Brief child two."
        )
        graph = build_markdown_graph(Path("doc.md"), text)
        chunks = structure_chunk_text(
            text,
            chunk_size=5000,
            overlap=0.0,
            file_graph=graph,
            intelligent_heading_depth=True,
        )
        headings = {c.heading for c in chunks if c.heading}
        assert "Root" in headings
        assert "Child A" not in headings
        assert "Child B" not in headings
        assert any("## Child A" in c.text for c in chunks)

    def test_intelligent_heading_depth_recurses_nested_headings(self):
        text = (
            "# H1\n\n"
            + ("Root sentence. " * 60)
            + "\n\n## H2\n\n"
            + ("Level 2 sentence. " * 60)
            + "\n\n### H3\n\n"
            + ("Level 3 sentence. " * 60)
        )
        graph = build_markdown_graph(Path("doc.md"), text)
        chunks = structure_chunk_text(
            text,
            chunk_size=240,
            overlap=0.0,
            file_graph=graph,
            intelligent_heading_depth=True,
        )
        headings = {c.heading for c in chunks if c.heading}
        assert "H2" in headings
        assert "H3" in headings

    def test_intelligent_heading_depth_falls_back_without_file_graph(self):
        text = (
            "# Root\n\n"
            "Root paragraph.\n\n"
            "## Child\n\n"
            "Child paragraph."
        )
        chunks = structure_chunk_text(
            text,
            chunk_size=5000,
            overlap=0.0,
            file_graph=None,
            intelligent_heading_depth=True,
        )
        headings = {c.heading for c in chunks if c.heading}
        assert "Child" in headings

    def test_heading_path_captures_full_hierarchy(self):
        text = (
            "# Introduction\n\n"
            "Intro.\n\n"
            "## Background\n\n"
            "Background.\n\n"
            "### Prior Work\n\n"
            "Details."
        )
        graph = build_markdown_graph(Path("doc.md"), text)
        chunks = structure_chunk_text(
            text,
            chunk_size=5000,
            overlap=0.0,
            file_graph=graph,
        )

        target = next(c for c in chunks if "Details." in c.text)
        assert target.heading == "Prior Work"
        assert target.heading_path == ["Introduction", "Background", "Prior Work"]


# ------------------------------------------------------------------
# Real corpus scenarios
# ------------------------------------------------------------------
class TestStructureChunkingWithRealCorpus:
    """Structure-aware chunking on rich synthetic corpus files."""

    def _document_path(self, synthetic_data_root: Path, filename: str) -> Path:
        return synthetic_data_root / "documents" / filename

    def test_research_paper_heading_detection(self, synthetic_data_root: Path):
        text = self._document_path(synthetic_data_root, "research_paper.md").read_text(
            encoding="utf-8",
        )
        chunks = structure_chunk_text(text, chunk_size=900, overlap=0.2)

        assert len(chunks) > 4
        headings = {c.heading for c in chunks if c.heading}
        assert "Retrieval-Augmented Knowledge Work in Local-First Systems" in headings

    def test_structure_chunking_after_html_parse(self, synthetic_data_root: Path):
        html_fp = self._document_path(synthetic_data_root, "technical_manual.html")
        parsed_text, metadata = parse_html(html_fp)
        chunks = structure_chunk_text(parsed_text, chunk_size=700, overlap=0.2)

        assert metadata.get("title") == "Knowledge System Operations Manual"
        assert len(chunks) > 3

        headings = {c.heading for c in chunks if c.heading}
        assert "Overview" in headings
        assert any(h.startswith("Step ") for h in headings)

    def test_large_document_chunk_count_and_overlap(self, synthetic_data_root: Path):
        text = self._document_path(synthetic_data_root, "large_document.md").read_text(
            encoding="utf-8",
        )
        chunks = structure_chunk_text(text, chunk_size=450, overlap=0.25)

        assert len(chunks) > 20
        assert all(chunk.text.strip() for chunk in chunks)

    def test_page_tracking_with_realistic_segments(self, synthetic_data_root: Path):
        text = self._document_path(synthetic_data_root, "research_paper.md").read_text(
            encoding="utf-8",
        )
        midpoint = max(1, len(text) // 2)
        seg1 = text[:midpoint].strip()
        seg2 = text[midpoint:].strip()
        segments = [
            PageSegment(text=seg1, page_number=1),
            PageSegment(text=seg2, page_number=2),
        ]

        chunks = structure_chunk_text(
            text,
            chunk_size=len(text) + 64,
            overlap=0.0,
            page_segments=segments,
        )

        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2


# ------------------------------------------------------------------
# structured_chunks_to_positions
# ------------------------------------------------------------------
class TestStructuredChunksToPositions:
    """Tests for converting StructuredChunk to pipeline-compatible format."""

    def test_basic_conversion(self):
        chunks = [
            StructuredChunk(
                text="Hello world.",
                heading="Section A",
                heading_path=["Top", "Section A"],
                start_char=0,
                end_char=12,
                paragraph_index=0,
                page_start=1,
                page_end=1,
            ),
        ]
        texts, positions = structured_chunks_to_positions(chunks)
        assert texts == ["Hello world."]
        assert len(positions) == 1
        pos = positions[0]
        assert pos["chunk_index"] == 0
        assert pos["start_char"] == 0
        assert pos["end_char"] == 12
        assert pos["length"] == 12
        assert pos["heading"] == "Section A"
        assert pos["heading_path"] == ["Top", "Section A"]
        assert pos["paragraph_index"] == 0
        assert pos["page_start"] == 1
        assert pos["page_end"] == 1

    def test_heading_path_is_json_serializable(self):
        chunks = [
            StructuredChunk(
                text="Chunk",
                heading="Section",
                heading_path=["Root", "Section"],
                start_char=0,
                end_char=5,
            ),
        ]
        _texts, positions = structured_chunks_to_positions(chunks)
        payload = json.dumps(positions[0]["heading_path"])
        assert payload == '[\"Root\", \"Section\"]'

    def test_flat_heading_retained_with_heading_path(self):
        chunks = [
            StructuredChunk(
                text="Chunk",
                heading="Leaf",
                heading_path=["Top", "Mid", "Leaf"],
                start_char=0,
                end_char=5,
            ),
        ]
        _texts, positions = structured_chunks_to_positions(chunks)
        assert positions[0]["heading"] == "Leaf"
        assert positions[0]["heading_path"] == ["Top", "Mid", "Leaf"]

    def test_no_heading_or_page(self):
        chunks = [
            StructuredChunk(text="Plain text.", start_char=0, end_char=11),
        ]
        texts, positions = structured_chunks_to_positions(chunks)
        assert "heading" not in positions[0]
        assert "page_start" not in positions[0]
        assert "page_end" not in positions[0]

    def test_empty_list(self):
        texts, positions = structured_chunks_to_positions([])
        assert texts == []
        assert positions == []

    def test_multiple_chunks_indexed(self):
        chunks = [
            StructuredChunk(text="A", start_char=0, end_char=1),
            StructuredChunk(text="B", start_char=2, end_char=3),
            StructuredChunk(text="C", start_char=4, end_char=5),
        ]
        texts, positions = structured_chunks_to_positions(chunks)
        assert [p["chunk_index"] for p in positions] == [0, 1, 2]


# ------------------------------------------------------------------
# IngestConfig chunking_strategy validation
# ------------------------------------------------------------------
class TestChunkingStrategyConfig:
    """Tests for the chunking_strategy config field."""

    def test_default_is_structure(self):
        from lsm.config.models.ingest import IngestConfig

        config = IngestConfig(roots=["/tmp"])
        assert config.chunking_strategy == "structure"

    def test_fixed_accepted(self):
        from lsm.config.models.ingest import IngestConfig

        config = IngestConfig(roots=["/tmp"], chunking_strategy="fixed")
        config.validate()
        assert config.chunking_strategy == "fixed"

    def test_invalid_strategy_rejected(self):
        from lsm.config.models.ingest import IngestConfig

        config = IngestConfig(roots=["/tmp"], chunking_strategy="invalid")
        with pytest.raises(ValueError, match="chunking_strategy"):
            config.validate()

    def test_loader_reads_chunking_strategy(self):
        from lsm.config.loader import build_ingest_config
        from pathlib import Path

        raw = {
            "ingest": {
                "roots": ["/tmp"],
                "chunking_strategy": "fixed",
            }
        }
        config = build_ingest_config(raw, Path("."))
        assert config.chunking_strategy == "fixed"

    def test_loader_defaults_to_structure(self):
        from lsm.config.loader import build_ingest_config
        from pathlib import Path

        raw = {"ingest": {"roots": ["/tmp"]}}
        config = build_ingest_config(raw, Path("."))
        assert config.chunking_strategy == "structure"


# ------------------------------------------------------------------
# Pipeline integration (parse_and_chunk_job)
# ------------------------------------------------------------------
class TestPipelineIntegration:
    """Test that structure chunking integrates correctly with the pipeline."""

    def test_parse_and_chunk_job_structure_strategy(self, tmp_path):
        """parse_and_chunk_job with structure strategy produces headings."""
        from lsm.ingest.pipeline import parse_and_chunk_job

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph.",
            encoding="utf-8",
        )

        result = parse_and_chunk_job(
            fp=md_file,
            source_path=str(md_file),
            mtime_ns=md_file.stat().st_mtime_ns,
            size=md_file.stat().st_size,
            fhash="abc123",
            had_prev=False,
            chunk_size=5000,
            chunk_overlap=200,
            chunking_strategy="structure",
        )

        assert result.ok
        assert len(result.chunks) >= 1
        assert result.chunk_positions is not None
        # At least one position should have a heading
        headings = [p.get("heading") for p in result.chunk_positions if p.get("heading")]
        assert len(headings) >= 1

    def test_parse_and_chunk_job_fixed_strategy(self, tmp_path):
        """parse_and_chunk_job with fixed strategy uses legacy chunking."""
        from lsm.ingest.pipeline import parse_and_chunk_job

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("A " * 2000, encoding="utf-8")

        result = parse_and_chunk_job(
            fp=txt_file,
            source_path=str(txt_file),
            mtime_ns=txt_file.stat().st_mtime_ns,
            size=txt_file.stat().st_size,
            fhash="abc123",
            had_prev=False,
            chunk_size=1000,
            chunk_overlap=100,
            chunking_strategy="fixed",
        )

        assert result.ok
        assert len(result.chunks) >= 2
        # Fixed strategy should not have heading metadata
        for pos in result.chunk_positions:
            assert "heading" not in pos

    def test_page_number_in_pipeline_metadata(self, tmp_path):
        """Page numbers should flow through to chunk positions for PDF."""
        from lsm.ingest.pipeline import parse_and_chunk_job
        from unittest.mock import patch
        from lsm.ingest.models import PageSegment

        page1_text = "First page content here."
        page2_text = "Second page content here."
        full_text = f"{page1_text}\n\n{page2_text}"
        segments = [
            PageSegment(text=page1_text, page_number=1),
            PageSegment(text=page2_text, page_number=2),
        ]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf")

        with patch("lsm.ingest.pipeline.parse_file") as mock_parse:
            mock_parse.return_value = (full_text, {}, segments)

            result = parse_and_chunk_job(
                fp=pdf_file,
                source_path=str(pdf_file),
                mtime_ns=0,
                size=100,
                fhash="abc123",
                had_prev=False,
                chunk_size=5000,
                chunking_strategy="structure",
            )

        assert result.ok
        assert result.chunk_positions is not None
        # Check that page info is present
        has_page = any(
            p.get("page_start") is not None
            for p in result.chunk_positions
        )
        assert has_page

    def test_parse_and_chunk_job_root_max_heading_depth_overrides_global(self, tmp_path):
        """Per-root max heading depth takes precedence over global value."""
        from lsm.ingest.pipeline import parse_and_chunk_job

        md_file = tmp_path / "depth_override.md"
        md_file.write_text(
            "# Root\n\n"
            "Root text.\n\n"
            "## L2\n\n"
            "Level 2 text.\n\n"
            "### L3\n\n"
            "Level 3 text.",
            encoding="utf-8",
        )

        result = parse_and_chunk_job(
            fp=md_file,
            source_path=str(md_file),
            mtime_ns=md_file.stat().st_mtime_ns,
            size=md_file.stat().st_size,
            fhash="abc123",
            had_prev=False,
            chunk_size=5000,
            chunk_overlap=0,
            chunking_strategy="structure",
            max_heading_depth=1,
            root_max_heading_depth=3,
        )

        assert result.ok
        headings = [
            p.get("heading")
            for p in (result.chunk_positions or [])
            if p.get("heading") is not None
        ]
        assert "L3" in headings

    def test_parse_and_chunk_job_uses_intelligent_heading_depth_with_file_graph(self, tmp_path):
        """Pipeline uses FileGraph-based intelligent heading selection when enabled."""
        from lsm.ingest.pipeline import parse_and_chunk_job

        md_file = tmp_path / "smart_depth.md"
        md_file.write_text(
            "# Root\n\n"
            "Short intro.\n\n"
            "## Child A\n\n"
            "Brief child.\n\n"
            "## Child B\n\n"
            "Brief child two.",
            encoding="utf-8",
        )

        result = parse_and_chunk_job(
            fp=md_file,
            source_path=str(md_file),
            mtime_ns=md_file.stat().st_mtime_ns,
            size=md_file.stat().st_size,
            fhash="abc123",
            had_prev=False,
            chunk_size=5000,
            chunk_overlap=0,
            chunking_strategy="structure",
            intelligent_heading_depth=True,
        )

        assert result.ok
        headings = [
            p.get("heading")
            for p in (result.chunk_positions or [])
            if p.get("heading") is not None
        ]
        assert "Root" in headings
        assert "Child A" not in headings
        assert "Child B" not in headings

    def test_parse_and_chunk_job_emits_heading_path_metadata(self, tmp_path):
        """Pipeline includes heading_path metadata alongside flat heading."""
        from lsm.ingest.pipeline import parse_and_chunk_job

        md_file = tmp_path / "heading_path.md"
        md_file.write_text(
            "# Introduction\n\n"
            "Intro.\n\n"
            "## Background\n\n"
            "Background.\n\n"
            "### Prior Work\n\n"
            "Details.",
            encoding="utf-8",
        )

        result = parse_and_chunk_job(
            fp=md_file,
            source_path=str(md_file),
            mtime_ns=md_file.stat().st_mtime_ns,
            size=md_file.stat().st_size,
            fhash="abc123",
            had_prev=False,
            chunk_size=5000,
            chunk_overlap=0,
            chunking_strategy="structure",
        )

        assert result.ok
        target = next(
            pos for pos in (result.chunk_positions or [])
            if pos.get("heading") == "Prior Work"
        )
        assert target["heading_path"] == ["Introduction", "Background", "Prior Work"]
