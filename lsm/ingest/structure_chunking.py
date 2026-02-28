"""
Structure-aware text chunking.

Splits text into chunks respecting document structure: headings, paragraphs,
and sentence boundaries.  Never splits a sentence across two chunks, never
mixes paragraphs, and preserves heading context in chunk metadata.

When ``PageSegment`` data is available (PDF / DOCX), each chunk carries the
page range it spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from lsm.config.models.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from lsm.ingest.models import PageSegment
from lsm.utils.text_processing import detect_heading as _detect_heading_info
from lsm.utils.text_processing import Paragraph as _Paragraph
from lsm.utils.text_processing import split_paragraphs as _split_paragraphs_base


# ------------------------------------------------------------------
# Dataclass for structured chunk output
# ------------------------------------------------------------------
@dataclass
class StructuredChunk:
    """A single chunk produced by structure-aware chunking."""

    text: str
    """Chunk text content."""

    heading: Optional[str] = None
    """Most recent heading above this chunk (if any)."""

    start_char: int = 0
    """Start character offset in the full document text."""

    end_char: int = 0
    """End character offset in the full document text."""

    paragraph_index: Optional[int] = None
    """Index of the first paragraph included in this chunk."""

    page_start: Optional[int] = None
    """1-based page number where this chunk starts (PDF/DOCX only)."""

    page_end: Optional[int] = None
    """1-based page number where this chunk ends (PDF/DOCX only)."""


# Sentence-boundary regex – handles common abbreviations gracefully.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])"   # lookbehind for sentence-ending punctuation
    r"(?:\s+)"      # whitespace after punctuation
    r"(?=[A-Z\u00C0-\u024F\d\"\'\(\[])"  # lookahead for uppercase / digit / quote
)


# ------------------------------------------------------------------
# Heading detection + paragraph splitting now live in utils.text_processing
# ------------------------------------------------------------------
def _detect_heading(line: str) -> Optional[str]:
    info = _detect_heading_info(line, allow_plain_headings=False)
    return info.text if info else None


def _split_paragraphs(text: str) -> List[_Paragraph]:
    return _split_paragraphs_base(text, allow_plain_headings=False)


def _is_heading_boundary(
    paragraph: _Paragraph,
    max_heading_depth: Optional[int],
) -> bool:
    """Return True when paragraph should be treated as a heading boundary."""
    if not paragraph.is_heading:
        return False
    if max_heading_depth is None:
        return True
    level = paragraph.heading_level or 1
    return level <= max_heading_depth


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences, preserving each sentence intact."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


# ------------------------------------------------------------------
# Page mapping
# ------------------------------------------------------------------
def _build_page_char_map(
    page_segments: List[PageSegment],
    full_text: str,
) -> List[tuple[int, int, int]]:
    """Build a list of ``(start_char, end_char, page_number)`` spans.

    Maps character offsets in *full_text* back to page numbers based on
    the original ``PageSegment`` data.
    """
    spans: List[tuple[int, int, int]] = []
    offset = 0
    for seg in page_segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue
        # Find this segment's text in the full text starting from offset
        start = full_text.find(seg_text, offset)
        if start == -1:
            # Fallback: use current offset
            start = offset
        end = start + len(seg_text)
        spans.append((start, end, seg.page_number))
        offset = end
    return spans


def _page_for_offset(
    offset: int,
    page_spans: List[tuple[int, int, int]],
) -> Optional[int]:
    """Return the 1-based page number for a character *offset*."""
    for start, end, page_num in page_spans:
        if start <= offset < end:
            return page_num
    # Beyond last span – return last page
    if page_spans:
        return page_spans[-1][2]
    return None


# ------------------------------------------------------------------
# Main chunking function
# ------------------------------------------------------------------
def structure_chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: float = 0.25,
    page_segments: Optional[List[PageSegment]] = None,
    max_heading_depth: Optional[int] = None,
    track_positions: bool = True,
) -> List[StructuredChunk]:
    """
    Chunk text using structure-aware rules.

    Chunking rules:
    - NEVER split a sentence across two chunks.
    - NEVER mix paragraphs (a chunk contains complete paragraphs or
      whole-sentence groups from a single paragraph).
    - NEVER mix headings: each heading boundary starts a new chunk and
      the heading text is stored in chunk metadata.
    - Overlap is achieved by repeating trailing sentences from the
      previous chunk at the start of the next chunk (configurable,
      default 25% of ``chunk_size``).

    Args:
        text: Full document text to chunk.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap ratio (0.0–1.0) controlling how many characters
            from the end of one chunk are repeated at the start of the
            next.  Defaults to 0.25 (25%).
        page_segments: Optional page-level segments from the parser for
            mapping chunks back to page numbers.
        max_heading_depth: Optional maximum heading depth treated as a chunk
            boundary. Headings deeper than this value are folded into parent
            section body text.
        track_positions: If True, populate character offsets and page
            numbers on each ``StructuredChunk``.

    Returns:
        List of ``StructuredChunk`` instances.
    """
    if not text or not text.strip():
        return []

    overlap_chars = int(chunk_size * max(0.0, min(1.0, overlap)))

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # Build page mapping if segments provided
    page_spans: Optional[List[tuple[int, int, int]]] = None
    if page_segments and track_positions:
        page_spans = _build_page_char_map(page_segments, text)

    chunks: List[StructuredChunk] = []

    # Accumulator for current chunk
    current_sentences: List[str] = []
    current_len = 0
    current_heading: Optional[str] = None
    active_boundary_heading: Optional[str] = None
    current_para_index: Optional[int] = None
    current_start_char: int = 0

    def _flush() -> None:
        """Emit the accumulated sentences as a chunk."""
        nonlocal current_sentences, current_len, current_start_char

        if not current_sentences:
            return

        chunk_text_str = " ".join(current_sentences)
        if not chunk_text_str.strip():
            current_sentences = []
            current_len = 0
            return

        end_char = current_start_char + len(chunk_text_str)

        page_start: Optional[int] = None
        page_end: Optional[int] = None
        if page_spans:
            page_start = _page_for_offset(current_start_char, page_spans)
            page_end = _page_for_offset(max(current_start_char, end_char - 1), page_spans)

        chunks.append(StructuredChunk(
            text=chunk_text_str,
            heading=current_heading,
            start_char=current_start_char,
            end_char=end_char,
            paragraph_index=current_para_index,
            page_start=page_start,
            page_end=page_end,
        ))

        current_sentences = []
        current_len = 0

    for para in paragraphs:
        # Heading paragraphs at/above max depth start a new chunk.
        if _is_heading_boundary(para, max_heading_depth):
            _flush()
            active_boundary_heading = para.heading
            current_heading = active_boundary_heading
            current_start_char = para.start_char
            current_para_index = para.index
            continue

        effective_heading = active_boundary_heading

        # If heading changed (e.g. paragraph under a new heading), flush.
        if effective_heading != current_heading:
            _flush()
            current_heading = effective_heading
            current_start_char = para.start_char
            current_para_index = para.index

        sentences = _split_sentences(para.text)
        if not sentences:
            continue

        for sent in sentences:
            sent_len = len(sent)

            # Would adding this sentence exceed chunk_size?
            added_len = sent_len + (1 if current_sentences else 0)
            if current_len + added_len > chunk_size and current_sentences:
                # Flush current chunk
                _flush()

                # Overlap: carry back trailing sentences from the flushed chunk
                if overlap_chars > 0 and chunks:
                    prev_text = chunks[-1].text
                    prev_sents = _split_sentences(prev_text)
                    overlap_sents: List[str] = []
                    overlap_len = 0
                    for s in reversed(prev_sents):
                        if overlap_len + len(s) + 1 > overlap_chars:
                            break
                        overlap_sents.insert(0, s)
                        overlap_len += len(s) + 1

                    if overlap_sents:
                        current_sentences = overlap_sents
                        current_len = sum(len(s) for s in overlap_sents) + len(overlap_sents) - 1

                # Update start_char from the paragraph
                # Approximate: use paragraph start + offset of this sentence
                sent_offset = para.text.find(sent)
                if sent_offset >= 0:
                    current_start_char = para.start_char + sent_offset
                else:
                    current_start_char = para.start_char
                current_para_index = para.index

            current_sentences.append(sent)
            current_len += added_len
            if current_para_index is None:
                current_para_index = para.index
            if not current_sentences or current_len == added_len:
                current_start_char = para.start_char

    # Flush remaining
    _flush()

    return chunks


# ------------------------------------------------------------------
# Convenience: convert StructuredChunks to pipeline-compatible format
# ------------------------------------------------------------------
def structured_chunks_to_positions(
    chunks: List[StructuredChunk],
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Convert ``StructuredChunk`` list to ``(texts, positions)`` tuple.

    The positions dicts are compatible with the existing pipeline metadata
    format used by ``chunk_text()``, with additional fields for heading,
    page_start, and page_end.

    Returns:
        Tuple of (chunk texts, position dicts).
    """
    texts: List[str] = []
    positions: List[Dict[str, Any]] = []

    for idx, sc in enumerate(chunks):
        texts.append(sc.text)
        pos: Dict[str, Any] = {
            "chunk_index": idx,
            "start_char": sc.start_char,
            "end_char": sc.end_char,
            "length": len(sc.text),
        }
        if sc.heading is not None:
            pos["heading"] = sc.heading
        if sc.paragraph_index is not None:
            pos["paragraph_index"] = sc.paragraph_index
        if sc.page_start is not None:
            pos["page_start"] = sc.page_start
        if sc.page_end is not None:
            pos["page_end"] = sc.page_end
        positions.append(pos)

    return texts, positions
