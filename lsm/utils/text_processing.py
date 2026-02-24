from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_HTML_HEADING_RE = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE)
_BOLD_LINE_RE = re.compile(r"^\*\*(.+)\*\*$")
_SECTION_HEADING_RE = re.compile(
    r"^(section|chapter|part)\s+(\d+(?:\.\d+)*)\s*[:.)-]?\s+(.+)$",
    re.IGNORECASE,
)

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class HeadingInfo:
    text: str
    level: int


@dataclass(frozen=True)
class Paragraph:
    """Internal representation of a parsed paragraph."""

    text: str
    heading: Optional[str]
    index: int
    start_char: int
    end_char: int
    is_heading: bool = False
    heading_level: Optional[int] = None
    start_line: int = 1
    end_line: int = 1


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def detect_heading(line: str, allow_plain_headings: bool = False) -> Optional[HeadingInfo]:
    stripped = line.strip()
    if not stripped:
        return None

    match = _MD_HEADING_RE.match(stripped)
    if match:
        level = len(match.group(1))
        return HeadingInfo(text=match.group(2).strip(), level=level)

    match = _HTML_HEADING_RE.match(stripped)
    if match:
        level = int(match.group(1))
        heading_text = re.sub(r"<[^>]+>", "", match.group(2))
        heading_text = re.sub(r"\s+", " ", heading_text).strip()
        if heading_text:
            return HeadingInfo(text=heading_text, level=level)

    match = _BOLD_LINE_RE.match(stripped)
    if match:
        return HeadingInfo(text=match.group(1).strip(), level=1)

    if allow_plain_headings:
        match = _SECTION_HEADING_RE.match(stripped)
        if match:
            number = match.group(2)
            level = min(6, max(1, number.count(".") + 1))
            return HeadingInfo(text=stripped, level=level)

    return None


def is_list_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return all(_LIST_ITEM_RE.match(line) for line in lines)


def _line_offsets(text: str) -> Tuple[List[str], List[int]]:
    lines = text.split("\n")
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1
    return lines, offsets


def _line_for_offset(offsets: List[int], offset: int) -> int:
    if not offsets:
        return 1
    idx = bisect_right(offsets, max(offset, 0)) - 1
    if idx < 0:
        return 1
    return idx + 1


def split_paragraphs(text: str, allow_plain_headings: bool = False) -> List[Paragraph]:
    normalized = normalize_newlines(text)
    lines, offsets = _line_offsets(normalized)

    paragraphs: List[Paragraph] = []
    current_heading: Optional[str] = None
    current_heading_level: Optional[int] = None
    idx = 0

    raw_blocks = re.split(r"\n\s*\n", normalized)

    char_offset = 0
    for block in raw_blocks:
        block_stripped = block.strip()
        if not block_stripped:
            char_offset += len(block) + 2
            continue

        start = normalized.find(block_stripped, char_offset)
        if start == -1:
            start = char_offset
        end = start + len(block_stripped)

        heading_info = detect_heading(block_stripped, allow_plain_headings=allow_plain_headings)
        if heading_info:
            current_heading = heading_info.text
            current_heading_level = heading_info.level
            is_heading = True
        else:
            is_heading = False

        start_line = _line_for_offset(offsets, start)
        end_line = _line_for_offset(offsets, max(end - 1, start))

        paragraphs.append(
            Paragraph(
                text=block_stripped,
                heading=current_heading,
                index=idx,
                start_char=start,
                end_char=end,
                is_heading=is_heading,
                heading_level=current_heading_level if is_heading else None,
                start_line=start_line,
                end_line=end_line,
            )
        )

        idx += 1
        char_offset = end

    return paragraphs


def _docx_heading_level(paragraph) -> Optional[int]:
    try:
        style_name = (paragraph.style.name or "").strip().lower()
    except Exception:
        style_name = ""

    if not style_name.startswith("heading"):
        return None

    parts = style_name.split()
    if len(parts) < 2:
        return None

    try:
        level = int(parts[1])
    except ValueError:
        return None

    if 1 <= level <= 6:
        return level
    return None


def extract_docx_text(path: Path) -> str:
    doc = Document(str(path))
    lines: List[str] = []

    for para in doc.paragraphs:
        if not para.text:
            continue
        heading_level = _docx_heading_level(para)
        if heading_level is not None:
            lines.append(f"{'#' * heading_level} {para.text}")
            lines.append("")
        else:
            lines.append(para.text)
            lines.append("")

    return "\n".join(lines).strip()
