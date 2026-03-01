"""
Multi-vector summarization for ingest pipeline.

Generates LLM-based summaries at section and file granularities.
Each summary is embedded and stored alongside regular chunks with a
distinguishing ``node_type`` (``section_summary`` or ``file_summary``).

**Cost note**: Each summary requires one LLM call. For a large corpus
with many heading sections, enabling ``enable_section_summaries`` can
be expensive. Enable selectively on high-value collections.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class SummaryChunk:
    """A summary chunk to be embedded and stored."""

    text: str
    """The summary text."""

    node_type: str
    """'section_summary' or 'file_summary'."""

    heading_path: Optional[List[str]] = None
    """Heading hierarchy for section summaries."""

    source_path: Optional[str] = None
    """Source file path."""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata to merge into the chunk."""


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_SECTION_SUMMARY_PROMPT = (
    "Summarize the following section of a document in 2-4 sentences. "
    "Focus on the key concepts and main points.\n\n"
    "Section heading: {heading}\n\n"
    "Section text:\n{text}\n\n"
    "Summary:"
)

_FILE_SUMMARY_PROMPT = (
    "Summarize the following document in 3-6 sentences. "
    "Cover the main topics, key arguments, and conclusions.\n\n"
    "Document outline:\n{outline}\n\n"
    "Document text (first portion):\n{text}\n\n"
    "Summary:"
)


# ------------------------------------------------------------------
# Section summary extraction
# ------------------------------------------------------------------

def extract_sections(
    raw_text: str,
    chunk_positions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Extract heading-based sections from chunked text.

    Groups consecutive chunks sharing the same ``heading_path`` into
    sections suitable for summarization.

    Returns a list of dicts with keys:
    - ``heading``: the section heading text
    - ``heading_path``: list of heading hierarchy strings
    - ``text``: concatenated chunk texts for this section
    """
    if not chunk_positions:
        return []

    sections: List[Dict[str, Any]] = []
    current_heading_path: Optional[str] = None
    current_section: Optional[Dict[str, Any]] = None

    for pos in chunk_positions:
        heading = pos.get("heading")
        heading_path_raw = pos.get("heading_path")
        start = pos.get("start_char", 0)
        end = pos.get("end_char", 0)

        if heading_path_raw is None:
            continue

        # Normalize heading_path to a comparable key
        if isinstance(heading_path_raw, list):
            hp_key = json.dumps(heading_path_raw)
            hp_list = heading_path_raw
        elif isinstance(heading_path_raw, str):
            try:
                hp_list = json.loads(heading_path_raw)
                hp_key = heading_path_raw
            except (json.JSONDecodeError, TypeError):
                hp_list = [heading_path_raw]
                hp_key = json.dumps(hp_list)
        else:
            continue

        chunk_text = raw_text[start:end] if raw_text and end > start else ""

        if hp_key != current_heading_path:
            if current_section and current_section["text"].strip():
                sections.append(current_section)
            current_heading_path = hp_key
            current_section = {
                "heading": heading or (hp_list[-1] if hp_list else ""),
                "heading_path": hp_list,
                "text": chunk_text,
            }
        elif current_section is not None:
            current_section["text"] += "\n" + chunk_text

    if current_section and current_section["text"].strip():
        sections.append(current_section)

    return sections


def _build_file_outline(
    chunk_positions: Optional[List[Dict[str, Any]]],
) -> str:
    """Build a heading outline from chunk positions."""
    if not chunk_positions:
        return "(no headings)"

    seen: List[str] = []
    for pos in chunk_positions:
        heading = pos.get("heading")
        if heading and heading not in seen:
            seen.append(heading)

    if not seen:
        return "(no headings)"

    return "\n".join(f"- {h}" for h in seen)


# ------------------------------------------------------------------
# Summary generation
# ------------------------------------------------------------------

def generate_section_summaries(
    raw_text: str,
    chunk_positions: Optional[List[Dict[str, Any]]],
    source_path: str,
    llm_config: Any,
    max_section_chars: int = 8000,
) -> List[SummaryChunk]:
    """Generate summary chunks for each heading section.

    Args:
        raw_text: Full document text.
        chunk_positions: Position info from chunking (with heading_path).
        source_path: Canonical source path for the file.
        llm_config: An LLMConfig instance for the summarization provider.
        max_section_chars: Maximum section text length to send to LLM.

    Returns:
        List of SummaryChunk with node_type='section_summary'.
    """
    sections = extract_sections(raw_text, chunk_positions)
    if not sections:
        return []

    from lsm.providers.factory import create_provider

    provider = create_provider(llm_config)
    summaries: List[SummaryChunk] = []

    for section in sections:
        section_text = section["text"][:max_section_chars]
        heading = section["heading"]
        heading_path = section["heading_path"]

        if len(section_text.strip()) < 50:
            continue

        prompt = _SECTION_SUMMARY_PROMPT.format(
            heading=heading,
            text=section_text,
        )

        try:
            summary_text = provider.send_message(prompt)
            if not summary_text or not summary_text.strip():
                continue

            summaries.append(SummaryChunk(
                text=summary_text.strip(),
                node_type="section_summary",
                heading_path=heading_path,
                source_path=source_path,
                metadata={
                    "heading": heading,
                    "heading_path": json.dumps(heading_path) if heading_path else None,
                    "node_type": "section_summary",
                },
            ))
        except Exception as exc:
            logger.warning(
                "Failed to generate section summary for '%s' in %s: %s",
                heading, source_path, exc,
            )

    return summaries


def generate_file_summary(
    raw_text: str,
    chunk_positions: Optional[List[Dict[str, Any]]],
    source_path: str,
    llm_config: Any,
    max_text_chars: int = 8000,
) -> Optional[SummaryChunk]:
    """Generate a single file-level summary.

    Args:
        raw_text: Full document text.
        chunk_positions: Position info from chunking (for outline).
        source_path: Canonical source path for the file.
        llm_config: An LLMConfig instance for the summarization provider.
        max_text_chars: Maximum text portion to send to LLM.

    Returns:
        A SummaryChunk with node_type='file_summary', or None on failure.
    """
    if not raw_text or len(raw_text.strip()) < 50:
        return None

    from lsm.providers.factory import create_provider

    provider = create_provider(llm_config)

    outline = _build_file_outline(chunk_positions)
    text_portion = raw_text[:max_text_chars]

    prompt = _FILE_SUMMARY_PROMPT.format(
        outline=outline,
        text=text_portion,
    )

    try:
        summary_text = provider.send_message(prompt)
        if not summary_text or not summary_text.strip():
            return None

        return SummaryChunk(
            text=summary_text.strip(),
            node_type="file_summary",
            source_path=source_path,
            metadata={
                "node_type": "file_summary",
            },
        )
    except Exception as exc:
        logger.warning(
            "Failed to generate file summary for %s: %s",
            source_path, exc,
        )
        return None


def make_summary_chunk_id(
    source_path: str,
    file_hash: str,
    node_type: str,
    index: int = 0,
) -> str:
    """Generate a deterministic chunk ID for a summary node."""
    raw = f"{source_path}:{file_hash}:{node_type}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
