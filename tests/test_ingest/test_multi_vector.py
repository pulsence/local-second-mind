"""
Tests for multi-vector summarization (Phase 14.1).

Validates that section and file summaries are generated, embedded,
and stored alongside regular chunks with the correct ``node_type``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from lsm.ingest.summarization import (
    SummaryChunk,
    extract_sections,
    generate_file_summary,
    generate_section_summaries,
    make_summary_chunk_id,
)


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------

class FakeLLMProvider:
    """Minimal LLM provider that returns canned summaries."""

    def __init__(self, response: str = "This is a summary."):
        self._response = response
        self.calls: List[str] = []

    def send_message(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return self._response


class FakeLLMConfig:
    """Stands in for an LLMConfig; used only to create a FakeLLMProvider."""

    def __init__(self, provider: FakeLLMProvider):
        self._provider = provider


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_positions(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build chunk_positions with heading_path entries."""
    positions: List[Dict[str, Any]] = []
    for sec in sections:
        positions.append({
            "start_char": sec["start"],
            "end_char": sec["end"],
            "heading": sec["heading"],
            "heading_path": sec.get("heading_path", [sec["heading"]]),
        })
    return positions


# ------------------------------------------------------------------
# Tests: extract_sections
# ------------------------------------------------------------------

def test_extract_sections_groups_by_heading_path():
    raw_text = "Introduction text here. " + "Details of section A. " + "Section B content goes here."
    positions = [
        {
            "start_char": 0,
            "end_char": 24,
            "heading": "Intro",
            "heading_path": ["Intro"],
        },
        {
            "start_char": 24,
            "end_char": 46,
            "heading": "Section A",
            "heading_path": ["Section A"],
        },
        {
            "start_char": 46,
            "end_char": 74,
            "heading": "Section B",
            "heading_path": ["Section B"],
        },
    ]

    sections = extract_sections(raw_text, positions)

    assert len(sections) == 3
    assert sections[0]["heading"] == "Intro"
    assert sections[1]["heading"] == "Section A"
    assert sections[2]["heading"] == "Section B"


def test_extract_sections_merges_same_heading():
    raw_text = "Chunk 1 of intro. Chunk 2 of intro. Next section."
    positions = [
        {
            "start_char": 0,
            "end_char": 18,
            "heading": "Intro",
            "heading_path": ["Intro"],
        },
        {
            "start_char": 18,
            "end_char": 36,
            "heading": "Intro",
            "heading_path": ["Intro"],
        },
        {
            "start_char": 36,
            "end_char": 50,
            "heading": "Next",
            "heading_path": ["Next"],
        },
    ]

    sections = extract_sections(raw_text, positions)

    assert len(sections) == 2
    assert "Chunk 1" in sections[0]["text"]
    assert "Chunk 2" in sections[0]["text"]


def test_extract_sections_empty_positions():
    sections = extract_sections("some text", [])
    assert sections == []

    sections = extract_sections("some text", None)
    assert sections == []


# ------------------------------------------------------------------
# Tests: generate_section_summaries
# ------------------------------------------------------------------

def test_section_summaries_generated(monkeypatch):
    fake_provider = FakeLLMProvider("Section summary generated.")
    fake_config = FakeLLMConfig(fake_provider)

    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        lambda cfg: fake_provider,
    )

    raw_text = "A" * 200 + "B" * 200
    positions = [
        {
            "start_char": 0,
            "end_char": 200,
            "heading": "Part A",
            "heading_path": ["Part A"],
        },
        {
            "start_char": 200,
            "end_char": 400,
            "heading": "Part B",
            "heading_path": ["Part B"],
        },
    ]

    summaries = generate_section_summaries(
        raw_text=raw_text,
        chunk_positions=positions,
        source_path="/docs/test.md",
        llm_config=fake_config,
    )

    assert len(summaries) == 2
    assert all(s.node_type == "section_summary" for s in summaries)
    assert summaries[0].metadata["heading"] == "Part A"
    assert summaries[1].metadata["heading"] == "Part B"
    assert len(fake_provider.calls) == 2


def test_section_summaries_skip_short_sections(monkeypatch):
    fake_provider = FakeLLMProvider("Summary.")
    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        lambda cfg: fake_provider,
    )

    raw_text = "Short."
    positions = [
        {
            "start_char": 0,
            "end_char": 6,
            "heading": "Tiny",
            "heading_path": ["Tiny"],
        },
    ]

    summaries = generate_section_summaries(
        raw_text=raw_text,
        chunk_positions=positions,
        source_path="/docs/test.md",
        llm_config=FakeLLMConfig(fake_provider),
    )

    assert len(summaries) == 0
    assert len(fake_provider.calls) == 0


# ------------------------------------------------------------------
# Tests: generate_file_summary
# ------------------------------------------------------------------

def test_file_summary_generated(monkeypatch):
    fake_provider = FakeLLMProvider("File-level summary text.")
    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        lambda cfg: fake_provider,
    )

    raw_text = "X" * 200
    positions = [
        {"start_char": 0, "end_char": 200, "heading": "Overview"},
    ]

    summary = generate_file_summary(
        raw_text=raw_text,
        chunk_positions=positions,
        source_path="/docs/test.md",
        llm_config=FakeLLMConfig(fake_provider),
    )

    assert summary is not None
    assert summary.node_type == "file_summary"
    assert summary.text == "File-level summary text."
    assert len(fake_provider.calls) == 1


def test_file_summary_skipped_for_short_text(monkeypatch):
    fake_provider = FakeLLMProvider("Summary.")
    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        lambda cfg: fake_provider,
    )

    summary = generate_file_summary(
        raw_text="Short.",
        chunk_positions=[],
        source_path="/docs/test.md",
        llm_config=FakeLLMConfig(fake_provider),
    )

    assert summary is None


def test_file_summary_returns_none_on_llm_failure(monkeypatch):
    def _fail(cfg):
        class FailProvider:
            def send_message(self, prompt, **kw):
                raise RuntimeError("LLM down")
        return FailProvider()

    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        _fail,
    )

    summary = generate_file_summary(
        raw_text="A" * 200,
        chunk_positions=[],
        source_path="/docs/test.md",
        llm_config=FakeLLMConfig(None),
    )

    assert summary is None


# ------------------------------------------------------------------
# Tests: node_type correctness
# ------------------------------------------------------------------

def test_node_type_set_correctly(monkeypatch):
    fake_provider = FakeLLMProvider("Summary text.")
    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        lambda cfg: fake_provider,
    )

    raw_text = "A" * 200
    positions = [
        {
            "start_char": 0,
            "end_char": 200,
            "heading": "Heading",
            "heading_path": ["Heading"],
        },
    ]

    section_summaries = generate_section_summaries(
        raw_text=raw_text,
        chunk_positions=positions,
        source_path="/test.md",
        llm_config=FakeLLMConfig(fake_provider),
    )
    file_summary = generate_file_summary(
        raw_text=raw_text,
        chunk_positions=positions,
        source_path="/test.md",
        llm_config=FakeLLMConfig(fake_provider),
    )

    assert section_summaries[0].node_type == "section_summary"
    assert section_summaries[0].metadata["node_type"] == "section_summary"
    assert file_summary.node_type == "file_summary"
    assert file_summary.metadata["node_type"] == "file_summary"


# ------------------------------------------------------------------
# Tests: make_summary_chunk_id
# ------------------------------------------------------------------

def test_summary_chunk_id_deterministic():
    id1 = make_summary_chunk_id("/test.md", "abc123", "section_summary", 0)
    id2 = make_summary_chunk_id("/test.md", "abc123", "section_summary", 0)
    assert id1 == id2

    id3 = make_summary_chunk_id("/test.md", "abc123", "file_summary", 0)
    assert id1 != id3


def test_summary_chunk_id_different_for_different_index():
    id1 = make_summary_chunk_id("/test.md", "abc123", "section_summary", 0)
    id2 = make_summary_chunk_id("/test.md", "abc123", "section_summary", 1)
    assert id1 != id2


# ------------------------------------------------------------------
# Tests: integration with ingest pipeline (parse_and_chunk_job)
# ------------------------------------------------------------------

def test_parse_and_chunk_job_with_summaries(tmp_path, monkeypatch):
    """Section and file summaries appended to chunks with node_type in positions."""
    from lsm.ingest.pipeline import parse_and_chunk_job

    doc = tmp_path / "test.md"
    doc.write_text("# Heading One\n\n" + "Content " * 50 + "\n\n# Heading Two\n\n" + "More content " * 50)

    fake_provider = FakeLLMProvider("Generated summary.")
    monkeypatch.setattr(
        "lsm.providers.factory.create_provider",
        lambda cfg: fake_provider,
    )

    result = parse_and_chunk_job(
        fp=doc,
        source_path=str(doc),
        mtime_ns=0,
        size=doc.stat().st_size,
        fhash="deadbeef",
        had_prev=False,
        enable_section_summaries=True,
        enable_file_summaries=True,
        summary_llm_config=FakeLLMConfig(fake_provider),
        chunking_strategy="structure",
    )

    assert result.ok
    # Should have regular chunks + section summaries + 1 file summary
    assert len(result.chunks) > 2

    # Check that some position entries have node_type
    summary_positions = [
        p for p in (result.chunk_positions or [])
        if p.get("node_type") in ("section_summary", "file_summary")
    ]
    assert len(summary_positions) >= 1

    # Check file_summary exists
    file_summary_positions = [
        p for p in (result.chunk_positions or [])
        if p.get("node_type") == "file_summary"
    ]
    assert len(file_summary_positions) == 1


# ------------------------------------------------------------------
# Tests: config fields
# ------------------------------------------------------------------

def test_ingest_config_summary_fields():
    from lsm.config.models.ingest import IngestConfig, RootConfig

    config = IngestConfig(
        roots=[RootConfig(path="/tmp/test")],
        enable_section_summaries=True,
        enable_file_summaries=True,
    )

    assert config.enable_section_summaries is True
    assert config.enable_file_summaries is True


def test_ingest_config_summary_fields_default():
    from lsm.config.models.ingest import IngestConfig, RootConfig

    config = IngestConfig(roots=[RootConfig(path="/tmp/test")])

    assert config.enable_section_summaries is False
    assert config.enable_file_summaries is False
