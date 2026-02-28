# Phase 7: FileGraph Heading Enhancements

**Status**: Completed

Adds configurable heading depth, intelligent heading selection via FileGraph, and
`heading_path` metadata. This phase is independent of the retrieval pipeline and can
proceed in parallel with Phases 5-6.

Reference: [RESEARCH_PLAN.md §4.1, §4.2, §4.3](../RESEARCH_PLAN.md#41-document-headings-configurable-depth)

---

## 7.1: Configurable Heading Depth

**Description**: Add `max_heading_depth` config field. Headings deeper than this value
are not treated as chunk boundaries.

**Tasks**:
- Add `max_heading_depth: Optional[int] = None` to `IngestConfig`
- Add `max_heading_depth: Optional[int] = None` to `RootConfig` (per-root override)
- Update `lsm/ingest/structure_chunking.py`:
  - `structure_chunk_text()` respects `max_heading_depth` — headings at depth >
    `max_heading_depth` flow into the parent chunk body
- Update `example_config.json` with new field
- Update config loader tests

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/config/models/ingest.py` — `max_heading_depth` field
- `lsm/ingest/structure_chunking.py` — depth filtering
- `example_config.json` — new field
- `tests/test_ingest/test_structure_chunking.py`:
  - Test: `max_heading_depth=2` — H3 and deeper are not chunk boundaries
  - Test: `max_heading_depth=None` — all headings are boundaries (existing behavior)
  - Test: per-root override takes precedence over global

**Success criteria**: Heading depth filtering works. Deeper headings flow into parent chunks.
Per-root overrides work.

---

## 7.2: Intelligent Heading Selection via FileGraph

**Description**: Add `intelligent_heading_depth` config flag. When enabled, heading depth is
decided dynamically based on section size relative to `chunk_size`, using `FileGraph`.

**Tasks**:
- Add `intelligent_heading_depth: bool = False` to `IngestConfig`
- Update `lsm/ingest/structure_chunking.py`:
  - Add optional `file_graph: Optional[FileGraph] = None` parameter to
    `structure_chunk_text()`
  - When `file_graph` is provided and `intelligent_heading_depth` is `True`:
    - Walk the heading tree from `FileGraph.root_ids`
    - For each heading node: compute section size from `end_char - start_char`
    - If size ≤ `chunk_size`: keep as one chunk; sub-headings flow into body
    - If size > `chunk_size` AND has children of type `"heading"`: split at child
      heading boundaries, recurse
    - Recursion depth bounded by `max_heading_depth` (if set)
  - When `file_graph` is not provided: use existing regex-based heading detection
- Update ingest pipeline to generate `FileGraph` for supported formats and pass it
  to `structure_chunk_text()`:
  - Markdown → `build_markdown_graph()`
  - HTML → `build_html_graph()`
  - DOCX → `build_docx_graph()`
  - Text → `build_text_graph()`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/config/models/ingest.py` — `intelligent_heading_depth` field
- `lsm/ingest/structure_chunking.py` — FileGraph-based chunking
- `lsm/ingest/pipeline.py` — FileGraph generation and passing
- `tests/test_ingest/test_structure_chunking.py`:
  - Test: large section with sub-headings splits at sub-heading boundaries
  - Test: small section with sub-headings keeps as one chunk
  - Test: deeply nested headings recurse correctly
  - Test: fallback to regex when no FileGraph provided

**Success criteria**: Intelligent heading selection produces right-sized chunks. Large
sections are split at sub-heading boundaries. Small sections are kept whole.

---

## 7.3: heading_path Metadata

**Description**: Add `heading_path` metadata to chunks — a JSON array recording the full
heading hierarchy.

**Tasks**:
- Update `structure_chunk_text()` to produce `heading_path` for each chunk:
  - Walk `parent_id` chain from the heading `GraphNode` up to document root
  - Collect `name` fields, reverse to root-to-leaf order
  - Store as JSON array: `["Introduction", "Background", "Prior Work"]`
- Update ingest pipeline to store `heading_path` in chunk metadata
- Update `lsm_chunks` schema (already has `heading_path TEXT` column from Phase 1)
- Retain flat `heading` string for BM25 indexing
- Update `Candidate` dataclass to include `heading_path` property

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/ingest/structure_chunking.py` — heading_path construction
- `lsm/ingest/pipeline.py` — metadata storage
- `lsm/query/session.py` — `Candidate.heading_path` property
- `tests/test_ingest/test_structure_chunking.py`:
  - Test: heading_path correctly captures hierarchy
  - Test: heading_path is JSON-serializable
  - Test: flat heading field is retained alongside heading_path

**Success criteria**: Chunks carry `heading_path` metadata with full hierarchy. Both
`heading` (flat string) and `heading_path` (JSON array) are stored.

---

## 7.4: Phase 7 Code Review and Changelog

**Tasks**:
- Review FileGraph integration for performance (graph building per file)
- Review heading_path construction for edge cases (no headings, single heading, deeply nested)
- Review tests: confirm real FileGraph operations, not mocks
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md` — document heading config options
- Update `.agents/docs/architecture/development/INGEST.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`
- `.agents/docs/architecture/development/INGEST.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---
