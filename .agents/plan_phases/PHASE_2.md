# Phase 2: File Graphing System

**Why second:** File graphs are consumed by advanced tooling (Phase 4) and specific agents (Librarian, Manuscript Editor in Phase 6). Needs utils from Phase 1.

**Depends on:** Phase 1.1 (utils module)

| Task | Description | Depends On |
|------|-------------|------------|
| 2.1 | Graph schema and interfaces | 1.1 |
| 2.2 | Code file grapher | 2.1 |
| 2.3 | Text document grapher | 2.1 |
| 2.4 | PDF document grapher | 2.1 |
| 2.5 | HTML document grapher | 2.1 |
| 2.6 | Tool integration hooks | 2.2, 2.3, 2.4, 2.5 |
| 2.7 | Tests and fixtures | 2.2, 2.3, 2.4, 2.5 |

## 2.1: Graph Schema and Interfaces
- **Description:** Define the graph data model and public interfaces for structural output.
- **Tasks:**
  - Define node schema (type, name, span, depth, parent/child, metadata) and serialization format. Schema must support both code nodes (function, class, import, block) and text nodes (heading, paragraph, list). Unified `GraphNode` type with a `node_type` discriminator.
  - Specify deterministic ordering rules and stable IDs for node references.
  - Add caching strategy: hash file contents, cache graph by content hash. Invalidate on content change, not timestamp.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/agents/tools/source_map.py`
- **Success criteria:** Graph schema is documented and stable across runs for identical inputs.

## 2.2: Code File Grapher
- **Description:** Build a code-aware grapher that exposes functions/classes/blocks similar to tree-sitter outlines.
- **Tasks:**
  - Implement a parser strategy (tree-sitter when available, fallback heuristic parser otherwise).
  - Emit structural nodes with precise line/byte spans.
  - Normalize language-specific nodes into a common schema (function/class/block/import).
- **Files:**
  - `lsm/utils/file_graph.py`
- **Success criteria:** Code graphs expose accurate node boundaries across supported languages.

## 2.3: Text Document Grapher
- **Description:** Build a text grapher for markdown/plain text/docx headings and paragraphs.
- **Tasks:**
  - Parse headings and subheadings into a hierarchy.
  - Identify paragraph nodes with line spans under each heading.
  - Add docx parsing for section and paragraph extraction.
- **Package boundary note:** Common text processing logic (heading extraction, paragraph segmentation, section hierarchy building) should live in `lsm/utils/text_processing.py` — a new shared module. Both the ingest package (`structure_chunking.py`, parsers) and the file graph tools should import from this shared module. This avoids `lsm/utils/file_graph.py` depending on `lsm/ingest/` or vice versa. Shared data models (e.g., `TextSection`, `HeadingNode`) also belong in `lsm/utils/text_processing.py`.
- Existing `PageSegment` and `StructuredChunk` in `lsm/ingest/models.py` remain ingest-specific. The shared text processing module provides a parallel, graph-oriented representation that the ingest models can optionally wrap or convert from.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/utils/text_processing.py`
- **Success criteria:** Text graphs represent heading hierarchy and paragraph boundaries consistently.

## 2.4: PDF Document Grapher
- **Description:** Build a read-only grapher for PDF documents that exposes structural elements for navigation and content retrieval.
- **Tasks:**
  - Parse PDF page structure, headings, and text blocks into graph nodes using existing `parse_pdf` infrastructure.
  - Map page boundaries and section headings into the unified `GraphNode` schema.
  - Emit page-level and section-level nodes with content spans for targeted reading.
- **Package boundary note:** Reuse shared text processing logic from `lsm/utils/text_processing.py` (established in 2.3) for heading extraction and hierarchy building. Leverage `PageSegment` data from the existing PDF parser.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/utils/text_processing.py`
- **Success criteria:** PDF graphs expose page and section structure for read-only navigation. Headings and content blocks are accurately mapped.

## 2.5: HTML Document Grapher
- **Description:** Build a read-only grapher for HTML documents that exposes structural elements (headings, sections, lists, tables) for navigation and content retrieval.
- **Tasks:**
  - Parse HTML heading hierarchy (`h1`–`h6`), semantic sections (`<section>`, `<article>`), and content blocks into graph nodes.
  - Handle nested structures and map them into the unified `GraphNode` schema.
  - Emit structural nodes with content spans for targeted reading.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/utils/text_processing.py`
- **Success criteria:** HTML graphs expose heading hierarchy and semantic sections for read-only navigation. Nested structures are correctly represented.

## 2.6: Tool Integration Hooks
- **Description:** Expose graph outputs to tools for section-aware read/edit operations.
- **Tasks:**
  - Expose graph via a `get_file_graph(path) -> FileGraph` function in `lsm/utils/file_graph.py`. Tools call this function; they do not parse files themselves.
  - Ensure graph output can be requested per file and per section.
  - Line-hash generation: each `GraphNode` carries a `line_hash` computed from its content. This is consumed by the edit engine in Phase 4.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/agents/tools/source_map.py`
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/file_metadata.py`
- **Success criteria:** Tools can retrieve graph output without duplicating parsing logic.

## 2.7: Tests and Fixtures
- **Description:** Validate graph output determinism and section accuracy.
- **Tasks:**
  - Add fixtures for code, text, PDF, and HTML files with expected graph outputs.
  - Add tests for stable ordering, span correctness, and cache hits.
- **Files:**
  - `tests/test_tools/`
  - `tests/fixtures/`
- **Success criteria:** Graph outputs match fixtures and remain stable across runs.
