# Ingest Pipeline Architecture

This document describes how LSM ingests documents into the local vector store.

## Goals

- Parse heterogeneous file types into normalized text.
- Chunk text into stable, structure-aware segments.
- Embed chunks locally.
- Persist vectors and metadata in the unified SQLite/PostgreSQL storage layer.
- Track incremental and versioned ingest state in database tables.

## High-Level Flow

```
roots -> scan -> parse -> normalize -> chunk -> embed -> write -> schema/version tracking
```

## Core Components

### File Discovery

- Implemented in `lsm/ingest/fs.py`.
- Recursively scans each configured ingest root.
- Applies `extensions` allowlist and `exclude_dirs` filter.
- Emits `(file_path, root_config)` tuples, enabling per-root metadata/overrides.

### Parsing

- Implemented in `lsm/ingest/parsers.py`.
- Supported formats: `.txt`, `.md`, `.rst`, `.pdf`, `.docx`, `.html`, `.htm`.
- Extracts metadata where available:
  - PDF: author/title/dates and optional OCR fallback
  - DOCX: core properties and page-break-aware text extraction
  - Markdown: YAML frontmatter
  - HTML: title/author/description and heading-preserving text conversion

### Chunking

Chunking strategy is selected by `ingest.chunking_strategy`.

#### Structure-Aware Chunking (`"structure"`, default)

- Implemented in `lsm/ingest/structure_chunking.py`.
- Preserves structure and metadata:
  - sentence boundaries
  - heading context
  - paragraph/page offsets
- Supports heading depth controls:
  - `ingest.max_heading_depth` (global)
  - `ingest.roots[].max_heading_depth` (per-root override)
- Supports adaptive FileGraph-driven splitting:
  - `ingest.intelligent_heading_depth`
  - uses section-size-aware recursion to decide whether child headings become boundaries
  - falls back to regex/paragraph heading detection when no FileGraph is available
- Emits both:
  - `heading` (flat string)
  - `heading_path` (root-to-leaf heading hierarchy)

#### Fixed-Size Chunking (`"fixed"`)

- Implemented in `lsm/ingest/chunking.py`.
- Character-window chunking with overlap.
- Tracks positional offsets but does not use heading-aware boundary logic.

### FileGraph Integration

- FileGraph builders are in `lsm/utils/file_graph.py`.
- Ingest builds text-aligned graphs for supported structure formats:
  - `build_markdown_graph`
  - `build_html_graph`
  - `build_docx_graph`
  - `build_text_graph`
- Graphs are used by structure chunking for:
  - intelligent boundary selection
  - full `heading_path` metadata construction

### Page Number Tracking

- PDF and DOCX parsers return `PageSegment` records.
- Structure chunking maps chunk character offsets back to page ranges.
- Metadata stores `page_number` as either a single page or range (`"3"`, `"3-4"`).

### Embedding

- Uses `sentence-transformers` locally.
- Embeddings are generated in GPU-friendly batches.
- Embeddings are normalized for query-time similarity usage.

### Storage

- Vector DB provider abstraction: `lsm/vectordb/base.py`.
- Default provider: `lsm/vectordb/sqlite_vec.py` (SQLite + sqlite-vec).
- Application schema tables are owned by `lsm.db.schema`.
- Chunk writes include both content and metadata fields (`heading`, `heading_path`, offsets, version fields, etc.).

### Manifest and Schema Versioning

- Manifest state is database-backed (`lsm_manifest`), not JSON sidecar files.
- Schema/version provenance is tracked in `lsm_schema_versions`.
- Incremental ingest uses mtime/size/hash fast-path checks plus schema compatibility checks.
- Re-ingest versions are tracked via `version` and `is_current` metadata.

## Pipeline Execution Details

`lsm/ingest/pipeline.py` uses a threaded architecture:

- parse thread pool (I/O + CPU parse work)
- single embedding worker (batch-optimized)
- single writer thread (serialized DB/vector writes)

Write behavior:

- staged chunks are flushed in batches
- manifest updates are applied only after successful writes
- SQLite path supports transactional chunk + manifest consistency

## Error Handling

- `skip_errors` controls whether parse failures are fatal.
- Per-page parse errors are captured in `ingest_error_report.json`.
- Empty/unparsable documents are skipped with error reporting in result summaries.

## Metadata Stored per Chunk

Each chunk can include:

- `source_path`, `source_name`, `ext`
- `mtime_ns`, `file_hash`, `chunk_index`, `ingested_at`
- `start_char`, `end_char`, `chunk_length`
- `heading` (flat heading)
- `heading_path` (JSON array hierarchy)
- `paragraph_index`
- `page_number`
- optional document metadata (title/author/tags/content_type)
- versioning metadata (`version`, `is_current`)

## AI Tagging Integration

- Implemented in `lsm/ingest/tagging.py`.
- Uses provider transport calls (`send_message`) with ingest-owned prompt/schema assets.
- Tags are merged into chunk metadata.

## Ingest Commands

Primary command paths include:

- `lsm ingest build`
- `lsm ingest build --force-reingest-changed-config`
- `lsm ingest build --force-file-pattern <glob>`
- `lsm db complete`
- `lsm db prune`

See `.agents/docs/architecture/api-reference/REPL.md` for shell/TUI command details.
