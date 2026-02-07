# Ingest Pipeline Architecture

This document describes how LSM ingests documents into the local vector store.

## Goals

- Parse heterogeneous file types into text.
- Chunk text into stable, overlapping segments.
- Embed chunks locally.
- Store vectors and metadata in ChromaDB.
- Track incremental updates with a manifest.

## High-Level Flow

```
roots -> scan -> parse -> normalize -> chunk -> embed -> write -> manifest
```

## Core Components

### File Discovery

- Implemented in `lsm/ingest/fs.py`.
- Recursively scans each `roots` directory.
- Applies `extensions` allowlist and `exclude_dirs` filter.
- Emits filesystem paths for processing.

### Parsing

- Implemented in `lsm/ingest/parsers.py`.
- Supported formats: `.txt`, `.md`, `.rst`, `.pdf`, `.docx`, `.html`, `.htm`.
- Extracts metadata where possible:
  - PDF: author, title, creation dates
  - DOCX: core properties (author, title, keywords)
  - Markdown: YAML frontmatter
  - HTML: title, author, description

### OCR Support

- Optional in `parse_pdf()` when `enable_ocr` is true.
- Uses PyMuPDF to render pages and pytesseract for OCR.
- OCR is only used for pages that appear image-based.

### Chunking

Two chunking strategies are available, controlled by `chunking_strategy` in
`IngestConfig`:

#### Structure-Aware Chunking (default, `"structure"`)

- Implemented in `lsm/ingest/structure_chunking.py`.
- Respects document structure: headings, paragraphs, and sentence boundaries.
- **Never splits a sentence** across two chunks.
- **Never mixes paragraphs** — each chunk contains complete paragraphs or
  whole-sentence groups from a single paragraph.
- **Never mixes headings** — heading boundaries start new chunks and the heading
  text is stored in chunk metadata.
- Overlap is achieved by repeating trailing sentences from the previous chunk
  (configurable via `chunk_overlap` as a proportion of `chunk_size`).
- Detects headings: Markdown `#` headings and bold-only lines (`**Heading**`).

#### Fixed-Size Chunking (`"fixed"`)

- Implemented in `lsm/ingest/chunking.py`.
- Simple character-based sliding window with overlap.
- Default `chunk_size`: 1800 characters.
- Default `chunk_overlap`: 200 characters.
- Tracks start and end character offsets per chunk.

### Page Number Tracking

- PDF and DOCX parsers return `PageSegment` objects preserving page boundaries.
- PDF pages are tracked via PyMuPDF's page-by-page extraction.
- DOCX page breaks are detected via `<w:lastRenderedPageBreak/>` and
  `<w:br w:type="page"/>` elements in the document XML.
- When a chunk spans multiple pages, metadata stores the range as
  `"START-END"` (e.g., `"3-4"`).
- Non-paginated formats (MD, HTML, TXT) omit page number metadata.

### Embedding

- Uses `sentence-transformers` locally.
- The model and device are configured in `IngestConfig`.
- Embeddings are normalized to match query-time similarity.

### Storage

- Implemented in `lsm/ingest/chroma_store.py`.
- Vectors are written in batches to ChromaDB.
- Chunk IDs are derived from `source_path`, `file_hash`, and chunk index.

### Manifest

- Stored at `manifest` (default `.ingest/manifest.json`).
- Tracks `mtime_ns`, `size`, and `file_hash` for each file.
- Enables fast skip of unchanged files.

## Pipeline Execution Details

The ingest pipeline in `lsm/ingest/pipeline.py` uses a threaded architecture:

- Thread pool for parsing (CPU and I/O bound).
- Single embedding worker (GPU-friendly batching).
- Single writer thread (exclusive Chroma ownership).

Data flow:

1. Files are scanned and compared against the manifest.
2. Unchanged files are skipped early.
3. Changed files are parsed and chunked.
4. Chunk batches are embedded on the worker.
5. Embeddings are written to Chroma in flush batches.
6. The manifest is updated only after successful writes.

## Error Handling

- `skip_errors` controls whether parsing failures abort the run.
- Page-level PDF errors are recorded in `ingest_error_report.json`.
- Empty or unparsable files are skipped but logged.

## Metadata Stored per Chunk

Each chunk includes:

- `source_path`
- `source_name`
- `ext`
- `mtime_ns`
- `file_hash`
- `chunk_index`
- `ingested_at`
- `start_char` / `end_char` / `chunk_length`
- `heading` — most recent heading above the chunk (structure chunking only)
- `paragraph_index` — index of the first paragraph in the chunk (structure chunking only)
- `page_number` — page number or range e.g. `"3"` or `"3-4"` (PDF/DOCX only)
- Optional metadata (title, author, tags)

## Incremental Updates

The manifest enables three fast checks:

1. If `mtime` and `size` match, skip hashing.
2. If file hash matches, update manifest and skip re-embedding.
3. If hash differs, re-ingest the file and replace its chunks.

## AI Tagging Integration

- AI tagging is supported via `lsm/ingest/tagging.py`.
- Tags are stored in chunk metadata.
- Tagging can be triggered in the TUI Ingest tab via `/tag`.

## Ingest Commands (TUI)

The TUI Ingest tab provides management commands for:

- collection stats and exploration
- re-ingest (`/build`)
- tagging (`/tag`)
- wiping the collection (`/wipe`)

See `docs/api-reference/REPL.md` for commands.
