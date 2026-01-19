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

- Implemented in `lsm/ingest/chunking.py`.
- Default `chunk_size`: 1800 characters.
- Default `chunk_overlap`: 200 characters.
- Tracks start and end character offsets per chunk.

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
