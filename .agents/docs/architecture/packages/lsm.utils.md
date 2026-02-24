# lsm.utils

Description: Shared utility modules for cross-cutting concerns: file graph analysis, text processing, logging, and path helpers.
Folder Path: `lsm/utils/`

## Modules

- [file_graph.py](../../lsm/utils/file_graph.py): Unified file graphing system — `GraphNode` schema, `FileGraph`, content-hash caching, and `get_file_graph()` entry point. Supports code files (Python, JS, TS, Java, Go, Rust, C/C++, C#), text/markdown/docx, PDF, and HTML.
- [text_processing.py](../../lsm/utils/text_processing.py): Shared text processing primitives — `HeadingInfo`, `Paragraph`, `split_paragraphs()`, `detect_heading()`, `is_list_block()`, and `extract_docx_text()`. Used by both `file_graph.py` and ingest parsers.
- [logger.py](../../lsm/utils/logger.py): Logger configuration helpers (`get_logger`).
- [paths.py](../../lsm/utils/paths.py): Path resolution utilities.

## Key Concepts

### GraphNode

`GraphNode` is a frozen dataclass that represents a structural element of any supported file type. All graphers (code, text, PDF, HTML) emit `GraphNode` instances with a common schema:

- `id` — deterministic, content-stable identifier
- `node_type` — discriminator: `function`, `class`, `import`, `block` (code); `heading`, `paragraph`, `list` (text/HTML/PDF)
- `start_line` / `end_line` — 1-based line span
- `start_char` / `end_char` — byte span within the file
- `depth` / `parent_id` / `children` — hierarchy
- `line_hash` — SHA-256 of the node's content lines; used by the Phase 4 edit engine to detect stale references
- `metadata` — type-specific extras (language, heading level, page number, etc.)

### FileGraph

`FileGraph` wraps the ordered node list with file-level metadata (content hash, file type). Serializes to/from dict via `to_dict()` / `from_dict()`.

### Caching

`get_file_graph(path)` caches by content hash (SHA-256 of file bytes). The cache is invalidated when file content changes, not by timestamp.

### Package Boundary

`lsm/utils/` has no dependency on `lsm/ingest/`, `lsm/agents/`, or `lsm/query/`. Those packages may import from `lsm/utils/`, not the reverse.
