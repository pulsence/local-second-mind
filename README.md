# Local Second Mind (LSM)

Local Second Mind is a local-first RAG system for personal knowledge management.
It ingests local documents, retrieves relevant context, and generates cited answers
with configurable LLM providers.

## Caveat Emptor

This project is primarily for personal use. Pull requests or issues may not be
reviewed unless they overlap with the maintainer's active use cases. Further, unitl
this project reaches v1.0.0 expect breaking changes between release versions. This
will particularly affect the config file structure which I am changing organically
as feature are added and change.

## Core Goals

- Build and maintain a living local knowledge base.
- Support incremental ingest as files change.
- Provide semantic retrieval with optional reranking.
- Generate grounded answers with source citations.
- Keep corpus data local while using external LLM APIs only when configured.

## Current Architecture

```text
Local Files
  -> Parse -> Chunk -> Embed
  -> Vector Store (ChromaDB or PostgreSQL/pgvector)
  -> Retrieval + Optional Rerank + Synthesis
  -> Answer + Citations
```

## Features

### Ingest

- Recursive ingest from configured roots
- Parsers for PDF, DOCX, Markdown, HTML, and text
- Deterministic chunking and local embeddings
- Incremental ingest via manifest and file metadata
- Optional AI tagging for chunks
- Progress callback support in ingest APIs and UI

### Query

- Semantic retrieval from local vector store
- Configurable relevance thresholds and retrieval depth
- Optional lexical/LLM/hybrid rerank strategies
- Source-policy modes (`grounded`, `insight`, `hybrid`, custom modes)
- Optional remote source blending via provider framework
- Cited synthesis with fallback behavior when provider calls fail

### Interfaces

- Textual TUI for interactive query/ingest/settings workflows
- Single-shot CLI commands for ingest automation
- Structured config loading and validation (`config.json` or `config.yaml`)

## Installation

### Requirements

- Python 3.10+

### Install

```bash
pip install -e .
```

## Environment Setup

Store secrets in `.env` (not in config files):

```bash
cp .env.example .env
```

Common variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `BRAVE_API_KEY`

See `.env.example` for the full list.

## CLI Usage

Default entrypoint:

```bash
lsm
```

Equivalent module form:

```bash
python -m lsm
```

### TUI (interactive)

Run `lsm` with no subcommand. Querying is done in the TUI Query tab.

### Single-shot ingest commands

```bash
lsm ingest build [--dry-run] [--force] [--skip-errors]
lsm ingest tag [--max N]
lsm ingest wipe --confirm
```

Global options:

```bash
lsm --config path/to/config.json
lsm --verbose
lsm --log-level DEBUG
lsm --log-file logs/lsm.log
```

## Configuration

Configuration is loaded from `config.json` or `config.yaml`.

Minimal working example:

```json
{
  "roots": ["C:/Users/You/Documents"],
  "vectordb": {
    "provider": "chromadb",
    "persist_dir": ".chroma",
    "collection": "local_kb"
  },
  "llms": [
    {
      "provider_name": "openai",
      "query": { "model": "gpt-5.2" }
    }
  ]
}
```

Important schema notes:

- Vector DB settings live under `vectordb` (`persist_dir`, `collection`, `provider`).
- LLM config is an ordered `llms` list with feature-level settings (`query`, `tagging`, `ranking`).
- Query behavior is configured under `query`.
- Mode/source-policy behavior is configured under `modes`.
- Remote integrations are configured under `remote_providers`.
- Notes are configured globally under top-level `notes`.
- Optional top-level `global_folder` controls default app data location.

For full configuration reference, see `docs/user-guide/CONFIGURATION.md`.

## Typical Workflow

1. Copy `example_config.json` to `config.json` and adjust paths/models.
2. Add API keys to `.env` as needed.
3. Build your collection: `lsm ingest build`.
4. Launch TUI: `lsm`.
5. Query from the Query tab and save notes if desired.

## OCR (Optional)

If enabling OCR for image-based PDFs, install the Tesseract executable and add
it to `PATH`. `pytesseract` is only a Python wrapper.

## Documentation

- `docs/user-guide/GETTING_STARTED.md`
- `docs/user-guide/CLI_USAGE.md`
- `docs/user-guide/CONFIGURATION.md`
- `docs/architecture/OVERVIEW.md`
- `docs/development/CHANGELOG.md`

## License

See `LICENSE`.
