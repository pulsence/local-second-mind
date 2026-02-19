# Agent Documentation Index

This document is the entry point for Claude/Codex agents working on Local Second Mind. For detailed architecture and file documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Project Overview

Local-first RAG system for personal knowledge management. Semantic search + LLM synthesis from local documents (PDF, DOCX, MD, HTML, TXT).

- **Language:** Python 3.10+
- **Entry:** `python -m lsm`
- **Config:** `config.json` or `config.yaml`

## Architecture Overview

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed package structure and key files reference.

### Packages

- [lsm.agents](./architecture/lsm.agents.md) - Agent framework, tools, memory
- [lsm.config](./architecture/lsm.config.md) - Configuration loading and models
- [lsm.ingest](./architecture/lsm.ingest.md) - Document parsing and embedding
- [lsm.providers](./architecture/lsm.providers.md) - LLM provider implementations
- [lsm.query](./architecture/lsm.query.md) - Query retrieval and synthesis
- [lsm.remote](./architecture/lsm.remote.md) - Remote source providers
- [lsm.ui](./architecture/lsm.ui.md) - User interfaces (TUI, Shell, Web)
- [lsm.vectordb](./architecture/lsm.vectordb.md) - Vector database providers


## Developer Guides

- [CREATE_PLAN.md](./CREATE_PLAN.md) - How to create task plans
- [COMMIT_MESSAGE.md](./COMMIT_MESSAGE.md) - Commit message format
- [CODING_PATTERNS.md](./CODING_PATTERNS.md) - Coding conventions

## Commands

```bash
# Run all tests
.venv-wsl/bin/python -m pytest tests/ -v

# Run with coverage
.venv-wsl/bin/python -m pytest tests/ -v --cov=lsm --cov-report=html

# CLI
.venv-wsl/bin/python -m lsm --help
.venv-wsl/bin/python -m lsm ingest --config config.json
.venv-wsl/bin/python -m lsm query --config config.json
.venv-wsl/bin/python -m lsm  # Interactive unified shell
```

## Environment Variables

API keys go in `.env`, never in config files:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- `BRAVE_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `CORE_API_KEY`
- `LSM_POSTGRES_CONNECTION_STRING`, `LSM_POSTGRES_TABLE` (PostgreSQL vector DB)
- See `.env.example` for full list

## Key Implementation Notes

- **Virtual environment:** Run project commands with `.venv-win\Scripts\python` (Windows PowerShell) to ensure dependencies match the project environment.
- **Config loading:** Raw dict -> dataclass construction -> validation
- **Config structure:** Zero flat top-level fields. All settings nested under section objects: `"global"`, `"ingest"`, `"vectordb"`, `"llms"`, `"query"`, `"modes"`, `"notes"`, `"remote_providers"`, `"agents"`.
- **Global config:** `embed_model`, `device`, `batch_size`, `global_folder` live in `GlobalConfig` (read from `"global"` section).
- **Ingest config:** All ingest-only fields (`roots`, `manifest`, `chunk_size`, etc.) read from `"ingest"` section.
- **Vector DB config:** Use `vectordb.persist_dir` and `vectordb.collection`; top-level fallback fields are removed.
- **Agent memory config:** `agents.memory.storage_backend` controls backend selection (`auto`, `sqlite`, `postgresql`).
- **Agent harness memory injection:** `AgentHarness` can build standing memory context before each LLM call.
- **Agent run summaries:** `AgentHarness` emits run summaries with tool usage, approvals/denials, artifacts, outcome, duration, and token usage.
- **Agent scheduler:** Stores runtime schedule metadata in `<agents_folder>/schedules.json`.
- **Meta-agent orchestration:** Builds deterministic TaskGraph plans, executes sub-agents, writes consolidated artifacts.
- **Sandbox monotonicity:** Spawned sub-agent sandboxes must remain subsets of the parent sandbox.
- **Chunking strategy:** `chunking_strategy` selects `"structure"` (default) or `"fixed"` (legacy).
- **Parser return types:** `parse_pdf`, `parse_docx`, and `parse_file` return 3-tuples `(text, metadata, page_segments)`.
- **Vector DB abstraction:** All modules use `BaseVectorDBProvider` interface.
- **PostgreSQL provider:** Requires `psycopg2-binary` and `pgvector` extension.
- **Agent sandbox security:** Security-critical. See `docs/development/SECURITY.md` for threat model and STRIDE coverage.

## Future Plans

See [future_plans/INGEST_FUTURE.md](../future_plans/INGEST_FUTURE.md) for retrieval and embedding improvements roadmap.

