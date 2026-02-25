# LSM Architecture Overview

Local Second Mind (LSM) is a local-first RAG system organized around clear module boundaries:

- configuration loading/validation
- ingest pipeline
- query pipeline
- provider abstractions (LLM, remote, vector DB)
- UI layers (TUI + shell commands)
- optional agent runtime

## Design Principles

### Local-first core
- Documents, embeddings, and vector data stay local.
- External network calls are optional and scoped to configured LLM and remote providers.

### Explicit configuration
- Behavior is driven by config dataclasses with validation.
- No flat top-level config fields; settings are grouped by section (`global`, `ingest`, `query`, `llms`, etc.).

### Provider abstraction
- LLMs, remote sources, and vector databases are behind interfaces/factories.
- Core pipelines do not depend on provider-specific SDK details.

### Separation of concerns
- Core business logic lives in `lsm/ingest/` and `lsm/query/`.
- UI modules orchestrate commands, rendering, and interaction.

## High-Level Architecture

```text
Config (loader + dataclasses)
        |
        +--> Ingest Pipeline ------------------+
        |    scan -> parse -> chunk -> embed   |
        |    -> vectordb write + manifest      |
        |                                       |
        +--> Query Pipeline --------------------+--> Answer + citations (+ optional notes/chats)
        |    embed query -> retrieve ->         |
        |    prefilter -> rerank -> synthesize  |
        |                                       |
        +--> Remote Providers (optional) -------+
        |
        +--> Agents Runtime (optional tool loop)
```

## Core Subsystems

### Configuration

- Loader: `lsm/config/loader.py`
- Root model: `lsm/config/models/lsm_config.py`
- Major sections: `global`, `ingest`, `vectordb`, `llms`, `query`, `modes`, `notes`, `chats`, `remote_providers`, `remote_provider_chains`, `agents`
- LLM config uses `providers + services` with runtime resolution (`resolve_service`).

### Ingest Pipeline

- Orchestration: `lsm/ingest/pipeline.py`
- File discovery: `lsm/ingest/fs.py`
- Parsing: `lsm/ingest/parsers.py`
- Chunking:
  - structure-aware: `lsm/ingest/structure_chunking.py` (default)
  - fixed-size fallback: `lsm/ingest/chunking.py`
- Optional enrichments: language detection, translation, AI tagging
- Incremental/versioning state: `lsm/ingest/manifest.py`

### Query Pipeline

- API + flow orchestration: `lsm/query/api.py`, `lsm/query/planning.py`
- Retrieval: `lsm/query/retrieval.py`
- Metadata prefiltering: `lsm/query/prefilter.py`
- Decomposition: `lsm/query/decomposition.py`
- Reranking: `lsm/query/rerank.py`
- Context + citations + notes:
  - `lsm/query/context.py`
  - `lsm/query/citations.py`
  - `lsm/query/notes.py`
- Session/cost state: `lsm/query/session.py`, `lsm/query/cost_tracking.py`

### Provider Layers

- LLM providers: `lsm/providers/`
  - OpenAI, OpenRouter, Anthropic, Gemini, Azure OpenAI, Local
- Remote providers: `lsm/remote/providers/`
  - Brave, Wikipedia, arXiv, Semantic Scholar, Crossref, OpenAlex, CORE, OAI-PMH, IxTheo, PhilPapers
- Vector DB providers: `lsm/vectordb/`
  - ChromaDB and PostgreSQL/pgvector

Factories:
- `lsm/providers/factory.py`
- `lsm/remote/factory.py`
- `lsm/vectordb/factory.py`

### UI Layers

- TUI app: `lsm/ui/tui/app.py`
- TUI screens: query, ingest, remote, agents, settings
- Shell command handlers: `lsm/ui/shell/commands/`
- CLI entrypoint: `lsm/__main__.py`

### Agents (Optional)

- Runtime + state: `lsm/agents/base.py`, `lsm/agents/harness.py`
- Tooling + sandbox: `lsm/agents/tools/`
- Built-in research agent: `lsm/agents/academic/research.py`
- Factory/registry: `lsm/agents/factory.py`

## Data and State

- Vector store: configured via `vectordb` (ChromaDB or PostgreSQL)
- Ingest manifest: tracks file version/hash data for incremental runs
- Query cache: in-memory TTL/LRU cache (`query.enable_query_cache`)
- Remote result cache: optional on-disk provider cache
- Chat and notes output: saved under configured directories
- Agent state/log files: persisted under `agents.agents_folder`

## References

- Ingest details: `.agents/docs/architecture/development/INGEST.md`
- Query details: `.agents/docs/architecture/development/QUERY.md`
- Provider architecture: `.agents/docs/architecture/development/PROVIDERS.md`
- Mode architecture: `.agents/docs/architecture/development/MODES.md`
- Config reference: `docs/user-guide/CONFIGURATION.md`
- Agents guide: `.agents/docs/architecture/development/AGENTS.md`
- Provider extension guide: `.agents/docs/architecture/api-reference/ADDING_PROVIDERS.md`
