# CLAUDE.md - Local Second Mind (v0.6.0)

## Project Overview
Local-first RAG system for personal knowledge management. Semantic search + LLM synthesis from local documents (PDF, DOCX, MD, HTML, TXT).

- **Language:** Python 3.10+
- **Entry:** `python -m lsm`
- **Config:** `config.json` or `config.yaml`

## Architecture

```
lsm/
  config/      # Dataclass configs + loader
    loader.py      # Raw config -> typed dataclass construction
    models/        # Config model modules (global/ingest/query/llm/modes/vectordb/lsm_config)
  ingest/      # Parse -> Chunk -> Embed -> Store pipeline
    pipeline.py    # Main ingest orchestration
    structure_chunking.py  # Structure-aware chunking (headings, paragraphs, sentences)
    chunking.py    # Legacy fixed-size chunking
    models.py      # PageSegment, ParseResult, WriteJob dataclasses
    commands.py    # Business logic handlers (return strings, no print)
    display.py     # Formatting utilities (return strings)
    stats.py       # Collection statistics
    explore.py     # File tree building utilities
    tagging.py     # AI tagging functionality
    language.py    # Language detection (langdetect, ISO 639-1)
    translation.py # LLM-based chunk translation
  query/       # Retrieve -> Rerank -> Synthesize pipeline
    retrieval.py   # Vector search and filtering
    rerank.py      # Reranking strategies
    synthesis.py   # LLM answer generation
    commands.py    # Business logic handlers (return strings, no print)
    display.py     # Formatting utilities (return strings)
    session.py     # SessionState and Candidate dataclasses
    notes.py       # Note generation
    citations.py   # Citation export
  remote/      # Remote source providers
    base.py        # BaseRemoteProvider, RemoteResult
    factory.py     # Provider factory and registration
    providers/     # Individual provider implementations (brave, wikipedia, arxiv, etc.)
  agents/      # Agent framework (runtime, tools, built-in agents)
    base.py        # BaseAgent and AgentState lifecycle model
    harness.py     # Runtime action loop, tool execution harness, and per-run summaries
    interaction.py # Thread-safe runtime<->UI interaction channel for permission/clarification prompts
    scheduler.py   # AgentScheduler recurring schedule engine + persistent schedule state
    task_graph.py  # AgentTask/TaskGraph orchestration graph models
    meta.py        # Built-in meta-agent orchestrator (task graph execution + shared-workspace synthesis)
    research.py    # Built-in research agent implementation
    memory/        # Agent memory storage models/backends/api/context integration
      models.py    # Memory + MemoryCandidate dataclasses
      store.py     # BaseMemoryStore, SQLiteMemoryStore, PostgreSQLMemoryStore
      api.py       # Memory lifecycle operations and ranked retrieval
      context_builder.py # Standing memory context builder for harness prompt injection
      migrations.py # Memory store migration helpers
    tools/         # Agent tool registry, sandbox, runners, and built-in tools
      runner.py    # Runner abstraction (LocalRunner) for tool execution
      ask_user.py  # Built-in clarification tool for user interaction prompts
  providers/   # LLM abstraction (OpenAI, Anthropic, Gemini, Azure, Ollama)
    helpers.py     # Shared prompts/utilities for provider implementations
  vectordb/    # Vector DB abstraction (ChromaDB, PostgreSQL)
    base.py        # BaseVectorDBProvider ABC, VectorDBGetResult, VectorDBQueryResult
    factory.py     # create_vectordb_provider(), register_provider()
    chromadb.py    # ChromaDB provider implementation
    postgresql.py  # PostgreSQL + pgvector provider implementation
    migrations/    # Migration tools (chromadb_to_postgres.py)
  logging.py   # Logging configuration (get_logger, setup_logging)
  ui/          # User interfaces
    shell/         # CLI interface
      commands/    # CLI command routing (ingest.py, query.py)
      ingest/      # Ingest CLI (cli.py)
    tui/           # Textual TUI interface
      app.py       # Main LSMApp class
      styles/      # Textual CSS split by base/screens/widgets
      completions.py # Autocomplete logic
      screens/     # Screen modules (query, ingest, settings, help)
      widgets/     # Custom widgets (results, input, status)
    web/           # Web interface (placeholder)
    desktop/       # Desktop app (placeholder)
```

**Design Pattern:** Core modules (`lsm/query/`, `lsm/ingest/`) contain pure business logic that returns results. UI modules handle command parsing, user interaction, and display formatting. Backwards compatability is not to be maintained and no deprecated methods are classes are to be preserved. Implementing features should follow a Test-Driven-Design
pattern. Whenever the user reports an error, determine if a test case should be created to cover the error and write
a test duplicating the error before fixing the error.

## Coding Patterns

**Dataclasses** - Use `__post_init__` for normalization, `validate()` for validation:
```python
@dataclass
class MyConfig:
    field: str
    def __post_init__(self):
        self.field = self.field.strip()
    def validate(self):
        if not self.field:
            raise ValueError("field required")
```

**Providers** - Extend ABC base classes:
- `lsm/providers/base.py` - LLM providers
- `lsm/remote/base.py` - Remote source providers
- `lsm/vectordb/base.py` - Vector DB providers

**Naming:** snake_case (functions), PascalCase (classes), UPPER_SNAKE_CASE (constants)

**Type hints:** Required on all function signatures. Use `from __future__ import annotations`.

**Docstrings:** Google style with Args, Returns, Raises sections.

**Logging:** `from lsm.logging import get_logger; logger = get_logger(__name__)`

## Creating a Task Plan
Every task plan should be broken down into numbered task phases. These phases are
major feature implimentations which are numbered: `## Phase N: Major Feature Name`.

Each phase is then broken into task blocks with a descriptive title as a heading.
For example: `### N.N: Feature Description`

Each task block is to be broken in to smaller sub-blocks as needed with headings
named: `#### N.N.N: Sub Feature Description`.

Each task block is to contain a list of tasks to complete, a short description
of the feature that will be implemented in this task block and the files to
modify or create.
Then after completing a task block the following taskes are to be done:
1. Create/update tests for new features
2. API keys must be configurable via `.env` file
3. Run tests: `pytest tests/ -v`
4. Update `docs/` with new features
5. Update `example_config.json` and `.env.example` with new config options
6. Write commit message in this format:
```
Added [feature name]

Feature Name:
- Bullet point describing capability
- Another capability
- Configuration options added
```
7. Update Architecture and Key Files and other related sections in 
   CLAUDE.md as needed

Each task block must have a `**success criteria:**`  which clearly describes what a successful
implementation of this task block results in.

Every Major Feature phase should have a final code review phase that
has the following tasks:
- Review the changes made in the Major Feature phase and ensure that the phase is
  entirely implemented and implemented completely
- Review the code related to the Major Feature phase and make sure that there is no remaining backwards
  compatability code or deprecated code or dead code.
- Review thet tests related to the Major Feature phase and ensure they are well structured and there are not
  mock or stub tests.

Every Major Feature phase should end with a task summarizing the changes
of the work actually complete to implement the Major Feature plan and
write them into the docs/CHANGELOG.md file.

## Commands

**Environment requirement:**  
- In WSL/Linux shells, run project commands using `.venv-wsl`.  
- In Windows shells, run project commands using `.venv-win`.  

Example on Windows PowerShell:
```powershell
.venv-win\Scripts\python -m pytest tests/ -v
.venv-win\Scripts\python -m lsm --help
```

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

## Key Files

| File | Purpose |
|------|---------|
| `lsm/config/loader.py` | Config parsing/building/serialization and validation entry points |
| `lsm/config/models/` | Config dataclasses split by domain (global/ingest/query/llm/modes/vectordb) |
| `lsm/config/models/global_config.py` | GlobalConfig dataclass for multi-module settings (embed_model, device, batch_size, global_folder) |
| `lsm/query/commands.py` | Query business logic handlers |
| `lsm/query/display.py` | Query formatting utilities |
| `lsm/query/retrieval.py` | Vector search and candidate filtering |
| `lsm/query/synthesis.py` | LLM answer synthesis |
| `lsm/ingest/commands.py` | Ingest business logic handlers |
| `lsm/ingest/display.py` | Ingest formatting utilities |
| `lsm/ingest/pipeline.py` | Ingest orchestration |
| `lsm/ingest/structure_chunking.py` | Structure-aware chunking (headings, paragraphs, sentences, page numbers) |
| `lsm/ingest/chunking.py` | Legacy fixed-size chunking |
| `lsm/ingest/models.py` | PageSegment, ParseResult, WriteJob dataclasses |
| `lsm/ingest/fs.py` | File discovery (iter_files) and folder tag collection (collect_folder_tags) |
| `lsm/ingest/language.py` | Language detection (langdetect, ISO 639-1 codes) |
| `lsm/ingest/translation.py` | LLM-based chunk translation for cross-language search |
| `lsm/ingest/stats_cache.py` | StatsCache class for caching collection statistics |
| `lsm/ingest/manifest.py` | Manifest load/save and `get_next_version()` for versioning |
| `lsm/query/planning.py` | Shared query planning (candidate retrieval, filtering, reranking) |
| `lsm/vectordb/base.py` | BaseVectorDBProvider ABC, VectorDBGetResult, VectorDBQueryResult dataclasses |
| `lsm/vectordb/factory.py` | create_vectordb_provider() factory with lazy class loading |
| `lsm/vectordb/chromadb.py` | ChromaDB provider implementation |
| `lsm/vectordb/postgresql.py` | PostgreSQL + pgvector provider implementation |
| `lsm/vectordb/migrations/chromadb_to_postgres.py` | ChromaDB-to-PostgreSQL migration tool |
| `lsm/config/models/constants.py` | Default values and WELL_KNOWN_EMBED_MODELS dimension dictionary |
| `lsm/providers/factory.py` | LLM provider creation |
| `lsm/providers/helpers.py` | Shared LLM provider prompts and utilities |
| `lsm/remote/__init__.py` | Remote provider registration |
| `lsm/remote/providers/` | Remote provider implementations |
| `lsm/agents/harness.py` | Agent runtime harness and orchestration |
| `lsm/agents/interaction.py` | Interaction channel datamodels and blocking request/response bridge |
| `lsm/agents/scheduler.py` | AgentScheduler service for interval/cron runs with overlap policies and `schedules.json` persistence |
| `lsm/agents/task_graph.py` | Meta-agent task graph datamodels and dependency-order helpers |
| `lsm/agents/meta.py` | Meta-agent orchestrator with dependency-aware sub-agent execution, shared run workspace, and final synthesis artifacts |
| `lsm/agents/research.py` | Built-in research agent |
| `lsm/agents/memory/models.py` | Memory and MemoryCandidate datamodels |
| `lsm/agents/memory/store.py` | Memory storage abstraction and SQLite/PostgreSQL backends |
| `lsm/agents/memory/api.py` | Memory lifecycle helpers and ranked search |
| `lsm/agents/memory/context_builder.py` | Standing memory context builder for AgentHarness |
| `lsm/agents/memory/migrations.py` | Memory backend migration helpers |
| `lsm/agents/tools/` | Agent tool system and sandbox |
| `lsm/agents/tools/runner.py` | Runner abstraction and LocalRunner execution limits |
| `lsm/agents/tools/ask_user.py` | Clarification request tool bound to `AgentHarness` interaction flow |
| `lsm/agents/tools/spawn_agent.py` | Meta-system tool for spawning sub-agent runs |
| `lsm/agents/tools/await_agent.py` | Meta-system tool for waiting on spawned sub-agent completion |
| `lsm/agents/tools/collect_artifacts.py` | Meta-system tool for collecting spawned sub-agent artifacts |
| `lsm/logging.py` | Logging configuration |
| `lsm/ui/tui/app.py` | TUI main application |
| `lsm/ui/tui/screens/` | TUI screen modules |
| `lsm/ui/shell/commands/` | CLI command routing |
| `lsm/ui/shell/commands/agents.py` | Shell runtime manager for multi-agent lifecycle, interaction routing, scheduler, and memory commands |
| `example_config.json` | Full config example |
| `.env.example` | Environment variable template |

## Environment Variables

API keys go in `.env`, never in config files:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- `BRAVE_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `CORE_API_KEY`
- `LSM_POSTGRES_CONNECTION_STRING`, `LSM_POSTGRES_TABLE` (PostgreSQL vector DB)
- See `.env.example` for full list

## Important Notes

- **Virtual environment:** Run project commands with `.venv\Scripts\python` (Windows PowerShell) to ensure dependencies match the project environment.
- **Config loading:** Raw dict -> dataclass construction -> validation
- **Config structure:** Zero flat top-level fields. All settings nested under section objects: `"global"`, `"ingest"`, `"vectordb"`, `"llms"`, `"query"`, `"modes"`, `"notes"`, `"remote_providers"`, `"agents"`.
- **Agent memory config:** `agents.memory.storage_backend` controls backend selection (`auto`, `sqlite`, `postgresql`). Relative `agents.memory.sqlite_path` resolves under `agents.agents_folder`.
- **Agent harness memory injection:** `AgentHarness` can build standing memory context before each LLM call; empty memory context is valid and results in no injection.
- **Agent run summaries:** `AgentHarness` emits `<agents_folder>/<agent_name>_<timestamp>/run_summary.json` with tool usage, approvals/denials, artifacts, outcome, duration, and token usage metadata.
- **Agent scheduler:** `AgentScheduler` stores runtime schedule metadata in `<agents_folder>/schedules.json` (`last_run_at`, `next_run_at`, `last_status`, `last_error`) and enforces overlap policy (`skip`, `queue`, `cancel`).
- **Scheduled-run safety defaults:** Scheduler runs are read-only by default, disable network by default, require explicit schedule-param opt-in for writes/network/exec, switch to `execution_mode="prefer_docker"` for network/exec runs, and support `params.force_docker=true` to require Docker for all scheduled tool risks.
- **Meta-agent orchestration:** `MetaAgent` builds deterministic `TaskGraph` plans from user goals (or structured JSON task specs), executes sub-agents with dependency-safe ordering, writes per-run workspace artifacts (`final_result.md`, `meta_log.md`), and scopes child sandboxes to shared-read/per-agent-write paths.
- **Agent interaction tools:** `ask_user` is always available in harness/base-agent allowlist filtering and routes clarification requests through `InteractionChannel`.
- **Agents TUI interaction panel:** Agents screen includes a Running Agents table (multi-run selection by `agent_id`) and an Interaction Request panel for approve/approve-session/deny/reply flows via runtime-manager interaction APIs.
- **Meta-agent system tools:** `spawn_agent`, `await_agent`, and `collect_artifacts` are available in the default tool registry for harness-driven orchestration flows.
- **Meta-agent UI/CLI:** shell supports `/agent meta start|status|log`; Agents TUI includes a Meta panel with task graph progress, sub-agent run status, and meta artifact listing.
- **TUI help version sync:** When project version changes (for example `pyproject.toml` and `lsm/__init__.py`), update the Help modal's "What's New" version text in `lsm/ui/tui/screens/help.py` (`_WHATS_NEW` and related section heading).
- **Sandbox monotonicity:** spawned sub-agent sandboxes must remain subsets of the parent sandbox across paths, network access, permission gates, runner policy, and runtime limits.
- **Curator memory mode:** `CuratorAgent` supports `--mode memory` to distill recent run summaries into `memory_candidates.md` and `memory_candidates.json`.
- **Memory commands/UI:** Query shell supports `/memory candidates|promote|reject|ttl`, and the TUI Agents screen includes a memory-candidates panel for refresh/approve/reject/TTL edits.
- **Global config:** `embed_model`, `device`, `batch_size`, `global_folder` live in `GlobalConfig` (read from `"global"` section), not on `IngestConfig`.
- **Ingest config:** All ingest-only fields (`roots`, `manifest`, `chunk_size`, etc.) read from `"ingest"` section.
- **Vector DB config schema:** Use `vectordb.persist_dir` and `vectordb.collection`; top-level fallback fields are removed.
- **Remote provider activation:** `remote_providers[]` entries do not have an `enabled` field; active providers are selected by mode (`source_policy.remote.enabled` + optional `remote_providers` list).
- **LLM config schema:** `llms` is an object with `providers` (connection details) and `services` (feature-to-model mappings). Services reference providers by name.
- **Chunking strategy:** `chunking_strategy` in `IngestConfig` selects `"structure"` (default, heading/paragraph/sentence-aware) or `"fixed"` (legacy sliding-window). Structure chunking never splits sentences, never mixes paragraphs or headings.
- **Page number tracking:** `parse_pdf` and `parse_docx` return `PageSegment` lists alongside text. Chunk metadata includes `page_number` for paginated formats (PDF, DOCX).
- **Parser return types:** `parse_pdf`, `parse_docx`, and `parse_file` return 3-tuples `(text, metadata, page_segments)`. Other parsers (`parse_txt`, `parse_md`, `parse_html`) return 2-tuples; `parse_file` wraps them with `None` for page_segments.
- **Incremental ingest:** Manifest tracks file hashes/mtimes - skip unchanged files
- **Service resolution:** `resolve_service(name)` looks up a service, falls back to `"default"`, then merges provider connection details to produce `LLMConfig`.
- **Error handling:** Use `skip_errors` mode in ingest for graceful per-file failures
- **Language detection:** Per-document (not per-chunk) via `langdetect`. Gated by `enable_language_detection` in `IngestConfig`. Result stored in `doc_metadata["language"]` and propagated to all chunks automatically.
- **Translation:** LLM-based, per-chunk, via `lsm/ingest/translation.py`. Requires `enable_language_detection` to also be true. Uses `llms.services.translation` (falls back to `"default"`). Stores `"translated_from"` in chunk metadata.
- **Embedding dimension:** Auto-detected from `WELL_KNOWN_EMBED_MODELS` in `GlobalConfig.__post_init__`. Validated at pipeline runtime against `model.get_sentence_embedding_dimension()`. Explicit `embedding_dimension` in config overrides auto-detection.
- **Root tagging:** `IngestConfig.roots` is `List[RootConfig]`. Config accepts strings, Path objects, dicts (`{"path": ..., "tags": [...], "content_type": ...}`), or `RootConfig` instances. Use `root_paths` property for `List[Path]`. Root tags/content_type propagated to chunk metadata as `root_tags`, `content_type`.
- **Folder tagging:** `.lsm_tags.json` files in subdirectories contain `{"tags": [...]}`. Tags collected root-to-leaf and stored as `folder_tags` in chunk metadata. Invalid JSON silently skipped.
- **iter_files:** Returns `Iterable[Tuple[Path, RootConfig]]` — each file paired with its originating root config.
- **Partial ingest:** `max_files` and `max_seconds` on `IngestConfig` (both `Optional[int]`, default `None`). Pipeline cleanly flushes queues and saves manifest when limits are reached. Validation rejects values < 1.
- **Stats caching:** `StatsCache` in `lsm/ingest/stats_cache.py` caches collection stats at `<persist_dir>/stats_cache.json`. Stale when chunk count changes or age exceeds `max_age_seconds` (default 3600). Invalidated after each ingest run.
- **MuPDF repair:** `_open_pdf_with_repair()` in `parsers.py` uses 3-strategy approach: direct open, garbage-collection rebuild, plain stream fallback. Repairable markers: `"syntax error"`, `"zlib error"`, `"xref"`, `"trailer"`, `"startxref"`, `"corrupt"`, `"malformed"`.
- **Chunk versioning:** `enable_versioning` on `IngestConfig` (default `False`). When enabled, old chunks get `is_current=False` metadata instead of being deleted. New chunks get `is_current=True` and `version` in metadata. `get_next_version()` in `manifest.py` computes next version from manifest. Query retrieval passes `{"is_current": True}` filter when versioning is active.
- **Vector DB abstraction:** All modules use `BaseVectorDBProvider` interface — no raw `chromadb` imports outside `lsm/vectordb/chromadb.py`. Factory function `create_vectordb_provider()` creates the right provider from `VectorDBConfig`.
- **VectorDB provider interface:** Abstract methods: `add_chunks()`, `query()`, `get()`, `update_metadatas()`, `delete_by_filter()`, `delete_all()`, `count()`, `get_stats()`. Result types: `VectorDBGetResult` (non-similarity) and `VectorDBQueryResult` (similarity with distances).
- **VectorDB get():** Supports retrieval by `ids`, by `filters`, with `limit`/`offset` pagination and `include` field selection (`"documents"`, `"metadatas"`, `"embeddings"`). Returns `VectorDBGetResult` with optional fields.
- **VectorDB filter format:** Use simple `{"key": "value"}` at the provider interface level. ChromaDB provider converts to `{"key": {"$eq": "value"}}` internally. PostgreSQL uses JSONB containment (`@>`).
- **PostgreSQL provider:** Requires `psycopg2-binary` and `pgvector` extension. Config: `VectorDBConfig(provider="postgresql", connection_string=..., embedding_dimension=...)`. Table auto-created with `id`, `text`, `metadata` (JSONB), and `embedding` (vector) columns.
- **Migration tool:** `lsm/vectordb/migrations/chromadb_to_postgres.py` migrates ChromaDB collections to PostgreSQL using batched provider `get()`/`add_chunks()` calls. Available via `migrate-vectordb` CLI command and `/migrate` TUI command.
- **Large collections:** Tested stable with 300K+ chunks
- **Agent sandbox security:** The agent sandbox is security-critical code. All changes to sandbox (`lsm/agents/tools/sandbox.py`), permission gate, tool execution, runner, or environment scrubbing code require corresponding adversarial security tests that attempt to bypass protections (path traversal, null byte injection, symlink escape, prompt injection, secret leakage, etc.). Tests must verify that attacks are rejected, not just that normal operations succeed. Never weaken sandbox restrictions without explicit user approval. See `.agents/docs/architecture/development/SECURITY.md` for the full threat model and STRIDE coverage matrix.

## Project Documentation

- `docs/` - End-user guides and changelog
- `.agents/docs/INDEX.md` - Developer and agent documentation entry point
- `.agents/docs/architecture/development/TUI_ARCHITECTURE.md` - TUI conventions for state, events, workers, timers, errors, and testing
- `tests/` - pytest test suite with fixtures in `conftest.py`
