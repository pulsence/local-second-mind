# Architecture Overview

This document provides the architecture and key files reference for Local Second Mind. See [INDEX.md](./INDEX.md) for the quick start guide.

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

**Design Pattern:** Core modules (`lsm/query/`, `lsm/ingest/`) contain pure business logic that returns results. UI modules handle command parsing, user interaction, and display formatting. Backwards compatability is not to be maintained and no deprecated methods are classes are to be preserved. Implementing features should follow a Test-Driven-Design pattern.

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

## Package Details

See individual package documentation in the [architecture](./architecture/) folder:

- [lsm.agents](./architecture/lsm.agents.md) - Agent framework, tools, memory
- [lsm.config](./architecture/lsm.config.md) - Configuration loading and models
- [lsm.ingest](./architecture/lsm.ingest.md) - Document parsing and embedding
- [lsm.providers](./architecture/lsm.providers.md) - LLM provider implementations
- [lsm.query](./architecture/lsm.query.md) - Query retrieval and synthesis
- [lsm.remote](./architecture/lsm.remote.md) - Remote source providers
- [lsm.ui](./architecture/lsm.ui.md) - User interfaces
- [lsm.vectordb](./architecture/lsm.vectordb.md) - Vector database providers
