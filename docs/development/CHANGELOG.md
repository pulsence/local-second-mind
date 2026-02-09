# Changelog

All notable changes to Local Second Mind are documented here.

## Unreleased

### Added

- Base provider message transport interface in `BaseLLMProvider` with `_send_message(...)` and `_send_streaming_message(...)`.
- Shared fallback answer helper on `BaseLLMProvider` to replace per-provider `_fallback_answer()` duplication.
- Shared provider JSON schema constants in `lsm/providers/helpers.py`:
  - `RERANK_JSON_SCHEMA`
  - `TAGS_JSON_SCHEMA`

### Changed

- Moved shared LLM business logic into `BaseLLMProvider` concrete methods:
  - `rerank(...)`
  - `synthesize(...)`
  - `stream_synthesize(...)`
  - `generate_tags(...)`
- Refactored OpenAI, Azure OpenAI, Anthropic, Gemini, and Local providers to keep provider-specific transport/config logic only.
- Removed duplicated `rerank/synthesize/stream_synthesize/generate_tags` implementations from provider classes in favor of base implementations.
- Verified provider factory and exports remain correct after refactor (`lsm/providers/factory.py`, `lsm/providers/__init__.py`).

## 0.4.0 - 2026-02-08

### Added

- Config restructuring with a dedicated top-level `global` object and zero flat top-level fields.
- LLM providers/services registry pattern (`llms.providers` + `llms.services`) with service resolution and fallback to `default`.
- Per-provider LLM pricing tables and cost estimation integrated into query cost tracking.
- Structure-aware chunking in `lsm/ingest/structure_chunking.py` that respects headings, paragraphs, and sentence boundaries.
- Heading detection for Markdown (`#`-style) and bold-only lines (common in PDF extractions).
- Page number tracking for PDF and DOCX documents via `PageSegment` dataclass.
- DOCX page break detection using `<w:lastRenderedPageBreak/>` and `<w:br w:type="page"/>` XML elements.
- `chunking_strategy` config option (`"structure"` or `"fixed"`) on `IngestConfig`.
- Chunk metadata now includes `heading`, `paragraph_index`, and `page_number` fields when using structure chunking.
- Language detection module (`lsm/ingest/language.py`) using `langdetect` for automatic document language identification (ISO 639-1 codes).
- `enable_language_detection` config option on `IngestConfig` (default `False`). Detected language stored in chunk metadata as `"language"`.
- LLM-based machine translation module (`lsm/ingest/translation.py`) for cross-language search on multilingual corpora.
- `enable_translation` and `translation_target` config options on `IngestConfig`. Uses `"translation"` LLM service from `llms.services`.
- `WELL_KNOWN_EMBED_MODELS` dictionary in `constants.py` mapping 30+ embedding models to their output dimensions.
- `embedding_dimension` field on `GlobalConfig` with auto-detection from well-known models. Pipeline validates actual model dimension matches config at startup.
- `RootConfig` dataclass in `lsm/config/models/ingest.py` supporting per-root `tags` and `content_type`.
- `IngestConfig.roots` now accepts strings, Path objects, dicts with `path`/`tags`/`content_type`, or `RootConfig` instances (all normalized to `List[RootConfig]`).
- `root_paths` property on `IngestConfig` for convenient `List[Path]` access.
- `.lsm_tags.json` subfolder tag support via `collect_folder_tags()` in `lsm/ingest/fs.py`.
- Root tags, content type, and folder tags propagated to chunk metadata as `root_tags`, `content_type`, and `folder_tags`.
- `iter_files()` now yields `(Path, RootConfig)` tuples to track file-to-root mapping.
- 30 new tests in `tests/test_ingest/test_root_config.py` covering RootConfig, config loading, folder tag discovery, and iter_files changes.
- Partial ingest support via `max_files` and `max_seconds` limits on `IngestConfig`. Pipeline cleanly flushes queues and saves manifest when limits are reached.
- Stats caching via `StatsCache` class in `lsm/ingest/stats_cache.py`. Cache stored at `<persist_dir>/stats_cache.json` with staleness detection (count mismatch or age expiry). Automatically invalidated after ingest.
- Enhanced MuPDF PDF repair with multi-stage strategy: direct open, garbage-collection rebuild (`garbage=4, deflate=True, clean=True`), and plain stream fallback. Expanded repairable error markers to include `"trailer"`, `"startxref"`, `"corrupt"`, and `"malformed"`.
- Chunk version control via `enable_versioning` on `IngestConfig`. Old chunks marked `is_current=False` instead of deleted. Version number tracked in chunk metadata and manifest. Query retrieval filters to `is_current=True` when versioning is active.
- `where_filter` parameter on `retrieve_candidates()` for metadata-level filtering at query time.
- Full PostgreSQL + pgvector support as an alternative vector database backend via `VectorDBConfig(provider="postgresql")`.
- `VectorDBGetResult` dataclass in `lsm/vectordb/base.py` for typed non-similarity retrieval results (ids, documents, metadatas, embeddings).
- `get()` abstract method on `BaseVectorDBProvider` supporting retrieval by IDs, filters, with pagination (`limit`/`offset`) and field selection (`include`).
- `update_metadatas()` and `delete_all()` abstract methods on `BaseVectorDBProvider`, implemented on both ChromaDB and PostgreSQL providers.
- `PostgreSQLProvider` with full implementation: `add_chunks()`, `query()`, `get()`, `update_metadatas()`, `delete_by_filter()`, `delete_all()`, `count()`, `get_stats()`, and `_normalize_filters()` for JSONB containment queries.
- ChromaDB-to-PostgreSQL migration tool in `lsm/vectordb/migrations/chromadb_to_postgres.py` with batched reads/writes and progress callback.
- `migrate-vectordb` CLI subcommand and `/migrate` TUI command for running migrations.
- PostgreSQL connection variables in `.env.example` (`LSM_POSTGRES_CONNECTION_STRING`, `LSM_POSTGRES_TABLE`).
- PostgreSQL vectordb example in `example_config.json`.
- Query result cache in `lsm/query/cache.py` with TTL expiration + LRU eviction, integrated into query execution behind `query.enable_query_cache`.
- Query chat modes (`query.chat_mode = single|chat`) with `/mode chat` and `/mode single` switching in TUI query commands.
- Global chat transcript settings via `ChatsConfig` (`enabled`, `dir`, `auto_save`, `format`) with transcript auto-save support.
- Chat conversation/session tracking in query state, including provider response/session ID chaining for follow-up turns.
- Provider-side LLM cache/session reuse support across OpenAI, Azure OpenAI, Anthropic, Gemini, and Local providers (where applicable by provider API).
- TUI live mode toggle for provider cache reuse: `/mode set llm_cache on|off`.
- Query metadata prefiltering improvements (`lsm/query/prefilter.py`, `lsm/query/planning.py`) using metadata inventory + deterministic author/year/title extraction.
- Prefiltering now supports all tag fields together: `ai_tags`, `user_tags`, `root_tags`, `folder_tags`, plus `content_type`.
- Context anchor controls in query TUI (`/context`, `/context doc ...`, `/context chunk ...`, `/context clear`) with anchor-first context prioritization.
- Added/updated query tests for metadata prefilter behavior and anchor prioritization (`tests/test_query/test_prefilter.py`, `tests/test_query/test_planning.py`).
- Natural language query decomposition in `lsm/query/decomposition.py` with `QueryFields` (`author`, `keywords`, `title`, `date_range`, `doi`, `raw_query`).
- Added deterministic field extraction (`extract_fields_deterministic`) and AI-assisted extraction (`extract_fields_ai`) with structured JSON parsing and deterministic fallback.
- Added decomposition dispatcher `decompose_query(method="deterministic"|"ai")` and test coverage in `tests/test_query/test_decomposition.py`.
- Added `llms.services.decomposition` support so query decomposition uses a dedicated configurable model/provider.
- Mode-level chat save overrides via `modes[].chats` (`auto_save`, `dir`) for per-mode transcript behavior.
- Dict-based remote provider protocol in `lsm/remote/base.py` and all built-in providers (`search_structured`, `get_input_fields`, `get_output_fields`, `get_description`) with normalized output fields (`url`, `title`, `description`, `doi`, `authors`, `year`, `score`, `metadata`).
- Remote provider structured protocol test coverage in `tests/test_providers/remote/test_structured_protocol.py`.
- Remote result disk caching via `lsm/remote/storage.py` with `save_results()` and `load_cached_results()`, integrated into query remote fetch flow.
- Remote provider config keys `cache_results` and `cache_ttl` for per-provider cache control.
- Remote provider chaining with `remote_provider_chains` config, `ChainLink` mapping (`output:input`), and `RemoteProviderChain` execution in `lsm/remote/chain.py`.
- Restructured TUI settings screen to config-aligned sub-tabs: `Global`, `Ingest`, `Query`, `LLM`, `Vector DB`, `Modes`, `Remote`, and `Chats/Notes`, with section-level Save/Reset controls.
- Live settings updates in TUI now write directly to `app.config` for edited fields, with validation/status feedback and config reload-on-reset.
- Agent framework foundation in new `lsm/agents/` package with `AgentStatus`, `AgentState`, `BaseAgent`, and core runtime models (`AgentLogEntry`, `ToolResponse`, `AgentContext`).
- New optional `agents` config block with `AgentConfig` and `SandboxConfig`, including loader/serializer support for `agents_folder`, token/iteration limits, context strategy, sandbox permissions, and per-agent overrides.
- Agent tool system in `lsm/agents/tools/` with `BaseTool`, `ToolRegistry`, `ToolSandbox`, and default tools for file IO, URL loading, local embedding queries, LLM prompting, remote provider queries, and remote chain execution.
- Agent runtime engine in `lsm/agents/harness.py` with JSON action loop, tool execution, context-window strategies (`compact` and `fresh`), background thread execution, pause/resume/stop controls, budget/iteration guards, and persisted per-run state files under `agents_folder`.
- Agent log helpers in `lsm/agents/log_formatter.py` for formatting, saving, and loading structured `AgentLogEntry` traces.
- Research agent in `lsm/agents/research.py` with LLM-driven topic decomposition, per-subtopic tool selection, iterative evidence synthesis, outline review loops, and markdown outline persistence.
- Agent registry/factory in `lsm/agents/factory.py` with built-in `research` registration and `create_agent()` entrypoint for extensible agent creation.
- Agent UI integration with new `lsm/ui/tui/screens/agents.py` tab (launcher, status, pause/resume/stop, log view) wired into `LSMApp`.
- Added shell-level agent command handlers in `lsm/ui/shell/commands/agents.py` supporting `/agent start|status|pause|resume|stop|log`, and query screen routing for `/agent` commands.

### Changed

- `parse_pdf()` and `parse_docx()` now return 3-tuples `(text, metadata, page_segments)` to preserve page boundary information.
- `parse_file()` updated to return 3-tuples consistently across all formats (page_segments is `None` for non-paginated formats).
- Pipeline writer thread now writes `heading`, `paragraph_index`, and `page_number` into vector DB chunk metadata.
- Default chunking strategy is `"structure"`; legacy fixed-size chunking available via `"fixed"`.
- All consumer modules (`stats.py`, `tagging.py`, `api.py`, `planning.py`, `retrieval.py`, `pipeline.py`, TUI screens/commands) now use `BaseVectorDBProvider` interface exclusively — no raw ChromaDB imports outside `lsm/vectordb/chromadb.py`.
- `init_collection()` removed from `retrieval.py` — consumers use `create_vectordb_provider()` factory.
- `get_by_filter()` removed from `BaseVectorDBProvider` — replaced by `get(filters=...)`.
- `require_chroma_collection()` utility removed — no longer needed with provider abstraction.
- `lsm/vectordb/utils.py` deleted entirely.
- Filter format normalized: simple `{"key": "value"}` instead of `{"key": {"$eq": "value"}}` at the provider interface level.
- `query.enable_llm_server_cache` default is now `true`.
- Removed `llm_prompt_cache_retention` query config option; provider backends control retention policy.
- Query planning now resolves `llms.services.decomposition` and passes that model config into metadata prefilter/decomposition flow.
- Chat auto-save now applies mode-level overrides before saving transcripts (global defaults can be overridden per mode).

## 0.3.2

### Added

- Clean ingest API in `lsm/ingest/api.py` with typed results for ingest, stats, info, and wipe operations.
- Progress callback support across ingest and query flows, including TUI progress integration.
- Shared LLM provider helpers in `lsm/providers/helpers.py` to centralize prompts, parsing, and fallback behavior.
- Global path management in `lsm/paths.py` with default user folder support for chats and notes.
- Expanded test coverage with new vector DB, ingest API, config, logging, and path test suites.
- New integration test fixtures and suites for ingest pipeline and query progress callbacks.

### Changed

- Refactored `lsm.ingest` toward a cleaner architecture that separates business logic from UI command handling.
- Consolidated duplicated provider logic across OpenAI, Anthropic, Gemini, Azure OpenAI, and Local providers.
- Removed legacy configuration fallbacks and typo-tolerant compatibility paths from config loading.
- Simplified remote provider activation semantics by mode-driven selection.
- Improved lazy loading for package/provider/vector DB components to reduce startup overhead.

### Fixed

- Improved ingest and query fault tolerance with better error handling and partial-result resilience.
- Added graceful handling for provider failures and remote timeout scenarios.
- Removed deprecated/legacy ingest code paths that conflicted with current architecture guidance.

## 0.3.1

### Added

- **TUI (Textual User Interface)** - Rich terminal interface with:
  - Tabbed navigation (Query, Ingest, Settings)
  - ResultsPanel widget with expandable citations
  - CommandInput widget with history and Tab autocomplete
  - StatusBar widget showing mode, chunks, cost, provider status
  - Keyboard shortcuts (Ctrl+B to build, Ctrl+E to expand, etc.)
  - Help modal (F1) with command reference
- Documentation expansion across user guide, architecture, API, and dev guides.
- Refinements to configuration reference.
- Added Anthropic, Gemini, Local (Ollama), and Azure OpenAI providers.
- Added provider health tracking and `/provider-status` REPL command.
- LLM configuration now uses an ordered `llms` list with per-feature selection.

### Changed

- Module restructuring for GUI preparation:
  - CLI/REPL code moved to `lsm/gui/shell/`
  - Remote providers moved to `lsm/remote/`
  - Split large REPL files into modular components

## 0.2.0

### Added

- Unified interactive shell with ingest and query contexts.
- Query modes (`grounded`, `insight`, `hybrid`) and mode switching in REPL.
- Notes system with `/note` and Markdown output.
- Remote provider framework with Brave Search integration.
- AI tagging for chunks in ingest REPL.

### Changed

- LLM configuration consolidated under `llm` with per-feature overrides.
- LLM configuration now supports ordered multi-provider selection via `llms`.
- Query pipeline now supports hybrid reranking and relevance gating.

### Fixed

- Incremental ingest reliability via manifest hash checks.

### Breaking Changes

- None known.

### Migration Notes

- Legacy `openai` and single-provider `llm` sections are removed; migrate to
  the ordered `llms` list schema.

## 0.1.0

### Added

- Initial local-first ingest pipeline with ChromaDB storage.
- Query pipeline with semantic retrieval and citations.
- Basic CLI entrypoints for ingest and query.
