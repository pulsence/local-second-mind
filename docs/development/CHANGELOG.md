# Changelog

All notable changes to Local Second Mind are documented here.

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
