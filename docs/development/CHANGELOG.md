# Changelog

All notable changes to Local Second Mind are documented here.

## Unreleased

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
