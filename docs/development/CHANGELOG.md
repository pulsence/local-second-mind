# Changelog

All notable changes to Local Second Mind are documented here.

## Unreleased

- Documentation expansion across user guide, architecture, API, and dev guides.
- Refinements to configuration reference.

## 0.2.0

### Added

- Unified interactive shell with ingest and query contexts.
- Query modes (`grounded`, `insight`, `hybrid`) and mode switching in REPL.
- Notes system with `/note` and Markdown output.
- Remote provider framework with Brave Search integration.
- AI tagging for chunks in ingest REPL.

### Changed

- LLM configuration consolidated under `llm` with per-feature overrides.
- Query pipeline now supports hybrid reranking and relevance gating.

### Fixed

- Incremental ingest reliability via manifest hash checks.

### Breaking Changes

- None known.

### Migration Notes

- If you were using a legacy `openai` section, migrate to `llm` but the legacy
  format is still supported.

## 0.1.0

### Added

- Initial local-first ingest pipeline with ChromaDB storage.
- Query pipeline with semantic retrieval and citations.
- Basic CLI entrypoints for ingest and query.
