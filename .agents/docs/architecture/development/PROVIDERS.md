# Provider System Architecture

This document describes the LSM v0.5.0 provider abstractions for both LLM and vector database backends.

## Design Goals

- Keep business logic stable while allowing provider-specific transport implementations.
- Support multiple provider backends through a registry + factory pattern.
- Keep app modules provider-agnostic through shared typed interfaces.

## LLM Provider Architecture

Core abstraction: `BaseLLMProvider` in `lsm/providers/base.py`. The architecture centers on a thin transport layer per provider with shared business logic in the base class. See [PROVIDERS.md](../api-reference/PROVIDERS.md) for the full contract, factory APIs, and method signatures.

### Integration Points

- Query pipeline uses rerank + synthesize features via resolved LLM services.
- Ingest uses tagging/translation services when enabled.
- Provider availability gates optional features at runtime.

## Vector DB Provider Architecture

Core abstraction: `lsm/vectordb/base.py` -> `BaseVectorDBProvider`.

### Provider Contract

Vector DB providers follow `BaseVectorDBProvider` in `lsm/vectordb/base.py` with standardized result types (`VectorDBQueryResult`, `VectorDBGetResult`) and factory registration via `lsm/vectordb/factory.py`.

Key behaviors:

- Uniform metadata filtering at the interface layer, normalized per backend.
- Availability/health checks gate query and ingest operations.
- Built-in backends cover ChromaDB and PostgreSQL/pgvector.

Migration helpers are available in `lsm/vectordb/migrations/chromadb_to_postgres.py` for ChromaDB -> PostgreSQL moves.

## Extension Workflow

To add a custom provider:

1. Implement the correct base class (`BaseLLMProvider` or `BaseVectorDBProvider`).
2. Register it via the corresponding factory `register_provider(...)`.
3. Configure it in `config.json` (`llms` or `vectordb`).
4. Add tests under `tests/test_providers/` or `tests/test_vectordb/`.
