# Provider System Architecture

This document describes the LSM v0.5.0 provider abstractions for both LLM and vector database backends.

## Design Goals

- Keep business logic stable while allowing provider-specific transport implementations.
- Support multiple provider backends through a registry + factory pattern.
- Keep app modules provider-agnostic through shared typed interfaces.

## LLM Provider Architecture

Core abstraction: `lsm/providers/base.py` -> `BaseLLMProvider`.

### Provider Contract

Each provider implements transport-only methods:

- `_send_message(system, user, temperature, max_tokens, **kwargs) -> str`
- `_send_streaming_message(system, user, temperature, max_tokens, **kwargs) -> Iterable[str]`
- `is_available() -> bool`
- `name` property
- `model` property

### Shared Base Logic

`BaseLLMProvider` now owns common LLM business flows:

- `rerank(...)`
- `synthesize(...)`
- `stream_synthesize(...)`
- `generate_tags(...)`
- `_fallback_answer(...)`

This removes duplicated prompt/parsing/retry logic from concrete providers.

### Health and Reliability

`BaseLLMProvider` includes:

- per-provider global health stats (`ProviderHealthStats`)
- centralized success/failure recording
- error categorization (`retryable` vs `fatal`)
- retry helper with exponential backoff
- circuit-breaker cooldown after repeated failures

### Factory and Registry

`lsm/providers/factory.py`:

- `PROVIDER_REGISTRY` maps provider names to lazy `module:ClassName` references
- `create_provider(config)` loads, instantiates, and availability-checks a provider
- `register_provider(name, provider_class)` allows custom provider registration

Built-in provider keys:

- `openai`
- `azure_openai`
- `anthropic` (`claude` alias)
- `gemini`
- `local` (Ollama-compatible)

### Configuration Model

Provider configuration uses the structured `llms` model:

- `llms.providers[]`: connection/auth entries per provider
- `llms.services{}`: named feature mappings to provider + model

Services are then resolved by feature (for example query, ranking, tagging, translation, decomposition) into a concrete `LLMConfig` passed to `create_provider(...)`.

## Vector DB Provider Architecture

Core abstraction: `lsm/vectordb/base.py` -> `BaseVectorDBProvider`.

### Provider Contract

Required methods:

- `add_chunks(ids, documents, metadatas, embeddings)`
- `get(ids=None, filters=None, limit=None, offset=0, include=None) -> VectorDBGetResult`
- `query(embedding, top_k, filters=None) -> VectorDBQueryResult`
- `update_metadatas(ids, metadatas)`
- `delete_by_id(ids)`
- `delete_by_filter(filters)`
- `delete_all() -> int`
- `count() -> int`
- `get_stats() -> Dict[str, Any]`
- `optimize() -> Dict[str, Any]`
- `health_check() -> Dict[str, Any]`
- `is_available() -> bool`
- `name` property

### Result Types

Typed dataclasses normalize provider output:

- `VectorDBQueryResult`: similarity query results (`ids`, `documents`, `metadatas`, `distances`)
- `VectorDBGetResult`: non-similarity retrieval with optional `documents`, `metadatas`, `embeddings`

### Filter Semantics

Provider interface accepts simple metadata filters (`{"key": "value"}`); providers normalize internally for backend-specific query languages.

### Factory and Registry

`lsm/vectordb/factory.py`:

- `create_vectordb_provider(config)` loads and initializes the configured backend
- `register_provider(name, provider_class)` registers custom vector DB providers

Built-in providers:

- `chromadb`
- `postgresql` (`pgvector`)

Migration helpers are available in `lsm/vectordb/migrations/chromadb_to_postgres.py` for ChromaDB -> PostgreSQL moves.

## Extension Workflow

To add a custom provider:

1. Implement the correct base class (`BaseLLMProvider` or `BaseVectorDBProvider`).
2. Register it via the corresponding factory `register_provider(...)`.
3. Configure it in `config.json` (`llms` or `vectordb`).
4. Add tests under `tests/test_providers/` or `tests/test_vectordb/`.
