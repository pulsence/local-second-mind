# Provider System Architecture

This document describes the provider abstractions used by LSM for LLM calls and vector DBs.

## Goals

- Encapsulate provider-specific APIs behind a stable interface.
- Support multiple LLM backends (OpenAI today; others in the future).
- Allow per-feature overrides (query, ranking, tagging).

## LLM Provider Interface

All providers implement `BaseLLMProvider`:

- `rerank(question, candidates, k, **kwargs)`
- `synthesize(question, context, mode, **kwargs)`
- `generate_tags(text, num_tags, existing_tags, **kwargs)`
- `is_available()`
- `health_check()`
- `name` and `model` properties

Optional:

- `estimate_cost(input_tokens, output_tokens)`

See `docs/api-reference/PROVIDERS.md` for method signatures.

## LLM Factory Pattern

Providers are created through a registry in `lsm/providers/factory.py`:

- `PROVIDER_REGISTRY` maps provider names to classes.
- `create_provider(config)` instantiates and validates availability.
- `register_provider(name, provider_class)` allows custom providers.

## OpenAI Provider

The OpenAI provider:

- Uses the Responses API via `openai.OpenAI`.
- Supports structured output for reranking and tagging when possible.
- Handles unsupported parameter errors by retrying without them.
- Provides a simple cost estimate per model family.

### Reranking

- Sends candidate snippets and a strict JSON schema.
- Returns top-k candidates with reasons (reasons are not used downstream).

### Synthesis

- Uses mode-specific instructions (`grounded` or `insight`).
- Uses `temperature` and `max_tokens` unless unsupported by model.

### Tagging

- Emits JSON tags and attempts to recover from malformed responses.

## Per-Feature Overrides

`LLMConfig` supports overrides for:

- `query` (synthesis)
- `ranking` (reranking)
- `tagging` (AI tag generation)

These overrides inherit from the base `llm` config when fields are not provided.

## Availability Checks

`is_available()` returns true if the API key is present or can be resolved from
environment variables.

## Extension Points

To add a provider:

1. Implement `BaseLLMProvider`.
2. Register it in `PROVIDER_REGISTRY` (or via `register_provider`).
3. Provide configuration under `llm.provider`.

See `docs/api-reference/ADDING_PROVIDERS.md` for a worked example.

## Vector DB Provider Interface

Vector DB providers implement `BaseVectorDBProvider`:

- `add_chunks(ids, documents, metadatas, embeddings)`
- `query(embedding, top_k, filters)`
- `delete_by_id(ids)`
- `delete_by_filter(filters)`
- `count()`
- `get_stats()`
- `optimize()`
- `health_check()`
- `name` property

See `lsm/vectordb/base.py` for method signatures.

## Vector DB Factory Pattern

Providers are created through a registry in `lsm/vectordb/factory.py`:

- `PROVIDER_REGISTRY` maps provider names to classes.
- `create_vectordb_provider(config)` instantiates and validates availability.
- `register_provider(name, provider_class)` allows custom providers.

## ChromaDB Provider

The ChromaDB provider:

- Uses persistent storage from `vectordb.persist_dir`.
- Supports `chroma_hnsw_space` configuration.
- Exposes `get_collection()` for legacy Chroma-only operations.

## PostgreSQL + pgvector Provider

The PostgreSQL provider:

- Uses `psycopg2` pooling and `pgvector` bindings.
- Creates schema/indexes on first use.
- Supports `hnsw` or `ivfflat` indexing.

See `docs/user-guide/VECTOR_DATABASES.md` for configuration examples.
