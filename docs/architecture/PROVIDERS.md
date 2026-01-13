# Provider System Architecture

This document describes the provider abstraction used by LSM for LLM calls.

## Goals

- Encapsulate provider-specific APIs behind a stable interface.
- Support multiple LLM backends (OpenAI today; others in the future).
- Allow per-feature overrides (query, ranking, tagging).

## Provider Interface

All providers implement `BaseLLMProvider`:

- `rerank(question, candidates, k, **kwargs)`
- `synthesize(question, context, mode, **kwargs)`
- `generate_tags(text, num_tags, existing_tags, **kwargs)`
- `is_available()`
- `name` and `model` properties

Optional:

- `estimate_cost(input_tokens, output_tokens)`

See `docs/api-reference/PROVIDERS.md` for method signatures.

## Factory Pattern

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
