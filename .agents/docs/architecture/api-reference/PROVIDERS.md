# LLM Provider API Reference

This document defines the `BaseLLMProvider` contract and related factory APIs.

## BaseLLMProvider

Location: `lsm/providers/base.py`

### Methods

#### rerank(question, candidates, k, **kwargs)

- `question: str`
- `candidates: list[dict]`
  - `text: str`
  - `metadata: dict`
  - `distance: float | None`
- `k: int`

Returns: `list[dict]` in the same candidate shape, ordered by relevance.

#### synthesize(question, context, mode="grounded", **kwargs)

- `question: str`
- `context: str` (contains `[S#]` source blocks)
- `mode: str` (`grounded` or `insight`)

Returns: `str` answer with inline citations.

#### generate_tags(text, num_tags=3, existing_tags=None, **kwargs)

- `text: str`
- `num_tags: int`
- `existing_tags: list[str] | None`

Returns: `list[str]` tags.

#### is_available()

Returns: `bool` (true if provider is configured).

#### health_check()

Returns: `dict` with availability, status, and recent success/failure stats.

### Properties

- `name: str` (provider name)
- `model: str` (model name)

### Optional

- `estimate_cost(input_tokens, output_tokens) -> float | None`

## Factory APIs

Location: `lsm/providers/factory.py`

### create_provider(config: LLMConfig) -> BaseLLMProvider

Creates and returns a provider instance using the registry.

### list_available_providers() -> list[str]

Returns provider names registered in `PROVIDER_REGISTRY`.

### register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None

Registers a custom provider at runtime.

## OpenAI Provider Behavior

Location: `lsm/providers/openai.py`

- Uses OpenAI Responses API for rerank and synthesize.
- Handles unsupported parameters by retrying without them.
- Provides rough cost estimation.

## OpenRouter Provider Behavior

Location: `lsm/providers/openrouter.py`

- Uses OpenRouter's OpenAI-compatible Chat Completions API.
- Supports routing with fallback models via `llms.providers[].fallback_models`.
- Emits prompt caching markers when `enable_llm_server_cache` is enabled.
- Captures response usage metadata for token tracking.

## Error Handling

Providers should raise exceptions for API errors. The query pipeline catches
rerank or synthesis errors and falls back when possible.
