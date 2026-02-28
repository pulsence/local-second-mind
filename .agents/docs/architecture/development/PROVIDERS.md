# Provider System Architecture

This document describes provider abstractions used by Local Second Mind.

## Design Goals

- Keep business/domain logic outside provider adapters.
- Keep provider adapters transport-focused and easy to test.
- Expose stable interfaces so query/ingest/app layers stay provider-agnostic.

## LLM Provider Architecture

Core abstraction: `BaseLLMProvider` in `lsm/providers/base.py`.

### Transport-Only Contract

Each LLM provider implements only transport methods:

- `send_message(...) -> str`
- `send_streaming_message(...) -> Iterable[str]`
- `is_available() -> bool`
- `name` / `model` properties
- Optional utilities (`list_models`, `estimate_cost`, health/retry helpers)

Domain methods such as `synthesize`, `rerank`, and `generate_tags` are not part
of provider adapters.

### Domain Ownership

Prompts, JSON schemas, and response parsing are owned by feature modules:

- Query synthesis instructions: `lsm/query/prompts.py`
- Query synthesis orchestration: `lsm/query/api.py`
- LLM rerank stage: `lsm/query/stages/llm_rerank.py`
- Query fallback answer generation: `lsm/query/fallback.py`
- Tag generation prompt + parsing: `lsm/ingest/tagging.py`
- Translation prompt + parsing: `lsm/ingest/translation.py`

`lsm/providers/helpers.py` intentionally contains only provider-generic helpers:

- `parse_json_payload(...)`
- `UnsupportedParamTracker`

### Registry / Factory

`lsm/providers/factory.py` registers provider adapters and constructs concrete
instances from resolved `LLMConfig`.

Supported LLM providers:

- `openai`
- `openrouter`
- `anthropic`
- `gemini`
- `local`

## Vector DB Provider Architecture

Core abstraction: `BaseVectorDBProvider` in `lsm/vectordb/base.py`.

Vector DB adapters provide storage/search/filter primitives while ingest/query
layers own higher-level orchestration.

## Extension Workflow

To add a provider:

1. Implement the relevant base class (`BaseLLMProvider` or `BaseVectorDBProvider`).
2. Register it in the matching factory.
3. Wire config resolution through `llms`/`vectordb` config models.
4. Add unit + integration tests in the corresponding `tests/` areas.
