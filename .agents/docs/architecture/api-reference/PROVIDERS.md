# LLM Provider API Reference

This document defines the `BaseLLMProvider` transport contract and factory APIs.

## BaseLLMProvider

Location: `lsm/providers/base.py`

### Required Methods

#### `send_message(...) -> str`

```python
def send_message(
    self,
    input: str,
    instruction: str | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 4096,
    previous_response_id: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: int | None = None,
    **kwargs,
) -> str
```

#### `send_streaming_message(...) -> Iterable[str]`

Signature matches `send_message(...)` and yields text chunks.

#### `is_available() -> bool`

Returns whether provider credentials/client setup are usable.

### Required Properties

- `name: str`
- `model: str`

### Optional Utility Methods

- `list_models() -> list[str]`
- `get_model_pricing() -> dict[str, float] | None`
- `estimate_cost(input_tokens: int, output_tokens: int) -> float | None`
- `health_check() -> dict[str, Any]`

### Notes

- Provider adapters are transport-only.
- Domain logic (`rerank`, `synthesize`, tagging prompts/parsers) lives in query/ingest modules, not in provider classes.

## Factory APIs

Location: `lsm/providers/factory.py`

### `create_provider(config: LLMConfig) -> BaseLLMProvider`

Constructs a provider instance from registry + resolved config.

### `list_available_providers() -> list[str]`

Returns registered provider names.

### `register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None`

Registers a custom provider class at runtime.

## Built-In Providers

- OpenAI (`lsm/providers/openai.py`)
- OpenRouter (`lsm/providers/openrouter.py`)
- Anthropic (`lsm/providers/anthropic.py`)
- Gemini (`lsm/providers/gemini.py`)
- Local/Ollama (`lsm/providers/local.py`)

## Error Handling

Providers raise transport/API exceptions. Higher-level query/ingest pipelines decide retries, fallback behavior, and user-facing failure handling.
