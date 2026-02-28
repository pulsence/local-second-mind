# Adding New LLM Providers

This guide covers how to add a new LLM provider adapter to Local Second Mind.

## Overview

LLM providers in LSM are transport adapters. They do not own reranking, synthesis, or tagging logic.

Core interface: `lsm/providers/base.py::BaseLLMProvider`

## Current Built-In Providers

- `openai`
- `openrouter`
- `anthropic`
- `gemini`
- `local`

## Implementation Steps

### 1. Create the provider module

Add `lsm/providers/<your_provider>.py`.

### 2. Implement `BaseLLMProvider`

Minimum required members:

- `name` property
- `model` property
- `is_available()`
- `send_message(...)`
- `send_streaming_message(...)`

Template:

```python
from __future__ import annotations

from typing import Iterable, Optional

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class YourProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        # initialize lazy client fields here
        super().__init__()

    @property
    def name(self) -> str:
        return "your_provider"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self.config.api_key)

    def send_message(
        self,
        input: str,
        instruction: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        previous_response_id: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_retention: Optional[int] = None,
        **kwargs,
    ) -> str:
        # map BaseLLMProvider args to vendor API args
        # return assistant text
        raise NotImplementedError

    def send_streaming_message(
        self,
        input: str,
        instruction: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        previous_response_id: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_retention: Optional[int] = None,
        **kwargs,
    ) -> Iterable[str]:
        # keep parameter parity with send_message
        # yield incremental chunks
        raise NotImplementedError
```

### 3. Register in factory

Update `lsm/providers/factory.py`:

- import your provider class
- add it to `PROVIDER_REGISTRY`

### 4. Ensure config compatibility

Your provider should work with `llms.providers[]` + `llms.services{}` resolution:

```json
{
  "llms": {
    "providers": [
      {
        "provider_name": "your_provider",
        "api_key": "${YOUR_PROVIDER_API_KEY}"
      }
    ],
    "services": {
      "default": { "provider": "your_provider", "model": "your-model" }
    }
  }
}
```

### 5. Add tests

Recommended test files:

- `tests/test_providers/test_<your_provider>.py`
- add mapping/parity tests for `send_message` and `send_streaming_message`
- add caching/unsupported-parameter behavior tests as needed

## Best Practices

- Keep provider classes transport-only.
- Keep unsupported parameter handling at debug log level.
- Preserve argument parity between `send_message` and `send_streaming_message`.
- Use `UnsupportedParamTracker` for model-specific unsupported argument detection.
- Let query/ingest modules own prompts, schemas, and parsing.
