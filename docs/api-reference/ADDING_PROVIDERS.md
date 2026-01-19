# Adding New LLM Providers to LSM

This guide explains how to add support for new LLM providers to Local Second Mind. The provider system is designed to be extensible and follows a plugin-like architecture.

## Table of Contents

- [Overview](#overview)
- [Provider Interface](#provider-interface)
- [Implementation Guide](#implementation-guide)
- [Example: Anthropic Claude](#example-anthropic-claude)
- [Testing Your Provider](#testing-your-provider)
- [Registration](#registration)
- [Best Practices](#best-practices)

## Overview

LSM uses an abstract provider interface ([BaseLLMProvider](../lsm/providers/base.py)) that all LLM providers must implement. This abstraction allows LSM to support multiple LLM backends while keeping the core query logic provider-agnostic.

### Current Providers

- **OpenAI** - GPT models (gpt-5.2, gpt-4o-mini, etc.)

### Planned Providers

- **Anthropic Claude** - Claude 3 family (Opus, Sonnet, Haiku)
- **Local Models** - Via Ollama, llama.cpp, or transformers
- **Azure OpenAI** - OpenAI models via Azure
- **Cohere** - Command models
- **Google Vertex AI** - PaLM and Gemini models

## Provider Interface

All providers must inherit from `BaseLLMProvider` and implement the following methods:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using the LLM.

        Args:
            question: User's question
            candidates: List of candidate dicts with 'text' and 'metadata'
            k: Number of candidates to return
            **kwargs: Provider-specific options

        Returns:
            Reranked list of candidates (top k)
        """
        pass

    @abstractmethod
    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        """
        Generate an answer with citations.

        Args:
            question: User's question
            context: Context block with source citations
            mode: Query mode ('grounded' or 'insight')
            **kwargs: Provider-specific options

        Returns:
            Generated answer text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is configured and available.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Current model name (e.g., 'gpt-5.2', 'claude-3-sonnet')."""
        pass

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate the cost for a request (optional).

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD, or None if not available
        """
        return None  # Default implementation
```

## Implementation Guide

### Step 1: Create Provider File

Create a new file in `lsm/providers/` for your provider:

```bash
lsm/providers/your_provider.py
```

### Step 2: Implement the Interface

```python
"""
Your Provider implementation.

Description of your provider and any special considerations.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from lsm.config.models import LLMConfig
from lsm.gui.shell.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class YourProvider(BaseLLMProvider):
    """
    Your Provider implementation.

    Configuration:
        provider: your_provider
        model: your-model-name
        api_key: ${YOUR_PROVIDER_API_KEY}
        temperature: 0.7
        max_tokens: 2000
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the provider.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._client = None
        self._api_key = config.api_key or os.getenv("YOUR_PROVIDER_API_KEY")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "your_provider"

    @property
    def model(self) -> str:
        """Return current model name."""
        return self.config.model

    def is_available(self) -> bool:
        """Check if provider is configured."""
        return self._api_key is not None

    @property
    def client(self):
        """Lazy-load the client."""
        if self._client is None:
            if not self.is_available():
                raise ValueError(
                    f"API key not configured for {self.name}. "
                    f"Set YOUR_PROVIDER_API_KEY environment variable."
                )
            # Initialize your client here
            # self._client = YourProviderClient(api_key=self._api_key)
            pass
        return self._client

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using LLM.

        Implementation details:
        1. Build prompt with question and candidates
        2. Call LLM API to score relevance
        3. Sort by score and return top k
        """
        logger.debug(f"Reranking {len(candidates)} candidates with {self.name}")

        # Build reranking prompt
        prompt = self._build_rerank_prompt(question, candidates)

        # Call your LLM API
        # response = self.client.complete(prompt, ...)

        # Parse response and reorder candidates
        # reranked = self._parse_rerank_response(response, candidates)

        # Return top k
        # return reranked[:k]

        # Placeholder implementation:
        return candidates[:k]

    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        """
        Generate an answer with citations.

        Implementation details:
        1. Select system prompt based on mode
        2. Build user message with question and context
        3. Call LLM API for generation
        4. Extract and return answer text
        """
        logger.debug(f"Synthesizing answer with {self.name} (mode={mode})")

        # Get mode-specific system prompt
        system_prompt = self._get_system_prompt(mode)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}
        ]

        # Call your LLM API
        # response = self.client.chat(
        #     messages=messages,
        #     temperature=self.config.temperature,
        #     max_tokens=self.config.max_tokens,
        #     **kwargs
        # )

        # Extract answer
        # answer = response['choices'][0]['message']['content']

        # Placeholder implementation:
        answer = "Answer not implemented yet"

        return answer

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate cost based on your provider's pricing.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Example pricing (update with actual rates)
        PRICING = {
            "your-model-name": {
                "input": 0.01 / 1000,   # $0.01 per 1K input tokens
                "output": 0.03 / 1000,  # $0.03 per 1K output tokens
            }
        }

        if self.model in PRICING:
            rates = PRICING[self.model]
            cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])
            return cost

        return None

    def _get_system_prompt(self, mode: str) -> str:
        """Get system prompt based on query mode."""
        if mode == "grounded":
            return """You are a research assistant. Answer the user's question using ONLY
information from the provided sources. Every claim must be cited with [S#].
If the sources don't contain the answer, say so clearly."""

        elif mode == "insight":
            return """You are a research analyst. Analyze the provided sources to identify:
- Recurring themes and patterns
- Contradictions or tensions
- Gaps or open questions
- Evolution of ideas across documents

Cite sources [S#] when referencing specific passages."""

        elif mode == "hybrid":
            return """You are a knowledgeable research assistant. Answer the question by:
1. First consulting the provided local sources (cite with [S#])
2. Then supplementing with your general knowledge (cite as [Model Knowledge])
3. Finally consulting web sources if provided (cite as [W#])

Clearly distinguish between source types in your answer."""

        else:
            return "You are a helpful assistant."

    def _build_rerank_prompt(self, question: str, candidates: List[Dict[str, Any]]) -> str:
        """Build prompt for reranking candidates."""
        prompt = f"Question: {question}\\n\\n"
        prompt += "Rank the following passages by relevance (most relevant first):\\n\\n"

        for i, cand in enumerate(candidates):
            text = cand["text"][:300]  # Truncate for efficiency
            prompt += f"[{i}] {text}...\\n\\n"

        prompt += "Return only the indices in order of relevance (comma-separated)."
        return prompt
```

### Step 3: Register Your Provider

Add your provider to the registry in `lsm/providers/factory.py`:

```python
from .your_provider import YourProvider

PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "your_provider": YourProvider,  # Add this line
}
```

### Step 4: Update Configuration Schema

Users should be able to configure your provider in `config.json`:

```json
{
  "llms": [
    {
      "provider_name": "your_provider",
      "api_key": "INSERT_YOUR_API_KEY_OR_USE_ENV_VAR",
      "query": { "model": "your-model-name" },
      "tagging": { "model": "your-model-name" },
      "ranking": { "model": "your-model-name" }
    }
  ]
}
```

## Example: Anthropic Claude

Here's a complete example for adding Anthropic Claude support:

```python
"""
Anthropic Claude provider implementation.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from lsm.config.models import LLMConfig
from lsm.gui.shell.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Configuration:
        provider: anthropic
        model: claude-3-sonnet-20240229
        api_key: ${ANTHROPIC_API_KEY}
        temperature: 0.7
        max_tokens: 2000
    """

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "claude-3-opus-20240229": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
        "claude-3-sonnet-20240229": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-haiku-20240307": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            if not self.is_available():
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it in config or as environment variable."
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        logger.debug(f"Reranking {len(candidates)} candidates with Claude")

        # Build reranking prompt
        prompt = f"{HUMAN_PROMPT} Question: {question}\\n\\n"
        prompt += "Rank the following passages by relevance to the question. "
        prompt += "Return ONLY the indices (comma-separated, most relevant first):\\n\\n"

        for i, cand in enumerate(candidates):
            text = cand["text"][:300]
            prompt += f"[{i}] {text}...\\n\\n"

        prompt += f"{AI_PROMPT}"

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse indices
            indices_str = response.content[0].text.strip()
            indices = [int(x.strip()) for x in indices_str.split(",")]

            # Reorder candidates
            reranked = [candidates[i] for i in indices if i < len(candidates)]

            # Fill in any missing candidates
            missing = [c for i, c in enumerate(candidates) if i not in indices]
            reranked.extend(missing)

            return reranked[:k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:k]

    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        logger.debug(f"Synthesizing answer with Claude (mode={mode})")

        # Build prompt
        system = self._get_system_prompt(mode)
        prompt = f"{HUMAN_PROMPT} {system}\\n\\n"
        prompt += f"Context:\\n{context}\\n\\n"
        prompt += f"Question: {question}{AI_PROMPT}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            answer = response.content[0].text
            return answer

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        if self.model in self.PRICING:
            rates = self.PRICING[self.model]
            cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])
            return cost
        return None

    def _get_system_prompt(self, mode: str) -> str:
        # Same as OpenAI provider
        if mode == "grounded":
            return """You are a research assistant. Answer using ONLY the provided sources.
Every claim must be cited with [S#]. If sources don't contain the answer, say so."""

        elif mode == "insight":
            return """Analyze the provided sources to identify recurring themes, contradictions,
gaps, and evolution of ideas. Cite sources [S#] when referencing specific passages."""

        else:
            return "You are a helpful research assistant."
```

## Testing Your Provider

### Unit Tests

Create tests in `tests/test_query/test_providers.py`:

```python
import pytest
from lsm.config.models import LLMConfig
from lsm.providers import YourProvider


def test_your_provider_initialization():
    config = LLMConfig(provider="your_provider", model="your-model")
    provider = YourProvider(config)

    assert provider.name == "your_provider"
    assert provider.model == "your-model"


def test_your_provider_availability():
    config = LLMConfig(provider="your_provider", model="your-model", api_key="test_key")
    provider = YourProvider(config)

    assert provider.is_available() is True


def test_your_provider_without_api_key():
    config = LLMConfig(provider="your_provider", model="your-model")
    provider = YourProvider(config)

    # Should not be available without API key
    assert provider.is_available() is False


@pytest.mark.integration
def test_your_provider_synthesis():
    """Integration test with real API (requires API key)."""
    config = LLMConfig(provider="your_provider", model="your-model")
    provider = YourProvider(config)

    if not provider.is_available():
        pytest.skip("API key not configured")

    context = "[S1] The capital of France is Paris."
    question = "What is the capital of France?"

    answer = provider.synthesize(question, context, mode="grounded")

    assert "Paris" in answer
    assert "[S1]" in answer  # Should cite the source
```

### Manual Testing

1. Add your provider to config:
```json
{
  "llms": [
    {
      "provider_name": "your_provider",
      "api_key": "your_api_key",
      "query": { "model": "your-model-name" },
      "tagging": { "model": "your-model-name" },
      "ranking": { "model": "your-model-name" }
    }
  ]
}
```

2. Test with query command:
```bash
lsm query "Test question"
```

3. Verify in the TUI:
```bash
lsm
> /query
[query] > What is X?
# Should use your provider
```

## Registration

### Static Registration

Add to `factory.py` (recommended for built-in providers):

```python
PROVIDER_REGISTRY["your_provider"] = YourProvider
```

### Dynamic Registration

For custom/plugin providers, users can register at runtime:

```python
from lsm.providers import register_provider
from my_custom_provider import MyCustomProvider

register_provider("mycustom", MyCustomProvider)
```

## Best Practices

### 1. Error Handling

Always handle API errors gracefully:

```python
try:
    response = self.client.complete(prompt)
except APIError as e:
    logger.error(f"API error: {e}")
    raise
except Timeout as e:
    logger.error(f"Request timeout: {e}")
    raise
```

### 2. Rate Limiting

Respect provider rate limits:

```python
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def _call_api(self, prompt: str) -> str:
    return self.client.complete(prompt)
```

### 3. Token Counting

Estimate tokens accurately for cost estimation:

```python
def count_tokens(text: str, model: str) -> int:
    # Use provider-specific tokenizer
    # For Claude: roughly 4 chars per token
    # For GPT: use tiktoken library
    pass
```

### 4. Logging

Log important events for debugging:

```python
logger.debug(f"Calling {self.name} with {len(prompt)} chars")
logger.info(f"Synthesis completed in {elapsed:.2f}s")
logger.warning(f"High token count: {tokens}")
```

### 5. Configuration Validation

Validate configuration early:

```python
def __init__(self, config: LLMConfig):
    self.config = config

    # Validate model name
    if not self._is_valid_model(config.model):
        raise ValueError(f"Invalid model: {config.model}")

    # Validate parameters
    if config.temperature < 0 or config.temperature > 2:
        raise ValueError(f"Temperature must be in [0, 2]")
```

### 6. Documentation

Document your provider thoroughly:

```python
class YourProvider(BaseLLMProvider):
    """
    Your Provider implementation.

    Supported Models:
        - model-1: Description
        - model-2: Description

    Configuration Example:
        ```json
        {
          "llms": [
            {
              "provider_name": "your_provider",
              "api_key": "${YOUR_PROVIDER_API_KEY}",
              "query": { "model": "model-1" },
              "tagging": { "model": "model-1" },
              "ranking": { "model": "model-1" }
            }
          ]
        }
        ```

    Environment Variables:
        YOUR_PROVIDER_API_KEY: API key for authentication

    Rate Limits:
        - Free tier: 10 requests/minute
        - Pro tier: 100 requests/minute

    Pricing:
        - model-1: $0.01/1K input tokens, $0.03/1K output tokens
        - model-2: $0.005/1K input tokens, $0.015/1K output tokens
    """
```

## Troubleshooting

### Provider Not Found

Error: `Unsupported LLM provider: 'your_provider'`

**Solution:** Ensure your provider is registered in `PROVIDER_REGISTRY`.

### API Key Issues

Error: `API key not configured for your_provider`

**Solution:**
1. Set environment variable: `export YOUR_PROVIDER_API_KEY=your_key`
2. Or add to config: `"api_key": "your_key"`

### Import Errors

Error: `ModuleNotFoundError: No module named 'your_provider_sdk'`

**Solution:** Add the SDK to dependencies in `pyproject.toml`:

```toml
[project.dependencies]
your-provider-sdk = "^1.0.0"
```

## Contributing

To contribute a new provider to LSM:

1. Fork the repository
2. Create a new branch: `git checkout -b add-your-provider`
3. Implement your provider following this guide
4. Add tests (aim for >80% coverage)
5. Update documentation
6. Submit a pull request

## References

- [BaseLLMProvider Source](../lsm/providers/base.py)
- [OpenAI Provider Implementation](../lsm/providers/openai.py)
- [Provider Factory](../lsm/providers/factory.py)
- [Configuration Schema](../lsm/config/models.py)

## Support

For questions or issues:
- Open an issue on GitHub
- Check existing provider implementations for examples
- Review the provider abstraction design in `base.py`


