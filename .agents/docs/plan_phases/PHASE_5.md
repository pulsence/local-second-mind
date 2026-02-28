# Phase 5: LLM Provider Simplification

**Status**: Completed

Transforms LLM providers into pure transport. Removes domain methods (`synthesize`,
`rerank`, `generate_tags`) from `BaseLLMProvider`. Makes `_send_message()` /
`_send_streaming_message()` public with updated signatures. Removes Azure OpenAI provider.
Fixes per-provider implementation gaps.

Reference: [RESEARCH_PLAN.md §6.4, §6.5](../RESEARCH_PLAN.md#64-llm-provider-simplification)

---

## 5.1: BaseLLMProvider Signature Update

**Description**: Make the private transport methods public and update their signatures.
This is the breaking change that all callers must adapt to.

**Tasks**:
- In `lsm/providers/base.py`:
  - Rename `_send_message()` → `send_message()` with new signature:
    ```python
    def send_message(
        self,
        input: str,                                      # was `user`
        instruction: Optional[str] = None,               # was `system`
        prompt: Optional[str] = None,                    # new
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        previous_response_id: Optional[str] = None,      # was in **kwargs
        prompt_cache_key: Optional[str] = None,           # was in **kwargs
        prompt_cache_retention: Optional[int] = None,     # new
        **kwargs,
    ) -> str: ...
    ```
  - Rename `_send_streaming_message()` → `send_streaming_message()` with matching
    signature (returns `Iterator[str]`)
  - Remove the old `_send_message` / `_send_streaming_message` names entirely — no
    backwards compatibility wrappers
- Update all callers of `_send_message()` / `_send_streaming_message()` across the
  entire codebase to use the new public names and signatures:
  - `base.py:synthesize()` — update call (removed entirely in 5.3)
  - `base.py:rerank()` — update call (removed entirely in 5.3)
  - `base.py:generate_tags()` — update call (removed entirely in 5.3)
  - Any other internal callers
- Run: `pytest tests/test_providers/ -v`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/providers/base.py` — method signatures
- All provider implementations — rename abstract method
- `tests/test_providers/` — update tests for new method names

**Success criteria**: `send_message()` and `send_streaming_message()` are public. Old
private names are removed. All provider tests pass with new signatures.

---

## 5.2: Per-Provider Implementation Updates

**Description**: Update each provider's `send_message()` implementation to match the new
signature and fix the gaps identified in §6.5.

**Tasks**:
- **OpenAI** (`lsm/providers/openai.py`):
  - Change `input` from JSON array to plain string (Responses API accepts string directly)
  - Make `instructions` conditional — only include when `instruction` is not `None`
  - Remove `enable_server_cache` gate on `previous_response_id` and `prompt_cache_key`
  - Add `prompt` parameter support
  - Add `prompt_cache_retention` parameter support
  - Update `send_streaming_message()` to support `previous_response_id`,
    `prompt_cache_key`, `prompt`, tools, and JSON schema (streaming parity)
- **Anthropic** (`lsm/providers/anthropic.py`):
  - Map `instruction` → `system` parameter in Anthropic Messages API
  - Map `prompt_cache_key` → `cache_control` headers
  - Log debug message for unsupported params (`previous_response_id`, `prompt`,
    `prompt_cache_retention`) using `UnsupportedParamTracker`
  - Update streaming for cache support
- **Gemini** (`lsm/providers/gemini.py`):
  - Use `system_instruction` config parameter consistently (not concatenation)
  - Make `instruction` optional (only set `system_instruction` when non-None)
  - Log debug message for unsupported caching params
- **OpenRouter** (`lsm/providers/openrouter.py`):
  - Map `prompt_cache_key` → `cache_control` headers
  - Log debug for unsupported params
  - Update streaming for cache support
- **Local** (`lsm/providers/local.py`):
  - Map `instruction` to system role message
  - Log debug for all unsupported caching params
- Write per-provider tests validating parameter mapping
- Write provider caching integration tests:
  - Test: `previous_response_id` round-trip (pass in → get back from response)
  - Test: `prompt_cache_key` is forwarded to provider API correctly
  - Test: streaming path supports the same caching parameters as non-streaming

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/providers/openai.py` — all fixes listed above
- `lsm/providers/anthropic.py` — all fixes listed above
- `lsm/providers/gemini.py` — all fixes listed above
- `lsm/providers/openrouter.py` — all fixes listed above
- `lsm/providers/local.py` — all fixes listed above
- `tests/test_providers/test_openai.py` — input format, instruction conditional, caching
- `tests/test_providers/test_anthropic.py` — system mapping, cache_control
- `tests/test_providers/test_gemini.py` — system_instruction, unsupported param logging
- `tests/test_providers/test_openrouter.py` — cache_control mapping
- `tests/test_providers/test_local.py` — system message mapping
- `tests/test_providers/test_caching_integration.py` — round-trip and streaming parity

**Success criteria**: Every provider correctly maps the new signature to its API. OpenAI
sends `input` as string. `instruction` is optional everywhere. Caching parameters are
unconditional (no enable gate). Streaming has parity with non-streaming for supported params.

---

## 5.3: Remove Domain Methods and Azure Provider

**Description**: Remove `synthesize()`, `stream_synthesize()`, `rerank()`, and
`generate_tags()` from `BaseLLMProvider`. Remove the Azure OpenAI provider entirely.

**Tasks**:
- In `lsm/providers/base.py`:
  - Remove `synthesize()` method
  - Remove `stream_synthesize()` method
  - Remove `rerank()` method
  - Remove `generate_tags()` method
- Remove `lsm/providers/azure_openai.py` entirely
- Update `lsm/providers/factory.py`:
  - Remove Azure OpenAI from provider registry
  - Remove Azure-specific config handling
- Remove Azure-specific config fields from `lsm/config/models/llm.py`:
  - Remove Azure-specific provider fields
  - Remove `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` from config loading
- Update all callers that use the removed methods:
  - `lsm/query/api.py:_synthesize_answer()` — temporarily call `send_message()` directly
    (will be replaced by RetrievalPipeline in Phase 8)
  - `lsm/query/rerank.py` — temporarily call `send_message()` directly
    (will be replaced by pipeline stage in Phase 11)
  - `lsm/ingest/tagging.py` — call `send_message()` directly
- Remove all Azure-specific tests
- Update `.env.example` — remove Azure variables

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/providers/base.py` — remove domain methods
- `lsm/providers/azure_openai.py` — delete file
- `lsm/providers/factory.py` — remove Azure registration
- `lsm/config/models/llm.py` — remove Azure config fields
- `lsm/query/api.py` — update synthesize call
- `lsm/query/rerank.py` — update rerank call
- `lsm/ingest/tagging.py` — update tagging call
- `tests/test_providers/test_azure_openai.py` — delete
- `.env.example` — remove Azure vars

**Success criteria**: `BaseLLMProvider` has only `send_message()`, `send_streaming_message()`,
`is_available()`, `list_models()`, and utility methods. No domain methods exist. Azure
provider is completely removed. All callers updated. Tests pass.

---

## 5.4: Phase 5 Code Review and Changelog

**Tasks**:
- Review that no caller still references `synthesize()`, `rerank()`, `generate_tags()`,
  or `_send_message()`
- Review per-provider parameter mapping for correctness
- Review that `UnsupportedParamTracker` logs are debug-level (not warning)
- Review Azure removal is complete (no dangling imports or config references)
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `.agents/docs/architecture/development/PROVIDERS.md`
- Update `.agents/docs/architecture/api-reference/PROVIDERS.md`
- Update `.agents/docs/architecture/api-reference/ADDING_PROVIDERS.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/PROVIDERS.md`
- `.agents/docs/architecture/api-reference/PROVIDERS.md`
- `.agents/docs/architecture/api-reference/ADDING_PROVIDERS.md`

**Success criteria**: `pytest tests/ -v` passes. No references to removed methods or Azure
provider. Changelog and docs updated.

---
