# Local Second Mind v0.5.0 Development Plan

## Context

v0.5.0 focuses on three pillars: (1) eliminating ~1500 lines of duplicated LLM provider code, (2) building a rock-solid test framework with real data and live service tests, and (3) hardening the agent sandbox with comprehensive security testing and framework flexibility for future agents and tools.

**Ordering rationale**: LLM providers are refactored first so tests written afterward don't need rewriting. The test framework comes second to establish the validation infrastructure the agent security hardening depends on.

**Mock policy**: Real tests are the primary suite. A small set of fast "smoke" tests using lightweight fakes (not full mocks) are kept for CI speed. All integration and live tests use real services.

**Docker policy**: Primary focus on hardening the LocalRunner. A preliminary DockerRunner foundation is created for future high-risk tools/agents, but no tools or agents requiring Docker are built in this version.

---

## Phase 1: LLM Provider Refactoring (COMPLETED)

Five providers (OpenAI 569 LOC, Azure OpenAI 432, Anthropic 344, Gemini 288, Local 262) independently implement identical business logic for `rerank()`, `synthesize()`, `stream_synthesize()`, and `generate_tags()`. The only differences are API call mechanics and response extraction. This phase consolidates shared logic into `BaseLLMProvider` and reduces each provider to its API-specific core.

### 1.1: Define Base Message Interface (COMPLETED)

Add two abstract methods and one concrete helper to `BaseLLMProvider`.

**Tasks:**
- Add `_send_message(system, user, temperature, max_tokens, **kwargs) -> str` as abstract method. Each provider implements this with its own SDK/HTTP client. Must set `self.last_response_id` if provider supports it. The `**kwargs` carry provider-specific extras: `reasoning_effort`, `enable_server_cache`, `previous_response_id`, `prompt_cache_key`, `json_schema`.
- Add `_send_streaming_message(system, user, temperature, max_tokens, **kwargs) -> Iterable[str]` as abstract method. Same signature, yields text chunks.
- Move `_fallback_answer()` from each provider to base class as concrete method using `self.name`.

**Files to modify:**
- `lsm/providers/base.py`

**Post-block:** tests in `tests/test_providers/test_base.py`, run `pytest tests/ -v`, update `docs/`, commit.

### 1.2: Move Business Logic to Base Class as Concrete Methods (COMPLETED)

Convert `rerank()`, `synthesize()`, `stream_synthesize()`, `generate_tags()` from abstract to concrete in `BaseLLMProvider`. They format prompts using existing `helpers.py` utilities and delegate to `_send_message` / `_send_streaming_message`.

#### 1.2.1: Concrete rerank()

- Remove `@abstractmethod`
- Call `prepare_candidates_for_rerank()`, format `RERANK_INSTRUCTIONS`, build JSON payload
- Call `self._send_message(system=instructions, user=json.dumps(payload), temperature=0.2, max_tokens=400, json_schema=RERANK_SCHEMA)`
- Parse with `parse_json_payload()` and `parse_ranking_response()`
- Record success/failure, fall back to `candidates[:k]` on error

#### 1.2.2: Concrete synthesize()

- Get instructions via `get_synthesis_instructions(mode)`, build user content via `format_user_content()`
- Call `self._send_message(system=instructions, user=user_content, temperature, max_tokens, **kwargs)`
- Record success, fallback on error

#### 1.2.3: Concrete stream_synthesize()

- Same setup as synthesize, calls `_send_streaming_message()`, yields chunks

#### 1.2.4: Concrete generate_tags()

- Get instructions via `get_tag_instructions()`, user content is `f"Text:\n{text[:2000]}"`
- Call `_send_message()` with `max_tokens=min(max_tokens, 200)`, `json_schema=TAGS_SCHEMA`
- Parse JSON response, clean and return tags

**Files to modify:**
- `lsm/providers/base.py`
- `lsm/providers/helpers.py` (add `RERANK_JSON_SCHEMA` and `TAGS_JSON_SCHEMA` constants)

**Post-block:** tests for each concrete method in `tests/test_providers/test_base.py` using a minimal test provider subclass, run `pytest tests/ -v`, update docs, commit.

### 1.3: Refactor Each Provider (COMPLETED)

Each provider is reduced to: `__init__`, `name`, `model`, `is_available()`, `list_models()`, `get_model_pricing()`, `_send_message()`, `_send_streaming_message()`, and provider-specific helpers. All duplicated `rerank/synthesize/stream_synthesize/generate_tags/_fallback_answer` implementations are deleted.

#### 1.3.1: Refactor OpenAIProvider

`_send_message()` builds request_args with `model`, `reasoning`, `instructions`, `input`, `max_output_tokens`. Handles `UnsupportedParamTracker` for temperature, `text.format` for structured output, server caching via `previous_response_id`. Calls `self._call_responses()`.

`_send_streaming_message()` similar but creates stream, yields deltas.

**Expected reduction:** ~569 -> ~180 lines.

#### 1.3.2: Refactor AzureOpenAIProvider

Mirrors OpenAI but uses `self.deployment_name` as model and `AzureOpenAI` client.

**Expected reduction:** ~432 -> ~150 lines.

#### 1.3.3: Refactor AnthropicProvider

`_send_message()` handles `cache_control` system payload, calls `client.messages.create()`, extracts via `_extract_text()`. Streaming uses `client.messages.stream()`.

**Expected reduction:** ~344 -> ~120 lines.

#### 1.3.4: Refactor GeminiProvider

`_send_message()` concatenates system+user, calls `client.models.generate_content()` with `_build_config()`. Already has `_generate()` helper that maps directly.

**Expected reduction:** ~288 -> ~100 lines.

#### 1.3.5: Refactor LocalProvider

Already has `_chat()` and `_chat_stream()` that map directly to the new interface. Mostly renaming.

**Expected reduction:** ~262 -> ~100 lines.

**Files to modify:**
- `lsm/providers/openai.py`
- `lsm/providers/azure_openai.py`
- `lsm/providers/anthropic.py`
- `lsm/providers/gemini.py`
- `lsm/providers/local.py`

**Post-block:** Update all provider test files, run `pytest tests/ -v`, update docs, commit.

### 1.4: Validation and Cleanup (COMPLETED)

- Verify provider factory creates all providers correctly
- Ensure no backward-compatible code remains in provider implementations
- Run full test suite, fix regressions
- Verify `lsm/providers/__init__.py` exports

**Post-block:** run `pytest tests/ -v`, commit.

---

## Phase 2: Improved Test Framework (COMPLETED)

Current state: 119 test files, ~1128 tests, almost entirely mock-based. Minimal test data (4 small files). No live service tests, no custom test config, no smoke/live tier separation.

### 2.1: Test Infrastructure and Configuration (COMPLETED)

#### 2.1.1: Create Test Configuration System

Load test configuration from `tests/.env.test` (gitignored) and environment variables with `LSM_TEST_` prefix. Configuration options:

- `LSM_TEST_CONFIG` - path to custom config.json for test suite
- `LSM_TEST_OPENAI_API_KEY`, `LSM_TEST_ANTHROPIC_API_KEY`, `LSM_TEST_GOOGLE_API_KEY` - provider credentials
- `LSM_TEST_OLLAMA_BASE_URL` - Ollama URL for local model tests
- `LSM_TEST_BRAVE_API_KEY` - Brave Search API key
- `LSM_TEST_SEMANTIC_SCHOLAR_API_KEY` - Semantic Scholar API key (optional, higher rate limits)
- `LSM_TEST_CORE_API_KEY` - CORE API key
- `LSM_TEST_EMBED_MODEL` - embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `LSM_TEST_TIER` - default tier: `smoke` | `integration` | `live` (default: `smoke`)

**Files to create:**
- `tests/test_config.py` - test configuration loader
- `tests/.env.test.example` - template

#### 2.1.2: Define Test Markers and Tiers

**Files to modify:**
- `pyproject.toml` - add markers

New markers:
- `smoke` - Fast tests, no live services, lightweight fakes OK. Default tier.
- `integration` - Real embeddings, real ChromaDB, real file parsing. No network.
- `live` - Actual API calls to LLM providers and remote sources.
- `live_llm` - Subset: LLM API calls specifically.
- `live_remote` - Subset: remote provider calls specifically.
- `docker` - Requires Docker.

Update `addopts` to default exclude `live` and `docker`: `-m "not live and not docker"`.

Invocations:
```
pytest tests/ -v                              # smoke + integration (no live/docker)
pytest tests/ -v -m "live"                    # only live tests
pytest tests/ -v -m ""                        # everything
pytest tests/ -v -m "live_llm"               # only live LLM tests
```

#### 2.1.3: Create Tier-Aware Conftest Fixtures

**Files to modify:**
- `tests/conftest.py`

New fixtures:
- `test_config` (session-scoped) - loaded from `tests/test_config.py`
- `real_embedder` (session-scoped) - loads actual `all-MiniLM-L6-v2`, cached across session
- `real_chromadb_provider(tmp_path)` - real ChromaDB against temp dir
- `real_openai_provider(test_config)` - real OpenAI, `pytest.skip()` if no key
- `real_anthropic_provider(test_config)` - same pattern
- `real_gemini_provider(test_config)` - same pattern
- `real_local_provider(test_config)` - same pattern
- `populated_chromadb(real_chromadb_provider, real_embedder, rich_test_corpus)` - ChromaDB with ingested test data

**Post-block:** `tests/test_infrastructure/test_test_config.py`, run tests, update `docs/development/TESTING.md`, commit.

### 2.2: Build Comprehensive Test Data Corpus (COMPLETED)

Current fixtures: 4 tiny files. Need rich, diverse data for realistic testing.

#### 2.2.1: Create Text Document Corpus

**Files to create in `tests/fixtures/synthetic_data/documents/`:**
- `philosophy_essay.txt` - ~2000 words, multi-paragraph, clear themes (epistemology/knowledge management, thematically relevant to LSM)
- `research_paper.md` - ~3000 words, proper Markdown: H2/H3 headings, bullet lists, code block, inline citations `[1]`, references section
- `technical_manual.html` - ~1500 words, nested divs, tables, lists, headings, bold/italic

#### 2.2.2: Create Edge Case Documents

- `short_note.txt` - 2 sentences
- `empty_with_whitespace.txt` - only whitespace characters
- `unicode_content.txt` - mixed scripts (Greek, Chinese, Arabic) with paragraph structure
- `large_document.md` - ~10000 words for chunking stress tests

#### 2.2.3: Create Folder Structure Test Data

```
documents/nested/
  .lsm_tags.json          -> {"tags": ["philosophy", "notes"]}
  subfolder_a/
    .lsm_tags.json        -> {"tags": ["epistemology"]}
    notes_a1.md
    notes_a2.txt
  subfolder_b/
    notes_b1.md
```

#### 2.2.4: Create Duplicate Detection Test Data

- `duplicate_content_1.txt` and `duplicate_content_2.txt` - identical content
- `near_duplicate.txt` - same content with minor word changes

#### 2.2.5: Create Test Configuration Files

**Files to create in `tests/fixtures/synthetic_data/configs/`:**
- `test_config_openai.json` - full config using OpenAI
- `test_config_local.json` - full config using Ollama
- `test_config_minimal.json` - minimal valid config

**Post-block:** `tests/test_fixtures/test_synthetic_data.py` verifying all fixture files exist/readable, run tests, update docs, commit.

### 2.3: Migrate Core Tests from Mocked to Real (COMPLETED)

Systematically replace mock-based tests with real implementations. Keep lightweight fakes only in smoke-tier tests.

#### 2.3.1: Migrate Parser Tests

Enhance `tests/test_ingest/test_parsers.py` to use the rich corpus. Verify text extraction quality, metadata, error handling, empty files.

#### 2.3.2: Migrate Chunking Tests

Update `tests/test_ingest/test_chunking.py` and `tests/test_ingest/test_structure_chunking.py` to chunk research paper (verify heading detection), large document (verify count/overlap), HTML after parsing, page number tracking.

#### 2.3.3: Create Real Embedding Tests

**File to create:** `tests/test_ingest/test_real_embeddings.py`

Marked `@pytest.mark.integration`:
- Load real `all-MiniLM-L6-v2`, embed actual text chunks
- Verify dimensions (384), semantically similar texts have lower cosine distance
- Verify batch embedding consistency

#### 2.3.4: Create Real ChromaDB Tests

**File to create:** `tests/test_vectordb/test_real_chromadb.py`

Marked `@pytest.mark.integration`:
- Create real collection in tmp_path, add real chunks with real embeddings
- Query, filter, update, delete, paginate, count
- Test with 100+ chunks

#### 2.3.5: Create Live LLM Provider Tests

**Files to create:**
- `tests/test_providers/test_live_openai.py`
- `tests/test_providers/test_live_anthropic.py`
- `tests/test_providers/test_live_gemini.py`
- `tests/test_providers/test_live_local.py`

Each marked `@pytest.mark.live_llm`, auto-skip if no key:
- Test `_send_message()`, `_send_streaming_message()`, `synthesize()`, `generate_tags()`, `rerank()`, `is_available()`, `health_check()`

#### 2.3.6: Create Live Remote Provider Tests

All 10 remote providers need live tests. **Files to create:**
- `tests/test_providers/remote/test_live_brave.py` (needs `LSM_TEST_BRAVE_API_KEY`)
- `tests/test_providers/remote/test_live_wikipedia.py` (no key needed)
- `tests/test_providers/remote/test_live_arxiv.py` (no key needed)
- `tests/test_providers/remote/test_live_semantic_scholar.py` (optional `LSM_TEST_SEMANTIC_SCHOLAR_API_KEY`)
- `tests/test_providers/remote/test_live_core.py` (needs `LSM_TEST_CORE_API_KEY`)
- `tests/test_providers/remote/test_live_oai_pmh.py` (no key needed)
- `tests/test_providers/remote/test_live_philpapers.py` (no key needed)
- `tests/test_providers/remote/test_live_ixtheo.py` (no key needed)
- `tests/test_providers/remote/test_live_openalex.py` (no key needed)
- `tests/test_providers/remote/test_live_crossref.py` (no key needed)

Each marked `@pytest.mark.live_remote`, auto-skip if required key is missing.

Each file tests:
- Search with a known query, verify result structure (title, url, snippet/abstract)
- Empty/nonsense query handling
- Rate limiting / timeout behavior
- Provider-specific features (e.g., arXiv categories, Crossref DOI lookup, Semantic Scholar fields)

**Post-block:** run all tiers, update docs, commit.

### 2.4: Full Integration Test Suite (COMPLETED)

**Files to create:**
- `tests/test_integration/test_full_pipeline.py`

#### 2.4.1: End-to-End Integration Test (no network)

Marked `@pytest.mark.integration`:
- Build real config, create document root with fixture files
- Run real ingest pipeline (parse -> chunk -> embed -> store to real ChromaDB)
- Verify manifest, run real query retrieval, verify candidates with correct metadata

#### 2.4.2: Full Live Pipeline Test

Marked `@pytest.mark.live`:
- Everything from 2.4.1 plus real LLM reranking, real LLM synthesis
- Verify synthesized answer has inline citations and is coherent

#### 2.4.3: Performance/Scale Test

Marked `@pytest.mark.performance`:
- Ingest 100+ chunks from large_document.md, measure query latency

**Post-block:** run tests at each tier, update docs, commit.

### 2.5: Clean Up Legacy Mock Tests (COMPLETED)

#### 2.5.1: Audit All Mock Usage

Survey every test file for `Mock`, `MagicMock`, `patch`, `mocker`. Categorize:
- **Replace** - covered by new real tests
- **Convert to lightweight fake** - keep for smoke speed but use simple fake classes (like existing `FakeSentenceTransformer`), not `unittest.mock`
- **Keep** - testing error paths not reproducible with real services (document why)

#### 2.5.2: Execute Migration

Replace/remove/convert per audit. For kept mocks, add comment explaining necessity.

#### 2.5.3: Clean Up Conftest

Remove `mock_openai_client`, `mock_embedder`, `mock_chroma_collection` fixtures from `conftest.py` after verifying no remaining dependents. Replace with real equivalents.

**Post-block:** run full suite, update docs, commit.

---

## Phase 3: Solid Agent System with Sandbox Security

Current state: functional framework (BaseAgent, AgentHarness, 9 tools, ToolSandbox with path canonicalization, ResearchAgent), but only ~13 tests, no security hardening beyond basic path checks, no environment scrubbing, no log redaction, no permission gates, no runner abstraction.

**Threat Profile:** The agent sandbox executes LLM-directed tool calls on behalf of the user. The primary threat actors are: (1) prompt injection via untrusted document content that manipulates agent behavior, (2) malicious or hallucinated tool arguments from the LLM attempting to escape sandbox boundaries, and (3) information leakage of API keys and secrets through agent logs or tool outputs. The STRIDE framework structures our security testing:

- **S**poofing: Tool identity verification, registry integrity
- **T**ampering: Path traversal, symlink escape, write path enforcement
- **R**epudiation: Agent action logging, artifact tracking
- **I**nformation Disclosure: Secret leakage via env vars, logs, tool output
- **D**enial of Service: Iteration caps, token budgets, output truncation
- **E**levation of Privilege: Permission gates, risk-level policies, sandbox monotonicity

**Security testing regime:** All security tests are adversarial — they actively attempt to break out of the sandbox using known attack patterns (path traversal, null byte injection, UNC paths, symlink escape, prompt injection, etc.). Tests must verify that defenses reject attacks rather than merely testing that normal operations succeed. New tools and agents must include security test coverage before merge. The security test suite runs as part of the default `pytest` invocation (no special markers required for path/permission/injection tests).

**Security documentation:** Phase 3 produces `docs/development/SECURITY.md` documenting the full threat model, STRIDE categories, attack surface inventory, testing methodology, and instructions for extending coverage when adding new tools or agents.

### 3.1: Tool Risk Metadata and Registry Enhancements

#### 3.1.1: Add Risk Metadata to BaseTool

Add class attributes to `BaseTool`:
```python
risk_level: str = "read_only"    # "read_only" | "writes_workspace" | "network" | "exec"
preferred_runner: str = "local"  # "local" | "docker"
needs_network: bool = False
```

Update `get_definition()` to include risk fields.

#### 3.1.2: Tag All Existing Tools

- `read_file`, `read_folder`, `query_embeddings` -> `risk_level="read_only"`
- `write_file`, `create_folder` -> `risk_level="writes_workspace"`, `needs_network=False`
- `load_url`, `query_llm`, `query_remote`, `query_remote_chain` -> `risk_level="network"`, `needs_network=True`

#### 3.1.3: Add Registry Query Methods

- `list_by_risk(risk_level) -> List[BaseTool]`
- `list_network_tools() -> List[BaseTool]`

**Files to modify:**
- `lsm/agents/tools/base.py`
- All tool files in `lsm/agents/tools/`

**Post-block:** `tests/test_agents/test_tool_risk_metadata.py`, run tests, update `docs/AGENTS.md`, commit.

### 3.2: Sandbox Security Hardening

#### 3.2.1: Harden Path Canonicalization

Add `_canonicalize_path(path: Path) -> Path` to `ToolSandbox`:
- `path.expanduser().resolve()`
- Detect symlinks pointing outside allowed paths via `path.is_symlink()`
- Reject null bytes in path strings
- On Windows: reject UNC paths (`\\\\server\\share`), alternate data streams (`:` in filename)
- Reject `..` appearing after path normalization

**Files to modify:**
- `lsm/agents/tools/sandbox.py`

#### 3.2.2: Add Environment Scrubbing

**File to create:** `lsm/agents/tools/env_scrubber.py`

`scrub_environment() -> Dict[str, str]`:
- Start with minimal env (PATH, HOME/USERPROFILE, TEMP/TMP, LANG)
- Exclude all `*_API_KEY`, `*_SECRET*`, `*_TOKEN*`, `*_PASSWORD*` patterns
- Return clean dict

#### 3.2.3: Add Log Redaction

**File to create:** `lsm/agents/log_redactor.py`

`redact_secrets(text: str) -> str`:
- Detect API key patterns (`sk-...`, `key_...`, base64 blobs)
- Replace with `[REDACTED]`
- Apply to all `AgentLogEntry.content` before persistence

**Files to modify:**
- `lsm/agents/log_formatter.py` - apply redaction before saving
- `lsm/agents/harness.py` - apply redaction to tool output before logging

#### 3.2.4: Add Interactive Permission Gates

**File to create:** `lsm/agents/permission_gate.py`

`PermissionGate` class with `check(tool, args) -> PermissionDecision`:

Precedence:
1. Per-tool explicit override in `sandbox.require_user_permission`
2. Per-risk-level policy in `sandbox.require_permission_by_risk`
3. Tool's `requires_permission` default
4. Allow (safe default for read-only tools)

`PermissionDecision` dataclass: `allowed`, `reason`, `requires_confirmation`, `tool_name`, `risk_level`

**Files to modify:**
- `lsm/config/models/agents.py` - add `require_permission_by_risk: Dict[str, bool]` to `SandboxConfig`
- `lsm/agents/tools/sandbox.py` - integrate `PermissionGate`

**Post-block:** tests, run suite, update `docs/AGENTS.md` and `example_config.json`, commit.

### 3.3: Comprehensive Security Test Suite

Tests covering all STRIDE threat categories. These tests actively attempt to break out of the sandbox.

#### 3.3.1: T1 - Arbitrary File Access (10 tests)

**File to create:** `tests/test_agents/test_security_paths.py`

- Relative path traversal: `read_file("../secrets.txt")` -> blocked
- Absolute paths outside allowlist: `/etc/passwd`, `C:\Windows\System32` -> blocked
- Symlink escape: symlink inside allowed dir pointing outside -> blocked
- Windows UNC path: `\\server\share\file` -> blocked
- Null byte injection: `"allowed/file\x00.txt"` -> blocked
- Deeply nested traversal: `a/b/c/../../../../outside` -> blocked
- Empty path -> rejected
- Write path traversal: `write_file("../../outside.txt")` -> blocked
- Case sensitivity edge cases on Windows
- `..` components remaining after resolution -> blocked

#### 3.3.2: T2+T3 - Command Execution & Privilege Escalation (8 tests)

**File to create:** `tests/test_agents/test_security_permissions.py`

- Unknown tool name rejected by registry
- Tool schema enforced (extra fields rejected)
- Permission gate blocks tools requiring permission
- Per-risk permission policy enforced
- Per-tool override takes precedence over risk policy
- Write tools blocked when write paths empty
- Network tools blocked when `allow_url_access=False`
- Local sandbox cannot exceed global sandbox

#### 3.3.3: T4 - Network Abuse (5 tests)

**File to create:** `tests/test_agents/test_security_network.py`

- `load_url` blocked when `allow_url_access=False`
- `query_remote` blocked when `allow_url_access=False`
- `query_remote_chain` blocked when `allow_url_access=False`
- Network tools allowed when `allow_url_access=True`
- Non-network tools unaffected by url_access setting

#### 3.3.4: T5 - Resource Exhaustion (5 tests)

**File to create:** `tests/test_agents/test_security_resources.py`

- Iteration cap enforced in harness
- Token budget exhaustion stops execution
- Budget tracking accuracy
- Large tool output handled (truncation)
- Graceful stop when budget reached mid-iteration

#### 3.3.5: T6 - Data Integrity (4 tests)

**File to create:** `tests/test_agents/test_security_integrity.py`

- Write tools only write to allowed paths
- Parent directory creation is safe
- Overwrite requires permission
- Artifacts tracked in agent state

#### 3.3.6: T7 - Prompt Injection (7 tests)

**File to create:** `tests/test_agents/test_security_injection.py`

- Non-JSON LLM response doesn't trigger tool execution
- Malformed JSON (missing action field) doesn't trigger execution
- Unknown tool name in action raises KeyError
- Embedded JSON in user content doesn't leak to action parsing
- Tool argument validation against schema
- "DONE" action terminates without tool execution
- Empty action string terminates loop

#### 3.3.7: T8 - Secret Leakage (8 tests)

**File to create:** `tests/test_agents/test_security_secrets.py`

- Env scrubber removes `*_API_KEY` variables
- Env scrubber removes `*_SECRET*`, `*_TOKEN*`, `*_PASSWORD*`
- Env scrubber preserves PATH, HOME, TEMP
- Log redactor masks API key patterns
- Log redactor masks base64 blobs
- Redacted logs contain no original secret values
- Tool output containing secrets is redacted before logging
- Agent state file contains no secrets

**Post-block:** run full suite, update `docs/development/TESTING.md`, commit.

### 3.3.8: Write Security Documentation

**File to create:** `docs/development/SECURITY.md`

Document the full agent sandbox security posture:

- **Threat model:** Describe the three primary threat actors (prompt injection, hallucinated arguments, secret leakage) and why LLM-directed execution requires defense-in-depth.
- **Attack surface inventory:** Enumerate every entry point where untrusted data reaches the sandbox (LLM tool call arguments, document content read by tools, URL fetches, user-supplied paths).
- **STRIDE coverage matrix:** Table mapping each STRIDE category to the specific defenses implemented and the test files that verify them (T1–T8 from section 3.3).
- **Testing methodology:** Explain the adversarial testing approach — tests attempt to break out, not just confirm happy paths. Include instructions for running the security suite (`pytest tests/test_agents/test_security_*.py -v`).
- **Extending coverage:** Step-by-step guide for adding security tests when creating new tools or agents, including which threat categories to consider and test templates to follow.
- **Permission gate reference:** Document the permission precedence chain (per-tool override > per-risk policy > tool default > allow) with examples.

**Post-block:** run tests, commit.

### 3.4: Runner Abstraction and Sandbox Config Extensions

#### 3.4.1: Create Runner Abstraction

**File to create:** `lsm/agents/tools/runner.py`

```python
@dataclass
class ToolExecutionResult:
    stdout: str
    stderr: str = ""
    runner_used: str = "local"
    runtime_ms: float = 0.0
    artifacts: List[str] = field(default_factory=list)

class BaseRunner(ABC):
    def run(self, tool, args, env) -> ToolExecutionResult: ...

class LocalRunner(BaseRunner):
    # Execute tool.execute(args) with timeout, output truncation, metric tracking
```

#### 3.4.2: Extend SandboxConfig

Add to `SandboxConfig`:
- `execution_mode: str = "local_only"` - `"local_only"` | `"prefer_docker"`
- `require_permission_by_risk: Dict[str, bool]`
- `limits: Dict[str, Any]` - `timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`
- `docker: Dict[str, Any]` - `enabled`, `image`, `network_default`, `cpu_limit`, `mem_limit_mb`, `read_only_root`

#### 3.4.3: Integrate Runner into Sandbox

Update `ToolSandbox.execute()`:
1. Run permission gate checks
2. Select runner based on `execution_mode` and tool `risk_level`
3. Scrub environment
4. Execute via selected runner
5. Apply log redaction to output
6. Return result

**Files to modify:**
- `lsm/agents/tools/sandbox.py`
- `lsm/agents/harness.py`
- `lsm/config/models/agents.py`

**Post-block:** `tests/test_agents/test_runner.py`, run tests, update docs and `example_config.json`, commit.

### 3.5: Docker Sandbox Foundation

Preliminary DockerRunner for future high-risk tools. No agents or tools require it in v0.5.0.

#### 3.5.1: Create DockerRunner

**File to create:** `lsm/agents/tools/docker_runner.py`

Uses `subprocess` to invoke `docker run`:
- Mounts workspace as RW, read paths as RO
- Sets `--network=none`, CPU/memory/pids limits
- Serializes tool+args as JSON via stdin
- Python entrypoint inside container executes tool
- Captures stdout, enforces timeout

#### 3.5.2: Create Sandbox Docker Image

**Files to create:**
- `Dockerfile.sandbox` - minimal Python 3.12-slim, non-root user, read-only root
- `lsm/agents/tools/_docker_entrypoint.py` - reads tool name + args from stdin, executes, prints result

#### 3.5.3: Add Runner Selection Policy

Update `ToolSandbox._select_runner()`:
- `read_only` and `writes_workspace` -> always LocalRunner
- `network` and `exec` -> DockerRunner if available, else require permission confirmation
- Log runner selection decisions

**Post-block:** `tests/test_agents/test_docker_runner.py` (marked `@pytest.mark.docker`), `tests/test_agents/test_runner_policy.py`, run tests, update docs, commit.

### 3.6: New Agent Tools (MVP Set)

Six tools needed by the planned Writing, Synthesis, and Curator agents.

#### 3.6.1: extract_snippets

**File to create:** `lsm/agents/tools/extract_snippets.py`

`risk_level="read_only"`. Input: `query`, `paths[]`, `max_snippets`, `max_chars_per_snippet`. Uses existing `lsm/query/retrieval.py` retrieval functions filtered to specific paths. Returns JSON array of `{source_path, snippet, score}`.

#### 3.6.2: file_metadata

**File to create:** `lsm/agents/tools/file_metadata.py`

`risk_level="read_only"`. Input: `paths[]`. Returns `[{path, size_bytes, mtime_iso, ext}]`.

#### 3.6.3: hash_file

**File to create:** `lsm/agents/tools/hash_file.py`

`risk_level="read_only"`. Input: `path`. Returns `{path, sha256}`.

#### 3.6.4: similarity_search

**File to create:** `lsm/agents/tools/similarity_search.py`

`risk_level="read_only"`. Input: `chunk_ids|paths`, `top_k`, `threshold`. Queries embeddings, returns pairs with similarity scores above threshold.

#### 3.6.5: source_map

**File to create:** `lsm/agents/tools/source_map.py`

`risk_level="read_only"`. Input: evidence list. Returns `{source_path: {count, top_snippets}}`.

#### 3.6.6: append_file

**File to create:** `lsm/agents/tools/append_file.py`

`risk_level="writes_workspace"`, `requires_permission=True`. Input: `path`, `content`. Appends to existing file.

#### 3.6.7: Register in Factory

Update `lsm/agents/tools/__init__.py` `create_default_tool_registry()`.

**Post-block:** `tests/test_agents/test_new_tools.py`, run tests, update `docs/AGENTS.md`, commit.

### 3.7: Agent Framework Flexibility

Make it easy to create new agents by extracting common patterns.

#### 3.7.1: Extract Common Helpers to BaseAgent

Move from `ResearchAgent` to `BaseAgent`:
- `_log(content, actor, ...)` - structured logging helper
- `_parse_json(value) -> Any` - safe JSON parsing
- `_consume_tokens(text)` - budget tracking
- `_budget_exhausted() -> bool` - budget check

**Files to modify:**
- `lsm/agents/base.py`
- `lsm/agents/research.py` - delegate to base

#### 3.7.2: Add Tool Allowlist to Harness

Allow agents to declare which tools they use:
```python
class AgentHarness:
    def __init__(self, ..., tool_allowlist: Optional[Set[str]] = None):
```

Filters tool definitions sent to LLM and refuses execution of unlisted tools.

#### 3.7.3: Automatic Per-Run Workspace

Harness creates `<agents_folder>/<agent_name>_<timestamp>/workspace/` and passes path in agent context.

**Files to modify:**
- `lsm/agents/base.py`
- `lsm/agents/harness.py`

**Post-block:** `tests/test_agents/test_agent_helpers.py`, `tests/test_agents/test_harness_allowlist.py`, run tests, update docs, commit.

### 3.8: Writing Agent

**File to create:** `lsm/agents/writing.py`

```python
class WritingAgent(BaseAgent):
    name = "writing"
    description = "Generate grounded written deliverables from the knowledge base."
```

Workflow: parse input -> grounding retrieval via `query_embeddings` -> build outline from evidence -> draft prose -> self-review -> persist `deliverable.md`.

Tool allowlist: `query_embeddings`, `read_file`, `read_folder`, `write_file`, `extract_snippets`, `source_map`.

Register in `lsm/agents/factory.py`.

**Post-block:** `tests/test_agents/test_writing_agent.py`, run tests, update docs and `example_config.json`, commit.

### 3.9: Synthesis Agent

**File to create:** `lsm/agents/synthesis.py`

```python
class SynthesisAgent(BaseAgent):
    name = "synthesis"
    description = "Distill multiple documents into compact summaries."
```

Workflow: scope selection -> candidate sources via `read_folder` -> evidence retrieval -> synthesize to target format (bullets/outline/narrative/QA) -> tighten to length -> coverage check -> persist `synthesis.md` and `source_map.md`.

Tool allowlist: `read_folder`, `query_embeddings`, `read_file`, `write_file`, `extract_snippets`, `source_map`.

Register in factory.

**Post-block:** `tests/test_agents/test_synthesis_agent.py`, run tests, update docs, commit.

### 3.10: Curator Agent

**File to create:** `lsm/agents/curator.py`

```python
class CuratorAgent(BaseAgent):
    name = "curator"
    description = "Maintain corpus quality with actionable reports."
```

Workflow: inventory via `read_folder` -> metadata via `file_metadata` -> exact duplicates via `hash_file` -> near-duplicates via `similarity_search` -> staleness/quality heuristics -> generate recommendations -> persist `curation_report.md`.

Tool allowlist: `read_folder`, `file_metadata`, `hash_file`, `query_embeddings`, `similarity_search`, `write_file`. Read-only by default.

Register in factory.

**Post-block:** `tests/test_agents/test_curator_agent.py`, run tests, update docs, commit.

---

## Phase 4: Memory System

Persistent agent posture and preferences. Memories affect *how* an agent works, not *what it knows*. Lifecycle: Candidate -> Promoted -> Expired. Memories are not automatically injected; they are retrieved selectively.

### 4.1: Memory Storage Backend

#### 4.1.1: Create Memory Data Model

**File to create:** `lsm/agents/memory/models.py`

```python
@dataclass
class Memory:
    id: str
    type: str          # "pinned" | "project_fact" | "task_state" | "cache"
    key: str
    value: Any         # JSON-serializable
    scope: str         # "global" | "agent" | "project"
    tags: List[str]
    confidence: float  # 0.0 - 1.0
    created_at: datetime
    last_used_at: datetime
    expires_at: Optional[datetime]
    source_run_id: str

@dataclass
class MemoryCandidate:
    id: str
    memory: Memory
    provenance: str    # Which agent/run proposed it
    rationale: str     # Why it was proposed
    status: str        # "pending" | "promoted" | "rejected"
```

#### 4.1.2: Create Storage Abstraction

**Files to create:**
- `lsm/agents/memory/store.py` - `BaseMemoryStore` ABC with SQLite and PostgreSQL implementations
- `lsm/agents/memory/__init__.py`

Use PostgreSQL when VectorDB is PostgreSQL, SQLite when VectorDB is ChromaDB. Create `memories` and `memory_candidates` tables.

Operations:
- `put_candidate(memory, provenance, rationale) -> str` (returns candidate ID)
- `promote(candidate_id) -> Memory`
- `reject(candidate_id)`
- `expire() -> int` (TTL cleanup, returns count expired)
- `get(memory_id) -> Memory`
- `delete(memory_id)`
- `search(scope, tags, type, limit, token_budget) -> List[Memory]`

#### 4.1.3: Enforce TTL Caps Per Memory Type

Default TTLs:
- `pinned`: no expiry
- `project_fact`: 90 days
- `task_state`: 7 days
- `cache`: 24 hours

Configurable via `agents.memory` config section.

#### 4.1.4: Create Migration Tool

**File to create:** `lsm/agents/memory/migrations.py`

Bidirectional migration between SQLite and PostgreSQL memory stores.

**Files to modify:**
- `lsm/config/models/agents.py` - add `MemoryConfig` dataclass with TTL settings, storage backend

**Post-block:** `tests/test_agents/test_memory_store.py` (CRUD, TTL, search, promotion), run tests, update docs, commit.

### 4.2: Memory API and Harness Integration

#### 4.2.1: Core Memory Operations

**File to create:** `lsm/agents/memory/api.py`

Functions:
- `memory_put_candidate(store, memory, provenance, rationale)`
- `memory_promote(store, candidate_id)` - user or curator approved
- `memory_expire(store)` - cleanup expired memories
- `memory_search(store, scope, tags, type, limit, token_budget)` - with recency scoring and pin weighting

Update `last_used_at` only for memories actually injected into agent context.

#### 4.2.2: Memory Context Builder

**File to create:** `lsm/agents/memory/context_builder.py`

`MemoryContextBuilder` class:
- Invoked before LLM call in harness
- Retrieves relevant memories for the current agent/scope
- Builds structured "Standing Context" block
- Injects memory block separately from tool descriptions
- Zero-memory injection is valid (no error)

#### 4.2.3: Integrate into AgentHarness

**Files to modify:**
- `lsm/agents/harness.py` - invoke `MemoryContextBuilder` before each LLM call, append standing context to system prompt

**Post-block:** `tests/test_agents/test_memory_api.py`, `tests/test_agents/test_memory_context_builder.py`, run tests, update docs, commit.

### 4.3: Memory Agent Tools

#### 4.3.1: Create memory_put Tool

**File to create:** `lsm/agents/tools/memory_put.py`

`risk_level="writes_workspace"`, `requires_permission=True`. Agents can propose memory candidates during execution. Input: `key`, `value`, `type`, `scope`, `tags`, `rationale`.

#### 4.3.2: Create memory_search Tool

**File to create:** `lsm/agents/tools/memory_search.py`

`risk_level="read_only"`. Agents can query existing memories. Input: `scope`, `tags`, `type`, `limit`.

Register both in `create_default_tool_registry()`.

**Post-block:** `tests/test_agents/test_memory_tools.py`, run tests, update docs, commit.

### 4.4: Curator Memory Distillation

#### 4.4.1: Add Run Summaries to Harness

**Files to modify:**
- `lsm/agents/harness.py` - emit `run_summary.json` at end of each run

Summary includes: agent name, topic, tools used, approvals/denials, artifacts created, run outcome, duration, token usage.

#### 4.4.2: Add Memory Mode to Curator Agent

**Files to modify:**
- `lsm/agents/curator.py` - add `--mode memory` support

In memory mode, Curator:
- Scans recent run summaries
- Extracts candidate memories from repeated tool patterns, repeated constraints, user approvals/denials
- Emits `memory_candidates.md` and `memory_candidates.json`

**Post-block:** `tests/test_agents/test_curator_memory_mode.py`, run tests, update docs, commit.

### 4.5: Memory UI/CLI

#### 4.5.1: CLI Commands

**Files to modify:**
- `lsm/ui/shell/commands/agents.py` - add memory subcommands

Commands:
- `/memory list [--scope global|agent|project] [--type pinned|project_fact|...]`
- `/memory promote <candidate_id>`
- `/memory delete <memory_id>`
- `/memory candidates` - list pending candidates

#### 4.5.2: TUI Memory View

**Files to modify:**
- `lsm/ui/tui/screens/agents.py` - add memory candidates panel

View: list of memory candidates with approve/reject/edit TTL actions.

**Post-block:** `tests/test_ui/shell/test_memory_commands.py`, `tests/test_ui/tui/test_memory_screen.py`, run tests, update docs and `example_config.json`, commit.

---

## Phase 5: Agent Scheduler

Time-based and cyclic agent execution. Preserves local-first philosophy. Safety-first for unattended runs.

### 5.1: Scheduler Configuration

#### 5.1.1: Define ScheduleConfig

**Files to modify:**
- `lsm/config/models/agents.py`

```python
@dataclass
class ScheduleConfig:
    agent_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    interval: str = "daily"        # "hourly" | "daily" | "weekly" | "3600s" | cron syntax
    enabled: bool = True
    concurrency_policy: str = "skip"   # "skip" | "queue" | "cancel"
    confirmation_mode: str = "auto"    # "auto" | "confirm" | "deny"
```

Add `schedules: List[ScheduleConfig]` to `AgentConfig`.

**Post-block:** `tests/test_config/test_schedule_config.py`, run tests, update docs, commit.

### 5.2: Scheduler Engine

#### 5.2.1: Implement AgentScheduler Service

**File to create:** `lsm/agents/scheduler.py`

`AgentScheduler` class:
- Persistent state in `<agents_folder>/schedules.json`
- Threaded tick loop (checks every 60s)
- For each due schedule: check concurrency policy, create agent, run via harness
- Track `last_run_at`, `next_run_at`, `last_status`, `last_error`
- Skip/queue/cancel logic for overlapping runs
- Graceful shutdown on stop signal

#### 5.2.2: Execution Safety Defaults

Scheduled agents default to:
- Read-only tools only (no `write_file`, `create_folder`, `load_url`)
- No network access (`allow_url_access=False`)
- Require explicit opt-in for writes or network via schedule `params`
- Use DockerRunner for any tools with `risk_level="network"` or `"exec"` (when Docker available)

**Post-block:** `tests/test_agents/test_scheduler.py` (schedule persistence, no overlaps, safe defaults, tick logic), run tests, update docs, commit.

### 5.3: Scheduler CLI/TUI

#### 5.3.1: CLI Commands

**Files to modify:**
- `lsm/ui/shell/commands/agents.py`

Commands:
- `/agent schedule add <agent_name> <interval> [--params '{"topic": "..."}']`
- `/agent schedule list`
- `/agent schedule enable|disable <schedule_id>`
- `/agent schedule remove <schedule_id>`
- `/agent schedule status` - show last run info for all schedules

#### 5.3.2: TUI Scheduler View

**Files to modify:**
- `lsm/ui/tui/screens/agents.py` - add scheduled agents panel

View: table of schedules with name, interval, last run, next run, status. Enable/disable toggle.

**Post-block:** `tests/test_ui/shell/test_schedule_commands.py`, `tests/test_ui/tui/test_schedule_screen.py`, run tests, update docs and `example_config.json`, commit.

---

## Phase 6: Meta-Agent (Orchestrator)

Coordinate multiple agents toward a single goal. Deterministic orchestration over freeform dialogue. Sub-agents run with equal or tighter permissions (monotone sandbox). Shared artifacts, not shared context.

### 6.1: Meta-Agent Core

#### 6.1.1: Implement MetaAgent Subclass

**File to create:** `lsm/agents/meta.py`

```python
class MetaAgent(BaseAgent):
    name = "meta"
    description = "Orchestrate multiple agents toward a single goal."
```

MetaAgent converts user goal into a task graph:
- Each task specifies: agent name, parameters, expected artifacts, dependencies
- Sequential execution (v1) following dependency order
- MetaAgent plans but does not execute domain work itself

#### 6.1.2: Task Graph Model

**File to create:** `lsm/agents/task_graph.py`

```python
@dataclass
class AgentTask:
    id: str
    agent_name: str
    params: Dict[str, Any]
    expected_artifacts: List[str]
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"

@dataclass
class TaskGraph:
    goal: str
    tasks: List[AgentTask]
    # Methods: topological_sort(), next_ready(), mark_complete(), is_done()
```

Register MetaAgent in `lsm/agents/factory.py`.

**Post-block:** `tests/test_agents/test_meta_agent.py`, `tests/test_agents/test_task_graph.py`, run tests, update docs, commit.

### 6.2: Meta-Agent System Tools

#### 6.2.1: Create Orchestration Tools

**Files to create:**
- `lsm/agents/tools/spawn_agent.py` - `spawn_agent(name, params)` starts sub-agent
- `lsm/agents/tools/await_agent.py` - `await_agent(agent_id)` blocks until completion
- `lsm/agents/tools/collect_artifacts.py` - `collect_artifacts(agent_id, pattern)` gathers outputs

All `risk_level="exec"`, `requires_permission=True`.

#### 6.2.2: Enforce Sandbox Downgrades

Sub-agent sandbox must be a subset of meta-agent sandbox. Implement in `AgentHarness`:
- When spawning sub-agent, derive sandbox from parent
- Validate monotone restriction (sub <= parent)
- Per sub-agent controlled write areas within shared workspace

**Files to modify:**
- `lsm/agents/harness.py`
- `lsm/agents/tools/sandbox.py`

**Post-block:** `tests/test_agents/test_meta_tools.py`, `tests/test_agents/test_sandbox_monotone.py`, run tests, update docs, commit.

### 6.3: Shared Workspace and Synthesis

#### 6.3.1: Per-Meta-Run Workspace

Create workspace structure:
```
<agents_folder>/meta_<timestamp>/
  workspace/           # shared artifacts
  sub_agents/
    research_001/      # sub-agent workspace
    writing_002/
  final_result.md
  meta_log.md
```

Sub-agents get read access to shared `workspace/` and write access to their own `sub_agents/<name>/` directory.

#### 6.3.2: Final Synthesis

MetaAgent collects sub-agent outputs, optionally runs final synthesis via LLM, persists `final_result.md` and `meta_log.md` with full execution trace.

**Post-block:** `tests/test_agents/test_meta_workspace.py`, `tests/test_agents/test_meta_synthesis.py`, run tests, update docs, commit.

### 6.4: Meta-Agent UI/CLI

#### 6.4.1: CLI Commands

**Files to modify:**
- `lsm/ui/shell/commands/agents.py`

Commands:
- `/agent meta start <goal>` - start meta-agent with goal description
- `/agent meta status` - show task graph progress
- `/agent meta log` - show meta-agent execution log

#### 6.4.2: TUI Meta-Agent View

**Files to modify:**
- `lsm/ui/tui/screens/agents.py`

View: meta-agent run visualization with sub-agent status tree, task graph progress, artifact list.

**Post-block:** `tests/test_ui/shell/test_meta_commands.py`, `tests/test_ui/tui/test_meta_screen.py`, run tests, update docs and `example_config.json`, commit.

---

## Phase 7: Documentation and Changelog

### 7.1: Final Documentation and Version Updates

**Version bump (0.4.0 -> 0.5.0):**
- `pyproject.toml` - `version = "0.5.0"`
- `lsm/__init__.py` - `__version__ = "0.5.0"`

**Files to modify:**
- `docs/README.md` - Update with v0.5.0 features
- `docs/AGENTS.md` - Comprehensive update: new agents, tools, sandbox, memory, scheduler, meta-agent
- `docs/architecture/PROVIDERS.md` - Updated provider architecture
- `docs/development/TESTING.md` - Complete testing guide with tiers
- `docs/development/CHANGELOG.md` - v0.5.0 changelog entry

#### 7.1.1: Write CHANGELOG Entry

Summarize all changes across all 7 phases:
- LLM Provider Refactoring (~1500 lines removed)
- Test Framework Overhaul (tiered execution, real data, live tests for all 10 remote providers)
- Agent Sandbox Hardening (STRIDE coverage, 47+ security tests)
- New Tools (8: extract_snippets, file_metadata, hash_file, similarity_search, source_map, append_file, memory_put, memory_search)
- New Agents (3: Writing, Synthesis, Curator)
- Memory System (storage, API, context builder, curator distillation, UI/CLI)
- Agent Scheduler (config, engine, safety defaults, UI/CLI)
- Meta-Agent Orchestrator (task graphs, system tools, shared workspace, synthesis)
- Runner abstraction and Docker foundation

#### 7.1.2: Update CLAUDE.md

Update Architecture section, Key Files table, Important Notes, Commands section with all v0.5.0 changes.

**Post-block:** run `pytest tests/ -v` final validation, commit.

---

## Verification Plan

1. **After Phase 1**: Run full test suite. All existing tests pass. Provider factory creates all 5 providers. `synthesize()`, `rerank()`, `generate_tags()`, `stream_synthesize()` work identically to before refactoring.
   - Update `docs/development/CHANGELOG.md` with changes in Phase 1

2. **After Phase 2**: Run each tier independently:
   - `pytest tests/ -v` (smoke + integration) - all pass, fast (<60s)
   - `pytest tests/ -v -m integration` - real embeddings and ChromaDB
   - `pytest tests/ -v -m live_llm` (with API keys) - real LLM calls succeed
   - `pytest tests/ -v -m live_remote` - all 10 remote providers tested
   - `pytest tests/ -v -m live` - full live pipeline end-to-end
   - Update `docs/development/CHANGELOG.md` with changes in Phase 2

3. **After Phase 3**: Run security tests explicitly:
   - `pytest tests/test_agents/test_security_*.py -v` - all 47+ security tests pass
   - Verify no path escapes succeed
   - Verify secret redaction in logs
   - Verify permission gates block unauthorized tools
   - Test each new agent with `@pytest.mark.integration` tests
   - Update `docs/development/CHANGELOG.md` with changes in Phase 3

4. **After Phase 4**: Memory system verification:
   - Memory CRUD operations (SQLite and PostgreSQL backends)
   - TTL enforcement and expiry
   - MemoryContextBuilder injects standing context
   - Curator memory distillation produces valid candidates
   - CLI/TUI memory commands functional
   - Update `docs/development/CHANGELOG.md` with changes in Phase 4

5. **After Phase 5**: Scheduler verification:
   - Schedule persistence across restarts
   - No overlapping runs (concurrency policy)
   - Safe defaults enforced (read-only, no network)
   - Tick loop timing accuracy
   - Update `docs/development/CHANGELOG.md` with changes in Phase 5

6. **After Phase 6**: Meta-agent verification:
   - Task graph topological sort and execution order
   - Sub-agent sandbox monotonicity (sub <= parent permissions)
   - Artifact handoff between sub-agents
   - Failure isolation (one sub-agent fails, others unaffected)
   - Final synthesis collects all outputs
   - Update `docs/development/CHANGELOG.md` with changes in Phase 6

7. **Final**: Full suite green at smoke+integration tier. Live tier passes with configured credentials.
