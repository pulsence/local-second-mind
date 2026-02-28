# Phase 13: Agent Pipeline Tools

**Status**: Pending

Replaces the agent's `query_knowledge_base` tool with three pipeline-backed tools:
`query_context`, `execute_context`, and `query_and_synthesize`. Agents gain fine-grained
control over the retrieval pipeline.

Reference: [RESEARCH_PLAN.md §5.12](../RESEARCH_PLAN.md#512-agent-integration-unified-tool-surface)

---

## 13.1: Pipeline Tools Implementation

**Description**: Create the three new agent tools and inject the pipeline into the tool
registry.

**Tasks**:
- Create `lsm/agents/tools/query_context.py`:
  - `QueryContextTool.__init__(pipeline: RetrievalPipeline)`
  - `execute(query, mode, filters, k, starting_prompt=None, conversation_id=None,
    prior_response_id=None)` → serialized `ContextPackage`
  - Calls `pipeline.build_sources(QueryRequest(...))`
  - Mode resolution: string → `BUILT_IN_MODES` lookup, or accept dict for custom
- Create `lsm/agents/tools/execute_context.py`:
  - `ExecuteContextTool.__init__(pipeline: RetrievalPipeline)`
  - `execute(question, context_package, synthesis_style)` → serialized `QueryResponse`
  - Calls `pipeline.synthesize_context()` then `pipeline.execute()`
- Create `lsm/agents/tools/query_and_synthesize.py`:
  - `QueryAndSynthesizeTool.__init__(pipeline: RetrievalPipeline)`
  - `execute(query, mode, starting_prompt=None, conversation_id=None,
    prior_response_id=None)` → serialized `QueryResponse`
  - Calls `pipeline.run(QueryRequest(...))`
- Update tool registry (`lsm/agents/tools/__init__.py` or registry module):
  - `create_default_tool_registry(config, collection, embedder, pipeline, memory_store)`:
    - Register pipeline tools only when `pipeline` is provided
  - Remove `query_knowledge_base` registration
- Update `query_llm` tool to call `provider.send_message()` directly and support direct
  cache chaining args (`previous_response_id`, `prompt_cache_key`,
  `prompt_cache_retention`) with `response_id` in tool output
- Retain `query_<provider>`, `query_remote_chain`, `extract_snippets`, `similarity_search`
  unchanged

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/agents/tools/query_context.py` — new tool
- `lsm/agents/tools/execute_context.py` — new tool
- `lsm/agents/tools/query_and_synthesize.py` — new tool
- `lsm/agents/tools/query_llm.py` — cache-aware direct LLM tool updates
- `lsm/agents/tools/__init__.py` — registry update
- `lsm/agents/tools/query_knowledge_base.py` — remove (or mark deprecated)
- `tests/test_agents/tools/test_query_context.py`:
  - Test: tool returns serialized ContextPackage
  - Test: mode resolution works (string and dict)
  - Test: conversation/prompt fields round-trip into ContextPackage
- `tests/test_agents/tools/test_execute_context.py`:
  - Test: tool returns serialized QueryResponse
- `tests/test_agents/tools/test_query_and_synthesize.py`:
  - Test: tool returns complete QueryResponse
  - Test: response_id/conversation_id are returned for chaining
- `tests/test_agents/tools/test_query_llm.py`:
  - Test: cache parameters are forwarded to provider.send_message()
  - Test: tool output includes `response_id` when provider returns one
- `tests/test_agents/tools/test_tool_registry.py`:
  - Test: pipeline tools registered when pipeline provided
  - Test: pipeline tools not registered when pipeline is None

**Success criteria**: Three new tools replace `query_knowledge_base`. Agents can inspect
`ContextPackage` between retrieval and synthesis. Tool registry conditionally registers
pipeline tools. Agent query tools support multi-turn prompt/conversation cache chaining.

---

## 13.2: Agent Mode Validation and run_bounded Context Tracking

**Description**: Implement `AgentHarness` validation for agent-composed `ModeConfig` and
make `run_bounded()` track both `context_label` and conversation chain state
(`conversation_id`, `prior_response_id` / `response_id`) per context.

**Tasks**:
- Update `lsm/agents/harness.py`:
  - Before accepting a `ModeConfig` from an agent via tool call, validate:
    - `retrieval_profile` is in the allowed list
    - `remote_policy.enabled` requires `allow_url_access=true` in sandbox config
  - Reject invalid `ModeConfig` with a clear error message in tool response
  - Add per-context runtime tracking keyed by `context_label`:
    - Isolated message history per label (existing behavior)
    - Isolated conversation chain metadata per label (`conversation_id`,
      `prior_response_id` / last `response_id`)
  - Ensure `continue_context=False` resets both history and conversation chain for that
    label
  - Ensure `continue_context=True` with same `context_label` resumes both history and
    conversation chain for that label
  - Ensure different `context_label` values never share conversation-chain state
  - Thread context-scoped conversation parameters through LLM calls in `run_bounded()`
    (`previous_response_id`, deterministic `prompt_cache_key`) and persist returned
    `response_id` for the next turn in the same label
- Update agent runtime models if needed (`lsm/agents/models.py`) to store context-scoped
  conversation metadata in a typed structure
- Extend run summary output to include context-level tracking metadata:
  - context labels seen
  - per-label iteration counts
  - per-label latest `conversation_id` / `response_id` (redacted/safe form)
- Update agent security tests
- Add harness tests for context and conversation tracking:
  - same `context_label` carries forward prior `response_id`
  - `continue_context=False` resets conversation chain
  - separate labels remain isolated
  - `context_label=None` remains isolated from named labels

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/agents/harness.py` — ModeConfig validation
- `lsm/agents/models.py` — context conversation metadata model (if required)
- `tests/test_agents/test_harness.py` — validation + context/conversation tracking tests
- `tests/test_agents/security/` — security tests for mode injection
- `tests/test_agents/test_base.py` — `_run_phase` forwarding tests for context/conversation state

**Success criteria**: Agents cannot enable remote sources without URL access permission.
Agents cannot use invalid retrieval profiles. `run_bounded()` preserves both message
history and conversation chain per `context_label`, with deterministic reset/isolation
behavior.

---

## 13.3: Phase 13 Code Review and Changelog

**Tasks**:
- Review tool serialization format for ContextPackage (agents receive JSON-serializable data)
- Review security: agent-composed modes cannot escalate sandbox permissions
- Validate v0.7.1 Agent Invariants (§7.7): all new pipeline tools go through
  `_run_phase()`, tool access is controlled via `tool_allowlist`, budget enforcement
  via `run_bounded()` is respected, workspace isolation is maintained
- Verify existing agents that used `query_knowledge_base` are updated to
  `query_and_synthesize`
- Verify agent conversation caching behavior:
  - `query_context` / `query_and_synthesize` round-trip conversation IDs correctly
  - `query_llm` exposes `response_id` for follow-up chaining
- Verify `run_bounded()` context tracking behavior:
  - `context_label` keeps independent histories and conversation chains
  - no cross-label leakage of `conversation_id` / `prior_response_id`
  - reset semantics (`continue_context=False`) clear both history and chain state
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/AGENTS.md` — document new pipeline tools
- Update `.agents/docs/architecture/development/AGENTS.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/AGENTS.md`
- `.agents/docs/architecture/development/AGENTS.md`

**Success criteria**: `pytest tests/ -v` passes. All agents updated to use new tools.
Changelog and docs updated.

---
