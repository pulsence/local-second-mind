# v0.7.1 Implementation Plan: Agent Foundation Refinement

**Version**: 0.7.1
**Research Plan**: [RESEARCH_PLAN.md](./RESEARCH_PLAN.md)

---

## Phases

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Interaction Channel — Two-Phase Timeout | Complete |
| 2 | AgentHarness and BaseAgent Infrastructure | Complete |
| 3 | query_knowledge_base Tool | Complete |
| 4 | Simple Agent Migration | Complete |
| 5 | Academic Agents Migration | Complete |
| 6 | Remaining Agent Migration | Complete |
| 7 | query_remote Tool Redesign | Complete |
| 8 | Final Code Review and Release | Complete |

---

# Phase 1: Interaction Channel — Two-Phase Timeout

**Status**: Complete

Adds an acknowledgment-based two-phase timeout to the agent interaction channel. An interaction
that has been displayed to the user in the TUI is acknowledged automatically; once acknowledged,
the channel waits indefinitely. This is independent of all other phases and can be developed
first.

Reference: [RESEARCH_PLAN.md §5](../docs/RESEARCH_PLAN.md#5-interaction-channel-two-phase-timeout)

---

## 1.1: InteractionConfig — acknowledged_timeout_seconds

**Description**: Add `acknowledged_timeout_seconds` to the `InteractionConfig` dataclass.
A value of `0` means infinite (no timeout once acknowledged). This is the only config change
in Phase 1.

**Tasks**:
- Add `acknowledged_timeout_seconds: int = 0` to `InteractionConfig`
- Ensure the new field is loaded from config via the existing config loading path
- Add `acknowledged_timeout_seconds` to `example_config.json` under `agents.interaction`
- Run relevant tests: `pytest tests/config/ -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/config/models/agents.py` (or wherever `InteractionConfig` is defined) — add field
- `example_config.json` — add `acknowledged_timeout_seconds: 0` to `agents.interaction`
- `tests/config/test_interaction_config.py` — test new field loads and defaults correctly

**Success criteria**: `InteractionConfig` loads `acknowledged_timeout_seconds` from config;
defaults to `0`; existing tests pass unchanged.

---

## 1.2: InteractionChannel — Two-Phase State Machine

**Description**: Modify `InteractionChannel.post_request()` to use a polling loop with two
timeout phases. Add `acknowledge_request()` and thread-safe acknowledged state.

**Tasks**:
- Add `_acknowledged: bool = False` and `_acknowledged_at: Optional[datetime] = None` as
  instance attributes, protected by the existing `_lock`
- Add `acknowledge_request(request_id: str) -> None`:
  - Under `_lock`, verify `request_id` matches the pending request's id
  - If it matches, set `_acknowledged = True`, `_acknowledged_at = datetime.utcnow()`
  - If it does not match, return without changing state
- Rewrite `post_request()` to use a polling loop:
  - Poll in small chunks (e.g., 0.5 s) using `event.wait(timeout=chunk)`
  - If event is set: response received — return it
  - If not yet acknowledged and elapsed > `timeout_seconds`: apply `timeout_action` and return
  - If acknowledged and `acknowledged_timeout_seconds == 0`: keep polling indefinitely
  - If acknowledged and `acknowledged_timeout_seconds > 0`: check elapsed-since-acknowledge
    and apply `timeout_action` if exceeded
- Reset `_acknowledged` and `_acknowledged_at` at the start of each new `post_request()` call
- All reads/writes to `_acknowledged` must be under `_lock`

**Files**:
- `lsm/agents/interaction.py` — `InteractionChannel` class
- `tests/agents/test_interaction.py` — new or extended test module:
  - Test: unacknowledged request times out after `timeout_seconds`
  - Test: acknowledged request does not time out (`acknowledged_timeout_seconds=0`)
  - Test: `acknowledge_request()` with wrong `request_id` does not acknowledge
  - Test: concurrent acknowledge and timeout polling — no race or deadlock
  - Test: `_acknowledged` resets between successive `post_request()` calls

**Success criteria**: An interaction posted and then immediately acknowledged never times out
when `acknowledged_timeout_seconds=0`. An unacknowledged interaction still times out after
`timeout_seconds`. Tests cover both phases and thread-safety.

---

## 1.3: AgentRuntimeManager and TUI Acknowledgment Signal

**Description**: Wire the acknowledgment signal from the TUI into `InteractionChannel`.
`AgentRuntimeManager` gains a forwarding method; the TUI calls it once per unique request.

**Tasks**:
- Add `acknowledge_interaction(agent_id: str, request_id: str) -> None` to `AgentRuntimeManager`:
  - Look up the agent run by `agent_id`
  - Call `run.interaction_channel.acknowledge_request(request_id)`
  - If the agent or channel is not found, return silently (defensive)
- In `_refresh_interaction_panel()` in `agents.py` (TUI screen):
  - Add `_acknowledged_interaction_ids: set[str]` as an instance attribute (init to `set()`)
  - When a new pending interaction is detected and rendered, check if `request_id` is in the set
  - If not: call `manager.acknowledge_interaction(agent_id, request_id)` and add `request_id`
    to the set
  - If yes: skip (do not acknowledge again)
- Add acknowledgment to shell path: when `/agent interact` renders an interaction prompt, call
  `manager.acknowledge_interaction(agent_id, request_id)` before blocking for user input

**Files**:
- `lsm/ui/shell/commands/agents.py` — `AgentRuntimeManager.acknowledge_interaction()`
- `lsm/ui/tui/screens/agents.py` — `_refresh_interaction_panel()`, `_acknowledged_interaction_ids`
- `tests/ui/test_agent_runtime_manager.py` — test `acknowledge_interaction()` forwards correctly
- `tests/ui/tui/test_agents_screen.py` — test that acknowledgment is sent once per unique
  `request_id` regardless of how many timer ticks fire

**Success criteria**: The TUI sends exactly one acknowledgment signal per interaction request
regardless of refresh frequency. The shell path sends acknowledgment before blocking on user
input. `AgentRuntimeManager.acknowledge_interaction()` correctly forwards to the channel.

---

## 1.4: Phase 1 Code Review and Changelog

**Tasks**:
- Review `InteractionChannel` changes: verify polling loop handles all timeout branch combinations
  correctly; verify no regression in the `timeout_action` (deny/approve) logic
- Review TUI changes: verify `_acknowledged_interaction_ids` does not grow unboundedly
  (clear stale ids when interactions are resolved)
- Review tests: confirm no mocks or stubs; confirm thread-safety tests are genuine concurrent tests
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 1 changes
- Update `.agents/docs/architecture/development/AGENTS.md`: document two-phase timeout in the
  interaction section

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/AGENTS.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog entry written. AGENTS.md interaction
section updated.

---

# Phase 2: AgentHarness and BaseAgent Infrastructure

**Status**: Complete

Adds the foundational infrastructure that all subsequent phases depend on:
`PhaseResult`, `AgentHarness.run_bounded()` with multi-context support,
`BaseAgent._run_phase()` with `context_label`, and `BaseAgent` workspace accessor methods.
No agents are migrated in this phase — only infrastructure is built and tested.

Reference: [RESEARCH_PLAN.md §2](../docs/RESEARCH_PLAN.md#2-core-architecture-persistent-harness-and-phase-execution)

---

## 2.1: PhaseResult Dataclass

**Description**: Define `PhaseResult` — the only return type from `_run_phase()`. It carries
operational output from one bounded execution phase but no financial or resource data.

**Tasks**:
- Define `PhaseResult` as a dataclass in `lsm/agents/base.py` (or a new `lsm/agents/phase.py`
  if preferred for import clarity):
  ```python
  @dataclass
  class PhaseResult:
      final_text: str          # last LLM response text for this phase
      tool_calls: list[dict]   # all tool calls made during this phase
      stop_reason: str         # "done" | "max_iterations" | "budget_exhausted" | "stop_requested"
  ```
- `PhaseResult` must NOT have `tokens_used`, `cost_usd`, or any accounting attributes
- Add `__slots__` or use `frozen=True` to prevent dynamic attribute injection

**Files**:
- `lsm/agents/base.py` (or `lsm/agents/phase.py`) — `PhaseResult` dataclass
- `tests/agents/test_phase_result.py`:
  - Test: `PhaseResult` can be constructed with the three fields
  - Test: `PhaseResult` has no `tokens_used` attribute (assert `AttributeError`)
  - Test: `PhaseResult` has no `cost_usd` attribute (assert `AttributeError`)
  - Test: all valid `stop_reason` values are accepted

**Success criteria**: `PhaseResult` exists and is importable from `lsm.agents`. It carries no
financial data. Type-level tests enforce this.

---

## 2.2: AgentHarness — run_bounded() with Context Management

**Description**: Add `run_bounded()` to `AgentHarness`. This method drives at most
`max_iterations` of the LLM + tool loop, then returns a `PhaseResult`. It introduces
multi-context support via `context_label`: the harness maintains a dictionary of named
conversation histories so agents can switch between independent LLM contexts.

**Context label design**:

The harness stores `_context_histories: dict[Optional[str], list]`, keyed by label.
- `context_label=None` (default) accesses the primary unnamed context
- `context_label="subtopic_A"` creates or resumes a named context
- Agents switch contexts simply by passing a different label on the next `_run_phase()` call
- `continue_context=False` resets the history for the current label only, not all contexts
- All accounting (tokens, cost) accumulates globally across all contexts — labels do not
  create separate budgets

**Tasks**:
- Add `_context_histories: dict[Optional[str], list]` to `AgentHarness.__init__()`
- Add `run_bounded(user_message, tool_names, max_iterations, continue_context, context_label, direct_tool_calls=None) -> PhaseResult`:
  - **Tool-only mode** (`direct_tool_calls` is not `None`): execute the provided list of tool
    calls directly via the sandbox without making any LLM call
    - **Stop-requested check** still applies at entry
    - **Budget check does NOT apply**: tool-only calls do not consume LLM budget
    - **Runtime safety checks** apply to every tool call before execution
    - Return `PhaseResult(final_text="", tool_calls=[...], stop_reason="done")`
  - **LLM mode** (`direct_tool_calls` is `None`):
    - Resolve the conversation history for the given `context_label` from `_context_histories`
    - If `continue_context=False`, reset that label's history to `[]`
    - At the start of each iteration: check budget and stop-requested state
    - Drive the LLM + tool loop up to `max_iterations`
  - Return `PhaseResult(final_text=..., tool_calls=[...], stop_reason=...)`
- Refactor `run()` to call `run_bounded()` internally

**Files**:
- `lsm/agents/harness.py` — `AgentHarness`
- `tests/agents/test_harness.py` — extend with run_bounded() coverage

**Success criteria**: `run_bounded()` exists and passes all unit tests. `run()` wraps
`run_bounded()` transparently. Context switching verified: named contexts are independent.

---

## 2.3: BaseAgent — Workspace Accessor Methods

**Description**: Add workspace path accessor methods and a filename helper to `BaseAgent`.
All agents must use these methods; no agent may construct paths by hand.

**Tasks**:
- Add the following methods to `BaseAgent`:
  ```python
  def _workspace_root(self) -> Path
  def _artifacts_dir(self) -> Path
  def _logs_dir(self) -> Path
  def _memory_dir(self) -> Path
  def _artifact_filename(self, name: str, suffix: str = ".md") -> str
  ```
- `_artifact_filename(name, suffix)`:
  - `timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")`
  - `safe_name = re.sub(r"[^\w\-]", "_", name).strip("_")`
  - Returns `f"{safe_name}_{timestamp}{suffix}"`

**Files**:
- `lsm/agents/base.py` — new methods on `BaseAgent`
- `tests/agents/test_base_agent.py` — new or extended tests for all accessors

**Success criteria**: All workspace accessor methods return correct `Path` objects. Directories
are created on first access. `_artifact_filename()` produces consistently formatted filenames.

---

## 2.4: BaseAgent — _run_phase() with context_label

**Description**: Add `_run_phase()` to `BaseAgent`. This is the only method agents may use
for LLM and tool activity.

**Tasks**:
- Add `_harness: Optional[AgentHarness] = None` as an instance attribute on `BaseAgent`
- Add `_run_phase()`:
  ```python
  def _run_phase(
      self,
      system_prompt: str = "",
      user_message: str = "",
      tool_names: Optional[list[str]] = None,
      max_iterations: int = 10,
      continue_context: bool = True,
      context_label: Optional[str] = None,
      direct_tool_calls: Optional[list[dict]] = None,
  ) -> PhaseResult:
  ```
  - On first call: create `AgentHarness` and store as `self._harness`
  - On subsequent calls: reuse `self._harness`
  - Delegate to `self._harness.run_bounded(...)`
- Ensure `_harness` is reset to `None` at the start of each `run()` call
- **Security invariants** — `BaseAgent` must have:
  - NO `_check_budget_and_stop()` method
  - NO `_tokens_used` attribute
  - No provider imports

**Files**:
- `lsm/agents/base.py` — `_harness`, `_run_phase()`, `_reset_harness()` on `BaseAgent`
- `tests/agents/test_base_agent.py` — extend with harness lifecycle and security invariant tests

**Success criteria**: `_run_phase()` creates one harness per `run()` lifecycle. Security
invariants are enforced by tests.

---

## 2.5: Phase 2 Code Review and Changelog

**Tasks**:
- Review `run_bounded()` budget enforcement, tool-only mode, and context isolation
- Review workspace accessors and harness lifecycle
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 2 changes
- Update `.agents/docs/architecture/packages/lsm.agents.md` and `AGENTS.md`

---

# Phase 3: query_knowledge_base Tool

**Status**: Complete

Implements the `query_knowledge_base` tool that wraps `query_sync()` and calls the full query
pipeline. Removes and deletes `query_embeddings`.

Reference: [RESEARCH_PLAN.md §3](../docs/RESEARCH_PLAN.md#3-knowledge-base-access-query_knowledge_base-tool)

---

## 3.1: QueryKnowledgeBaseTool Implementation

**Description**: Create `QueryKnowledgeBaseTool`, a built-in agent tool that calls the full
`query_sync()` pipeline and returns a JSON-serialized result.

**Tasks**:
- Create `lsm/agents/tools/query_knowledge_base.py`:
  - Class `QueryKnowledgeBaseTool(BaseTool)`
  - `name = "query_knowledge_base"`
  - `risk_level = "read_only"`
  - `input_schema`: `query` (string, required), optional `top_k` (int), optional `filters`
  - `execute()`: validate `query`, construct fresh `SessionState`, call `query_sync()`, return JSON

**Files**:
- `lsm/agents/tools/query_knowledge_base.py` — new file
- `tests/agents/tools/test_query_knowledge_base.py` — full test coverage

**Success criteria**: `QueryKnowledgeBaseTool` runs end-to-end against a mocked `query_sync()`.

---

## 3.2: Tool Registry — Add New Tool, Remove query_embeddings

**Description**: Register `query_knowledge_base` in the default tool registry and delete
`query_embeddings` entirely.

**Tasks**:
- Add `QueryKnowledgeBaseTool` to `create_default_tool_registry()`
- Remove `QueryEmbeddingsTool` registration
- Delete `lsm/agents/tools/query_embeddings.py`
- Remove `query_embeddings` from all agent `tool_allowlist` definitions

**Files**:
- `lsm/agents/tools/__init__.py` — register new tool, deregister old
- `lsm/agents/tools/query_embeddings.py` — DELETE
- `tests/agents/tools/test_registry.py` — verify absence of query_embeddings, presence of query_knowledge_base

**Success criteria**: `query_knowledge_base` is in the default registry. `query_embeddings` is
absent and deleted from disk. No import errors across the codebase.

---

## 3.3: Phase 3 Code Review and Changelog

**Tasks**:
- Verify no remaining references to `query_embeddings` anywhere in the codebase
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` and `.agents/docs/architecture/packages/lsm.agents.md`

---

# Phase 4: Simple Agent Migration

**Status**: Complete

Migrates `GeneralAgent`, `LibrarianAgent`, and `ManuscriptEditorAgent` from direct
`AgentHarness` instantiation to `self._run_phase()`.

Reference: [RESEARCH_PLAN.md §2.9](../docs/RESEARCH_PLAN.md#29-scope-of-agent-migrations)

---

## 4.1: GeneralAgent Migration

**Tasks**:
- Remove `harness = AgentHarness(...)` construction from `GeneralAgent.run()`
- Replace with `result = self._run_phase(system_prompt=..., user_message=task.objective, ...)`
- Apply workspace accessor changes for any file output
- Remove `AgentHarness` import if unused

**Files**:
- `lsm/agents/productivity/general.py`
- `tests/agents/test_general_agent.py` — verify no direct AgentHarness instantiation

---

## 4.2: LibrarianAgent Migration

Same pattern as 4.1 applied to `LibrarianAgent`.

**Files**:
- `lsm/agents/productivity/librarian.py`
- `tests/agents/test_librarian_agent.py`

---

## 4.3: ManuscriptEditorAgent Migration

Same pattern as 4.1 applied to `ManuscriptEditorAgent`.

**Files**:
- `lsm/agents/productivity/manuscript_editor.py`
- `tests/agents/test_manuscript_editor_agent.py`

---

## 4.4: Phase 4 Code Review and Changelog

**Tasks**:
- Grep each agent file for `AgentHarness`, `sandbox.execute()`, `provider.synthesize()` — must be empty
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`

---

# Phase 5: Academic Agents Migration

**Status**: Complete

Migrates `ResearchAgent`, `SynthesisAgent`, and `CuratorAgent` from manual execution
patterns to `_run_phase()`. `ResearchAgent` gains `query_knowledge_base` integration,
per-subtopic `context_label` usage, improved subtopic logging, and corrected artifact paths.

Reference: [RESEARCH_PLAN.md §2.9](../docs/RESEARCH_PLAN.md#29-scope-of-agent-migrations)

---

## 5.1: ResearchAgent Refactor

**Description**: Full refactor of `ResearchAgent` to use `_run_phase()`.

**Phase orchestration** — rewrite `run()`:
- Phase 1 — DECOMPOSE: `_run_phase(..., tool_names=[], max_iterations=1)`
- Phase 2 — RESEARCH (per subtopic): `_run_phase(..., tool_names=["query_knowledge_base"], max_iterations=3, context_label=f"subtopic:{subtopic}")`
- Phase 3 — SYNTHESIZE: `_run_phase(..., tool_names=[], max_iterations=1, context_label=None)`
- Phase 4 — REVIEW: `_run_phase(...)` in primary context

**Subtopic logging**:
- `subtopic_lines = "\n".join(f"  [{i}] {st}" for i, st in enumerate(subtopics, 1))`
- `self._log(f"Research iteration {iteration} — {len(subtopics)} subtopics:\n{subtopic_lines}")`

**Artifact output**:
- `output_path = self._artifacts_dir() / self._artifact_filename(safe_topic)`

**Removed methods** (delete entirely):
- `_select_tools()`, `_collect_findings()`, `_summarize_findings()`, `_build_sources_block()`, `_extract_sources_from_output()`
- `_tokens_used` counter and all manual token tracking

**Files**:
- `lsm/agents/academic/research.py`
- `tests/agents/academic/test_research_agent.py`

---

## 5.2: SynthesisAgent Refactor

- Map phases to `_run_phase()` calls
- Remove direct provider/sandbox calls, manual token tracking, hardcoded paths

**Files**:
- `lsm/agents/academic/synthesis.py`
- `tests/agents/academic/test_synthesis_agent.py`

---

## 5.3: CuratorAgent Refactor

Same pattern as 5.2 applied to `CuratorAgent`.

**Files**:
- `lsm/agents/academic/curator.py`
- `tests/agents/academic/test_curator_agent.py`

---

## 5.4: Phase 5 Code Review and Changelog

**Tasks**:
- Grep `lsm/agents/academic/` for `AgentHarness`, `sandbox.execute`, `provider.synthesize`, `_tokens_used` — all must be empty
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` and `AGENTS.md`

---

# Phase 6: Remaining Agent Migration

**Status**: Complete

Audits and refactors all remaining agents: `WritingAgent`, `AssistantAgent`,
`NewsAssistantAgent`, `CalendarAssistantAgent`, `EmailAssistantAgent`, and `MetaAgent`.

**LLM-driven agents** (migrate to `_run_phase()`):
- `WritingAgent` — 3-phase structure (OUTLINE → DRAFT → REVIEW)

**Data-pipeline agents** (workspace accessors + token cleanup + tool-only `_run_phase()` for tool dispatch):
- `AssistantAgent`, `NewsAssistantAgent`, `CalendarAssistantAgent`, `EmailAssistantAgent`

**Orchestration agents** (audit only):
- `MetaAgent`

Reference: [RESEARCH_PLAN.md §2.9](../docs/RESEARCH_PLAN.md#29-scope-of-agent-migrations)

---

## 6.1: WritingAgent Audit and Refactor

**Pre-audit findings**:
- `_tokens_used` counter — REMOVE
- `create_provider()` call at top of `run()` — REMOVE
- `_collect_grounding()`, `_build_outline()`, `_draft_deliverable()`, `_review_deliverable()`, `_run_tool()`, `_extract_source_paths()` — REMOVE all
- `_save_deliverable()` — REPLACE path with workspace accessors
- `query_embeddings` in `tool_allowlist` — REPLACE with `query_knowledge_base`

**Rewritten as 3-phase orchestration**:
- Phase 1 — OUTLINE: `_run_phase(..., tool_names=["query_knowledge_base"], max_iterations=3, context_label="outline")`
- Phase 2 — DRAFT: `_run_phase(..., tool_names=[], max_iterations=1, context_label="draft", continue_context=False)`
- Phase 3 — REVIEW: `_run_phase(..., tool_names=[], max_iterations=1, context_label="draft", continue_context=True)`

**Files**:
- `lsm/agents/productivity/writing.py`
- `tests/agents/test_writing_agent.py`

---

## 6.2: AssistantAgent Audit and Refactor

**Pre-audit findings**:
- `_tokens_used` — REMOVE
- `_run_tool()` for `memory_put` — REMOVE; replace with `_run_phase(direct_tool_calls=[...])`
- `_resolve_output_dir()` — REPLACE with workspace accessors

**Files**:
- `lsm/agents/assistants/assistant.py`
- `tests/agents/test_assistant_agent.py`

---

## 6.3: NewsAssistantAgent Audit and Refactor

**Pre-audit findings**:
- `_tokens_used` — REMOVE
- `_resolve_output_dir()` — REPLACE with workspace accessors
- Direct `create_remote_provider()` calls — KEEP (intended pattern)

**Files**:
- `lsm/agents/assistants/news_assistant.py`
- `tests/agents/test_news_assistant_agent.py`

---

## 6.4: CalendarAssistantAgent Audit and Refactor

**Pre-audit findings**:
- `_tokens_used` — REMOVE
- `_run_tool()` for `ask_user` — REMOVE; replace with `_run_phase(direct_tool_calls=[...])`
- `_bind_interaction_tools()` — VERIFY compatible with new harness architecture
- `_resolve_output_dir()` — REPLACE with workspace accessors

**Files**:
- `lsm/agents/assistants/calendar_assistant.py`
- `tests/agents/test_calendar_assistant_agent.py`

---

## 6.5: EmailAssistantAgent Audit and Refactor

Same scope as 6.4 applied to `EmailAssistantAgent`.

**Files**:
- `lsm/agents/assistants/email_assistant.py`
- `tests/agents/test_email_assistant_agent.py`

---

## 6.6: MetaAgent Audit and Refactor

**Tasks**:
- Audit for `create_provider`, `provider.synthesize`, `AgentHarness`, `_tokens_used`, `sandbox.execute`
- For any direct LLM calls: replace with `_run_phase()`
- Remove `_tokens_used` and manual token tracking
- Replace hardcoded output paths with workspace accessors

**Files**:
- `lsm/agents/meta/meta.py`
- `tests/agents/test_meta_agent.py`

---

## 6.7: Phase 6 Code Review and Changelog

**Tasks**:
- Grep `lsm/agents/` for `AgentHarness`, `sandbox.execute`, `provider.synthesize`, `_tokens_used`, `query_embeddings` — all results must be empty (outside harness.py and base.py)
- Review all `_bind_interaction_tools()` implementations
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` and `AGENTS.md`

---

# Phase 7: query_remote Tool Redesign

**Status**: Complete

Redesigns `QueryRemoteTool` from a single tool that accepts a `provider` name argument to a
factory-based pattern where each configured remote source gets its own named tool instance.

**Current design**: Single `QueryRemoteTool` with `name = "query_remote"`.

**New design**: Each configured remote source becomes its own tool instance:
- Instance `name` returns `f"query_{provider_cfg.name}"`
- `create_default_tool_registry()` instantiates one `QueryRemoteTool` per configured remote source
- Agents declare which remote sources they want via `remote_source_allowlist: Optional[set[str]] = None`

---

## 7.1: QueryRemoteTool — Per-Source Instance Design

**Tasks**:
- Refactor `QueryRemoteTool.__init__` to accept `RemoteProviderConfig` and `LSMConfig`
- Change `name` to a property: `return f"query_{self._provider_cfg.name}"`
- Remove `provider` parameter from `input_schema` and `execute()`
- Delete `_find_provider()` helper

**Files**:
- `lsm/agents/tools/query_remote.py`
- `tests/agents/tools/test_query_remote.py`

---

## 7.2: Tool Registry — Per-Source Tool Instantiation

**Tasks**:
- In `create_default_tool_registry()`: iterate `config.remote_providers` and register one `QueryRemoteTool` per provider
- Add `remote_source_allowlist: Optional[set[str]] = None` class attribute to `BaseAgent`
- In `AgentHarness` tool filtering: filter per-source tools against the agent's `remote_source_allowlist`

**Files**:
- `lsm/agents/tools/__init__.py`
- `lsm/agents/base.py`
- `lsm/agents/harness.py`
- `tests/agents/tools/test_registry.py` — verify per-source tools, no generic query_remote, allowlist filtering

---

## 7.3: Update NewsAssistantAgent, CalendarAssistantAgent, EmailAssistantAgent

**Description**: Remove `_resolve_lsm_config()` hack that looked up `query_remote` from the
tool registry. Replace with direct `lsm_config` injection.

**Tasks**:
- Add `lsm_config: LSMConfig` as a constructor parameter to all three agents
- `_resolve_lsm_config()` becomes `return self.lsm_config`
- Update agent factory to inject `lsm_config` at construction time

**Files**:
- `lsm/agents/assistants/news_assistant.py`
- `lsm/agents/assistants/calendar_assistant.py`
- `lsm/agents/assistants/email_assistant.py`
- `lsm/agents/factory.py`

---

## 7.4: Phase 7 Code Review and Changelog

**Tasks**:
- Grep entire codebase for `"query_remote"` string — must be empty outside tests asserting absence
- Grep for `_find_provider` — must be empty
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`, `lsm.agents.md`, `REMOTE.md`

---

# Phase 8: Final Code Review and Release

**Status**: Complete

---

## 8.1: Test Suite Cleanup

**Tasks**:
- Confirm every agent file has tests asserting: no direct AgentHarness, no direct provider calls, no direct sandbox.execute, no `_tokens_used`
- Verify `query_embeddings` and `query_remote` (generic) absence tests are in place
- Remove tests for deleted methods
- Run full test suite: `pytest tests/ -v`

---

## 8.2: docs/user-guide/AGENTS.md Update

**Tasks**:
- For each agent: document `tool_allowlist`, `remote_source_allowlist`, brief workflow
- Ensure tool names reflect v0.7.1 state (`query_knowledge_base`, per-source `query_<name>`)

---

## 8.3: Final Complete Code Review

**Tasks**:
- Grep entire `lsm/agents/` for `AgentHarness(`, `sandbox.execute`, `provider.synthesize`, `provider.complete`, `_tokens_used`, `create_provider(` — all must be empty outside harness.py and base.py
- Grep for `query_embeddings` — must be empty
- Grep for `"query_remote"` — must be empty outside tests
- Review `PhaseResult`, `BaseAgent` security invariants, `run_bounded()` budget/context isolation
- Run full test suite with coverage: `pytest tests/ -v --cov=lsm --cov-report=html`

---

## 8.4: Final Documentation and Release

**Tasks**:
- Bump version to `0.7.1` in `pyproject.toml`, `lsm/__init__.py`, `README.md`, `lsm/ui/tui/screens/help.py`
- Add `## 0.7.1 - <date>` entry to `docs/CHANGELOG.md`
- Update `example_config.json`: verify `acknowledged_timeout_seconds` under `agents.interaction`
- Final review of `AGENTS.md` and `lsm.agents.md`

**Success criteria**: `pytest tests/ -v` passes. All files reflect `0.7.1`. Release committed and pushed.
