# Phase 2: AgentHarness and BaseAgent Infrastructure

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1

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
- Run relevant tests: `pytest tests/agents/test_phase_result.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

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
    calls directly via the sandbox without making any LLM call; `user_message`, `tool_names`,
    `max_iterations`, and `continue_context` are ignored in this mode
    - **Stop-requested check** still applies at entry: if stop has been signaled, return
      `PhaseResult(final_text="", tool_calls=[], stop_reason="stop_requested")` immediately
    - **Budget check does NOT apply**: budget tracks LLM token usage; tool-only calls do not
      consume LLM budget and must not be gated by it
    - **Runtime safety checks** apply to every tool call before execution:
      - Validate the tool name against the agent's `tool_allowlist`; if a tool is not
        permitted, record an error result for that call and continue (do not raise)
      - Sandbox risk-level and rate-limit checks apply as normal (same path as LLM-driven
        tool execution) to prevent runaway or high-frequency tool calls
    - Return `PhaseResult(final_text="", tool_calls=[...], stop_reason="done")`
  - **LLM mode** (`direct_tool_calls` is `None`):
    - Resolve the conversation history for the given `context_label` from `_context_histories`
    - If `continue_context=False`, reset that label's history to `[]`
    - At the start of each iteration: check budget and stop-requested state; if either is true,
      return the appropriate `PhaseResult` immediately without making an LLM call
    - Drive the LLM + tool loop (using existing harness loop logic, factored out of `run()`):
      - Call LLM with current history + user_message
      - Parse tool requests
      - Validate each tool against allowlist; if `tool_names` is not None, further restrict
        to that subset
      - Execute permitted tools via sandbox
      - Accumulate results into history for the current label
      - Repeat up to `max_iterations`
  - Return `PhaseResult(final_text=..., tool_calls=[...], stop_reason=...)`
  - Token accumulation, cost tracking, and run summary updates remain inside this method
  - `PhaseResult` never contains token or cost data
- Refactor `run()` to call `run_bounded()` internally:
  - `run()` initializes the harness (system prompt, memory context) then calls `run_bounded()`
    in a loop until `stop_reason in ("done", "budget_exhausted", "stop_requested")`
  - All observable behavior of `run()` for existing callers is preserved
- Run relevant tests: `pytest tests/agents/test_harness.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/harness.py` — `AgentHarness`:
  - `__init__`: add `_context_histories`
  - Add `run_bounded()`
  - Refactor `run()` to wrap `run_bounded()`
- `tests/agents/test_harness.py` — extend or new:
  - Test: `run_bounded()` returns `stop_reason="done"` when LLM signals done before limit
  - Test: `run_bounded()` returns `stop_reason="max_iterations"` at the limit
  - Test: `run_bounded()` returns `stop_reason="budget_exhausted"` immediately on entry if
    budget is exhausted — no LLM call is made
  - Test: after budget exhaustion, a subsequent `run_bounded()` call also returns immediately
  - Test: `run_bounded()` returns `stop_reason="stop_requested"` when stop is signaled
  - Test: `PhaseResult` from `run_bounded()` has no `tokens_used` or `cost_usd` field
  - Test: `continue_context=True` — conversation history from prior phase is included in LLM
    messages (captured via mock LLM)
  - Test: `continue_context=False` — history for that label is reset; LLM receives no prior context
  - Test: `context_label="label_a"` and `context_label="label_b"` maintain independent histories;
    switching between them does not contaminate each other
  - Test: `context_label=None` (default) uses a separate primary context from any named context
  - Test: `tool_names=["tool_a"]` restricts LLM to only `tool_a` even if `tool_b` is in allowlist
  - Test: `tool_names=[]` — LLM executes but tool calls are rejected
  - Test: `tool_names=None` — all allowlisted tools are available
  - Test: `direct_tool_calls=[{...}]` — no LLM call is made; tools are executed and results
    appear in `PhaseResult.tool_calls`; `stop_reason="done"`
  - Test: `direct_tool_calls=[]` — no LLM call is made; `PhaseResult.tool_calls` is empty;
    `stop_reason="done"`
  - Test: `direct_tool_calls` with stop requested at entry — returns immediately without
    executing any tool; `stop_reason="stop_requested"`
  - Test: `direct_tool_calls` with budget exhausted does NOT short-circuit — tools still
    execute; budget state is irrelevant in tool-only mode
  - Test: `direct_tool_calls` containing a tool not in `tool_allowlist` — that call produces
    an error result in `PhaseResult.tool_calls`; other permitted calls still execute
  - Test: `direct_tool_calls` sandbox rate-limit/risk checks apply — a tool exceeding the
    sandbox risk level produces an error result for that call rather than raising
  - Test: token totals in `run_summary.json` are the sum across all `run_bounded()` calls in
    a single `run()` lifecycle
  - Test: `AgentHarness.run()` behavior is identical to pre-refactor for single-phase agents

**Success criteria**: `run_bounded()` exists and passes all unit tests. `run()` wraps
`run_bounded()` transparently. All existing harness tests continue to pass. Context switching
is verified: named contexts are independent and the primary context is distinct from all named ones.

---

## 2.3: BaseAgent — Workspace Accessor Methods

**Description**: Add workspace path accessor methods and a filename helper to `BaseAgent`.
All agents must use these methods; no agent may construct paths by hand.

**Tasks**:
- Import `datetime`, `re`, and `Path` (if not already imported) in `base.py`
- Add the following methods to `BaseAgent`:
  ```python
  def _workspace_root(self) -> Path
  def _artifacts_dir(self) -> Path
  def _logs_dir(self) -> Path
  def _memory_dir(self) -> Path
  def _artifact_filename(self, name: str, suffix: str = ".md") -> str
  ```
- `_workspace_root()` calls `ensure_agent_workspace(self.name, self.agent_config.agents_folder)`
  and returns the result (a `Path`); directories are created if absent
- `_artifacts_dir()`, `_logs_dir()`, `_memory_dir()` return the appropriate subdirectory
  under `_workspace_root()`
- `_artifact_filename(name, suffix)`:
  - `timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")`
  - `safe_name = re.sub(r"[^\w\-]", "_", name).strip("_")`
  - Returns `f"{safe_name}_{timestamp}{suffix}"`
- Run relevant tests: `pytest tests/agents/test_base_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/base.py` — new methods on `BaseAgent`
- `tests/agents/test_base_agent.py` — new or extended:
  - Test: `_artifacts_dir()` returns `Path(<agents_folder>/<agent_name>/artifacts/)`
  - Test: `_logs_dir()` returns `Path(<agents_folder>/<agent_name>/logs/)`
  - Test: `_memory_dir()` returns `Path(<agents_folder>/<agent_name>/memory/)`
  - Test: `_workspace_root()` creates directories on the filesystem when they do not exist
  - Test: `_artifact_filename("some topic name")` → `"some_topic_name_YYYYMMDD_HHMMSS.md"`
  - Test: `_artifact_filename()` sanitizes spaces, colons, slashes to `_`
  - Test: `_artifact_filename()` strips leading/trailing underscores from sanitized name
  - Test: `_artifact_filename("name", suffix=".json")` uses the given suffix

**Success criteria**: All workspace accessor methods return correct `Path` objects. Directories
are created on first access. `_artifact_filename()` produces consistently formatted filenames.

---

## 2.4: BaseAgent — _run_phase() with context_label

**Description**: Add `_run_phase()` to `BaseAgent`. This is the only method agents may use
for LLM and tool activity. It creates the shared harness on first call and reuses it for
subsequent calls. The `context_label` parameter allows agents to route a phase into a named
conversation context, enabling independent parallel contexts within a single run.

**Tasks**:
- Add `_harness: Optional[AgentHarness] = None` as an instance attribute on `BaseAgent`
  (initialized to `None` in `__init__`)
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
  - On first call (`_harness is None`): create `AgentHarness` using `system_prompt`,
    `self.llm_registry`, `self.tool_registry`, `self.sandbox`, `self.agent_config`; store
    as `self._harness`
  - On subsequent calls: reuse `self._harness`; `system_prompt` is ignored (harness is already
    initialized) — log a debug warning if `system_prompt` differs from the original
  - When `direct_tool_calls` is not `None`: `system_prompt` and `user_message` are not used
    (the harness is still created if needed for sandbox access, but no LLM call is made)
  - Delegate to `self._harness.run_bounded(user_message, tool_names, max_iterations,
    continue_context, context_label, direct_tool_calls=direct_tool_calls)`
  - Return the `PhaseResult` from `run_bounded()`
- Ensure `_harness` is reset to `None` at the start of each `run()` call so that successive
  invocations of `run()` on the same agent instance each get a fresh harness (implement via
  a `_reset_harness()` helper called at the top of the concrete `run()` override, or via a
  `BaseAgent.run()` wrapper that subclasses call with `super()`)
- **Security invariants** — verify by inspection that `BaseAgent`:
  - Has NO `_check_budget_and_stop()` method
  - Has NO `_tokens_used` attribute
  - Does not import token counting or cost utilities
- Run relevant tests: `pytest tests/agents/test_base_agent.py tests/agents/test_harness.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Context label — recommended usage pattern for agents**:

```python
# Example: ResearchAgent collecting per-subtopic findings in independent contexts
for subtopic in subtopics:
    result = self._run_phase(
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        user_message=f"Phase: RESEARCH. Subtopic: '{subtopic}'.",
        tool_names=["query_knowledge_base"],
        max_iterations=3,
        context_label=f"subtopic:{subtopic}",  # each subtopic has its own context
    )
    if result.stop_reason in ("budget_exhausted", "stop_requested"):
        break

# Synthesis in the primary context (no label), uncontaminated by per-subtopic conversations
result = self._run_phase(
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    user_message="Phase: SYNTHESIZE. Write the research outline.",
    tool_names=[],
    context_label=None,  # primary context
)
```

**Files**:
- `lsm/agents/base.py` — `_harness`, `_run_phase()`, `_reset_harness()` on `BaseAgent`
- `tests/agents/test_base_agent.py` — extend:
  - Test: first `_run_phase()` call creates exactly one `AgentHarness` instance (spy on init)
  - Test: second `_run_phase()` call on same agent instance reuses the same harness (init
    count remains 1)
  - Test: second call to `run()` on same agent creates a fresh harness (init count = 2 total)
  - Test: `_run_phase()` returns a `PhaseResult` with no token/cost fields
  - Test: `context_label` is forwarded to `run_bounded()` (verify via mock harness)
  - Test: `direct_tool_calls=[{...}]` — no LLM call is made; `run_bounded()` receives
    `direct_tool_calls` and executes tools without an LLM round-trip
  - Test: `direct_tool_calls` mode still creates/reuses the harness (sandbox still needed)
  - Test: `BaseAgent` has no `_check_budget_and_stop` attribute (assert `AttributeError`)
  - Test: `BaseAgent` has no `_tokens_used` attribute (assert `AttributeError`)

**Success criteria**: `_run_phase()` creates one harness per `run()` lifecycle. Context labels
are forwarded correctly. Security invariants are enforced by tests. `PhaseResult` is the only
return value.

---

## 2.5: Phase 2 Code Review and Changelog

**Tasks**:
- Review `run_bounded()` budget enforcement: verify budget check is before LLM call in every
  LLM-mode iteration path, not after; verify no cost data reaches `PhaseResult`
- Review `run_bounded()` tool-only mode: verify budget check is NOT applied; verify
  stop-requested check IS applied; verify allowlist and sandbox safety checks are applied
  per tool call; verify a disallowed tool produces an error result, not an exception
- Review `_run_phase()` harness lifecycle: verify `_harness` reset at start of `run()` and
  not during `_run_phase()` calls
- Review `context_label` implementation: verify named contexts do not share history; verify
  token budget is global (not per-context)
- Review workspace accessors: verify all return `pathlib.Path`; verify directories are created
  lazily (on access, not on agent construction)
- Review tests: no mocks or stubs; genuine concurrent tests for any thread-safety concerns
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 2 changes
- Update `.agents/docs/architecture/packages/lsm.agents.md`: document `PhaseResult`,
  `_run_phase()`, `run_bounded()`, workspace accessors, and `context_label`
- Update `.agents/docs/architecture/development/AGENTS.md`: update `BaseAgent` section
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/packages/lsm.agents.md`
- `.agents/docs/architecture/development/AGENTS.md`

**Success criteria**: `pytest tests/ -v` passes. All Phase 2 infrastructure is documented.
