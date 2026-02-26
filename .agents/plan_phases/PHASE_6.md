# Phase 6: Remaining Agent Migration

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1

Audits and refactors all remaining agents: `WritingAgent`, `AssistantAgent`,
`NewsAssistantAgent`, `CalendarAssistantAgent`, `EmailAssistantAgent`, and `MetaAgent`.

Agents are grouped into two categories based on their pre-audit findings:

**LLM-driven agents** (must migrate to `_run_phase()`):
- `WritingAgent` — has `_tokens_used`, direct `provider.synthesize()` (3 calls in
  `_build_outline`, `_draft_deliverable`, `_review_deliverable`), direct `sandbox.execute()`
  via `_run_tool()`, hardcoded output path in `_save_deliverable()`, uses `query_embeddings`

**Data-pipeline agents** (no LLM calls; apply workspace accessors and token cleanup only):
- `AssistantAgent` — pure data aggregation (loads run summaries, builds stats); only
  uses `sandbox.execute()` for `memory_put` via `_run_tool()`; has `_tokens_used`; hardcoded
  output path via `_resolve_output_dir()`
- `NewsAssistantAgent` — pure data pipeline; fetches news directly from remote providers
  (no sandbox, no LLM); has `_tokens_used`; hardcoded output path via `_resolve_output_dir()`
- `CalendarAssistantAgent` — direct remote provider calls + approval flow via `ask_user`;
  no LLM; has `_tokens_used`; hardcoded path; `_bind_interaction_tools()` creates a fake
  harness stub for `ask_user` — this must be reviewed post-harness refactor
- `EmailAssistantAgent` — same pattern as `CalendarAssistantAgent`; direct remote provider
  calls + approval flow; no LLM; has `_tokens_used`; hardcoded path; `_bind_interaction_tools()`

**Orchestration agents** (audit only; scope of changes may vary):
- `MetaAgent` — orchestrates sub-agents via `spawn_agent`/`await_agent`; has direct
  `create_provider()` import; must be audited for any `AgentHarness` direct usage and
  `_tokens_used`

Reference: [RESEARCH_PLAN.md §2.9](../docs/RESEARCH_PLAN.md#29-scope-of-agent-migrations)

---

## 6.1: WritingAgent Audit and Refactor

**Description**: Full refactor of `WritingAgent` to use `_run_phase()`. The agent currently
has a 3-phase structure (OUTLINE → DRAFT → REVIEW) implemented via direct `provider.synthesize()`
calls. This entire pattern is replaced by `_run_phase()` calls.

**Pre-audit findings**:
- `_tokens_used` counter — REMOVE
- `create_provider()` call at top of `run()` — REMOVE
- `_collect_grounding()` — calls tools via direct `sandbox.execute()` and inspects `query_embeddings` — REMOVE entire method; grounding is now handled via `query_knowledge_base` in the OUTLINE phase
- `_build_outline()` — `provider.synthesize()` call — REMOVE entire method
- `_draft_deliverable()` — `provider.synthesize()` call — REMOVE entire method
- `_review_deliverable()` — `provider.synthesize()` call — REMOVE entire method
- `_run_tool()` — direct `sandbox.execute()` — REMOVE
- `_extract_source_paths()` — helper for removed `_collect_grounding()` — REMOVE
- `_save_deliverable()` — hardcoded path — REPLACE with workspace accessors
- `query_embeddings` in `tool_allowlist` — REPLACE with `query_knowledge_base`

**Tasks**:
- Rewrite `WritingAgent.run()` as a 3-phase `_run_phase()` orchestration:
  - Phase 1 — OUTLINE: `self._run_phase(system_prompt=WRITING_SYSTEM_PROMPT, user_message=f"Phase: OUTLINE. Topic: '{topic}'. Use query_knowledge_base to gather evidence and produce a grounded outline.", tool_names=["query_knowledge_base"], max_iterations=3, context_label="outline")`
  - Phase 2 — DRAFT: `self._run_phase(..., user_message="Phase: DRAFT. Using the outline, write the full grounded deliverable.", tool_names=[], max_iterations=1, context_label="draft", continue_context=False)`
  - Phase 3 — REVIEW: `self._run_phase(..., user_message="Phase: REVIEW. Revise the draft for clarity and factual grounding. Return final markdown only.", tool_names=[], max_iterations=1, context_label="draft", continue_context=True)`
  - Check `result.stop_reason` after each phase; skip remaining phases if `"budget_exhausted"` or `"stop_requested"`
- Replace `_save_deliverable()` path construction with workspace accessors:
  `output_path = self._artifacts_dir() / self._artifact_filename(topic)`
- Delete removed methods listed above
- Remove `create_provider` import if no longer used
- Remove `AgentHarness` import if no longer used
- Update `tool_allowlist`: replace `"query_embeddings"` with `"query_knowledge_base"`
- Run relevant tests: `pytest tests/agents/test_writing_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/productivity/writing.py` — major rewrite
- `tests/agents/test_writing_agent.py` — update:
  - Test: phases execute in order OUTLINE → DRAFT → REVIEW
  - Test: OUTLINE phase calls `_run_phase()` with `tool_names=["query_knowledge_base"]`
  - Test: DRAFT and REVIEW phases call `_run_phase()` with `tool_names=[]`
  - Test: no direct `sandbox.execute()` call exists in `writing.py`
  - Test: no direct `provider.synthesize()` or `provider.complete()` call exists in `writing.py`
  - Test: `AgentHarness` is not directly instantiated in `writing.py`
  - Test: output file is written to `self._artifacts_dir()` with `_artifact_filename()` format
  - Test: when `stop_reason="budget_exhausted"` after OUTLINE, DRAFT and REVIEW are skipped
  - Test: `WritingResult.output_path` reflects the new `artifacts/` location
  - Test: `WritingAgent` has no `_tokens_used` attribute (assert `AttributeError`)

**Success criteria**: `WritingAgent.run()` uses `_run_phase()` exclusively. All deleted methods
are gone. Workspace accessor methods are used for all file paths. All tests pass.

---

## 6.2: AssistantAgent Audit and Refactor

**Description**: `AssistantAgent` is a pure data aggregation agent — it collects run summaries,
builds statistics, and writes output files. It does not call the LLM. Changes are:
removing `_tokens_used`, removing `_run_tool()` and routing its single `memory_put` call
through `_run_phase()`, and applying workspace accessors for output paths. Direct
`sandbox.execute()` calls must not remain.

**Pre-audit findings**:
- `_tokens_used` (line 76) — REMOVE
- `_run_tool()` (lines 307–327) — direct `sandbox.execute()`; only called for `memory_put` —
  REMOVE entirely; replace with `self._run_phase(direct_tool_calls=[{"tool": "memory_put",
  "args": <args>}])` so no LLM call is made
- `_resolve_output_dir()` — hardcoded path construction — REPLACE with workspace accessors

**Tasks**:
- Remove `_tokens_used = 0` initialization from `run()`
- Remove `_run_tool()` entirely; replace its single `memory_put` invocation with
  `self._run_phase(direct_tool_calls=[{"tool": "memory_put", "args": <args>}])`
- Replace `_resolve_output_dir()` path logic with `self._artifacts_dir() / self._artifact_filename(topic)`
  for the run directory
- Remove `_stop_logged = False` if this attribute is defined on `BaseAgent` only for
  `_tokens_used` lifecycle; verify and clean up
- Confirm `AssistantAgent` makes no direct `provider.synthesize()`, `AgentHarness`, or
  `sandbox.execute()` calls
- Run relevant tests: `pytest tests/agents/test_assistant_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/assistants/assistant.py`
- `tests/agents/test_assistant_agent.py` — update:
  - Test: `AssistantAgent` has no `_tokens_used` attribute (assert `AttributeError`)
  - Test: `run()` does not call `provider.synthesize()` or `provider.complete()`
  - Test: `run()` does not directly instantiate `AgentHarness` or call `sandbox.execute()`
  - Test: output files are written to `self._artifacts_dir()`

**Success criteria**: `AssistantAgent` has no `_tokens_used` and no direct `sandbox.execute()`
calls. Output paths use workspace accessors. All tests pass.

---

## 6.3: NewsAssistantAgent Audit and Refactor

**Description**: `NewsAssistantAgent` is a pure data pipeline — it fetches news from remote
providers directly and writes output files. No LLM, no sandbox. Migration to `_run_phase()` is
not applicable. Changes are limited to: removing `_tokens_used` and applying workspace accessors.

**Pre-audit findings**:
- `_tokens_used` (line 74) — REMOVE
- Direct calls to `create_remote_provider()` via `_resolve_providers()` — KEEP (intended pattern)
- `_resolve_output_dir()` — hardcoded path construction — REPLACE with workspace accessors
- No `sandbox.execute()` calls — no change needed
- No LLM calls — no change needed

**Tasks**:
- Remove `_tokens_used = 0` initialization from `run()`
- Replace `_resolve_output_dir()` path logic with workspace accessors
- Confirm no `provider.synthesize()`, `AgentHarness`, or `sandbox.execute()` calls exist
- Run relevant tests: `pytest tests/agents/test_news_assistant_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/assistants/news_assistant.py`
- `tests/agents/test_news_assistant_agent.py` — update:
  - Test: `NewsAssistantAgent` has no `_tokens_used` attribute (assert `AttributeError`)
  - Test: output files are written to `self._artifacts_dir()`
  - Test: no direct `AgentHarness` instantiation

**Success criteria**: `NewsAssistantAgent` has no `_tokens_used`. Output paths use workspace
accessors. All tests pass.

---

## 6.4: CalendarAssistantAgent Audit and Refactor

**Description**: `CalendarAssistantAgent` fetches calendar data directly from remote providers
and uses `ask_user` for approval gating. No LLM. Migration to `_run_phase()` is not applicable.
Changes: remove `_tokens_used`, apply workspace accessors, and review `_bind_interaction_tools()`
in context of the new harness architecture.

**Pre-audit findings**:
- `_tokens_used` (line 72) — REMOVE
- `_run_tool()` (lines 625–645) — direct `sandbox.execute()`; only called for `ask_user` —
  REMOVE entirely; replace with `self._run_phase(direct_tool_calls=[{"tool": "ask_user",
  "args": <args>}])` so the tool is dispatched through the harness sandbox without an LLM call
- `_bind_interaction_tools()` — creates a fake `_Harness` stub to bind `ask_user` to the
  interaction channel; VERIFY this stub is still compatible after Phase 2 harness changes;
  `ask_user` must still receive a valid interaction channel when called via `_run_phase()`
- `_resolve_output_dir()` — hardcoded path — REPLACE with workspace accessors
- Direct `create_remote_provider()` calls — KEEP (intended pattern)

**Tasks**:
- Remove `_tokens_used = 0` initialization from `run()`
- Remove `_run_tool()` entirely; replace each `ask_user` invocation with
  `self._run_phase(direct_tool_calls=[{"tool": "ask_user", "args": <args>}])`
- Replace `_resolve_output_dir()` path logic with workspace accessors
- Verify `_bind_interaction_tools()` correctly wires the interaction channel to `ask_user`
  when the tool is dispatched through `_run_phase(direct_tool_calls=[...])` — update if needed
- Confirm no direct LLM, `AgentHarness`, or `sandbox.execute()` calls exist
- Run relevant tests: `pytest tests/agents/test_calendar_assistant_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/assistants/calendar_assistant.py`
- `tests/agents/test_calendar_assistant_agent.py` — update:
  - Test: `CalendarAssistantAgent` has no `_tokens_used` attribute (assert `AttributeError`)
  - Test: no direct `sandbox.execute()` call
  - Test: output files are written to `self._artifacts_dir()`
  - Test: no direct `AgentHarness` instantiation
  - Test: `_bind_interaction_tools()` correctly binds interaction channel to `ask_user` when
    called via `_run_phase(direct_tool_calls=[...])`

**Success criteria**: `CalendarAssistantAgent` has no `_tokens_used` and no direct
`sandbox.execute()` calls. Output paths use workspace accessors. Interaction tool binding
verified via `_run_phase(direct_tool_calls=[...])`. All tests pass.

---

## 6.5: EmailAssistantAgent Audit and Refactor

**Description**: `EmailAssistantAgent` is the same pattern as `CalendarAssistantAgent` —
direct remote provider calls, `ask_user` approval gating, no LLM. Same scope of changes.

**Pre-audit findings**:
- `_tokens_used` (line 92) — REMOVE
- `_run_tool()` (lines 605–625) — direct `sandbox.execute()` for `ask_user` —
  REMOVE entirely; replace with `self._run_phase(direct_tool_calls=[{"tool": "ask_user",
  "args": <args>}])` — same approach as 6.4
- `_resolve_output_dir()` — hardcoded path — REPLACE with workspace accessors
- `_bind_interaction_tools()` — same fake harness stub pattern as CalendarAssistant —
  VERIFY post-refactor compatibility (same concern as 6.4)
- Direct `create_remote_provider()` calls — KEEP

**Tasks**:
- Same tasks as 6.4 applied to `EmailAssistantAgent`
- Remove `_tokens_used = 0` from `run()`
- Remove `_run_tool()` entirely; replace `ask_user` invocations with
  `self._run_phase(direct_tool_calls=[{"tool": "ask_user", "args": <args>}])`
- Replace `_resolve_output_dir()` with workspace accessors
- Verify `_bind_interaction_tools()` compatibility with `_run_phase(direct_tool_calls=[...])`
- Run relevant tests: `pytest tests/agents/test_email_assistant_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/assistants/email_assistant.py`
- `tests/agents/test_email_assistant_agent.py` — update with same patterns as 6.4

**Success criteria**: Same as 6.4 for `EmailAssistantAgent`.

---

## 6.6: MetaAgent Audit and Refactor

**Description**: `MetaAgent` orchestrates sub-agents via `spawn_agent`, `await_agent`, and
`collect_artifacts`. It imports `create_provider` from `lsm.providers.factory`. Audit all
execution paths, determine whether direct LLM calls exist, and apply the appropriate migration.

**Tasks**:
- Read `lsm/agents/meta/meta.py` fully to map all execution paths
- Audit: grep for `create_provider`, `provider.synthesize`, `AgentHarness`, `_tokens_used`,
  `sandbox.execute` — document all occurrences
- For any direct LLM calls: replace with `self._run_phase()` using an appropriate system prompt
- For any direct `AgentHarness` instantiation: remove and replace with `_run_phase()`
- Remove `_tokens_used` and all manual token tracking
- Replace any hardcoded file output paths with workspace accessors
- Confirm `MetaAgent` orchestration tools (`spawn_agent`, `await_agent`, `collect_artifacts`)
  continue to function correctly after refactor
- Run relevant tests: `pytest tests/agents/test_meta_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/meta/meta.py`
- `tests/agents/test_meta_agent.py` — update:
  - Test: `MetaAgent` has no `_tokens_used` attribute (assert `AttributeError`)
  - Test: no direct `create_provider()` call in `meta.py` (if LLM calls existed and were migrated)
  - Test: no direct `AgentHarness` instantiation in `meta.py`
  - Test: orchestration flow (spawn → await → collect) produces expected artifact structure

**Success criteria**: `MetaAgent` has no direct LLM or harness calls (where applicable). No
`_tokens_used`. Output paths use workspace accessors. Orchestration behavior is preserved.

---

## 6.7: Phase 6 Code Review and Changelog

**Tasks**:
- Grep `lsm/agents/productivity/writing.py` for `AgentHarness`, `sandbox.execute`,
  `provider.synthesize`, `provider.complete`, `_tokens_used` — all results must be empty
- Grep `lsm/agents/assistants/` for `_tokens_used` — all results must be empty
- Grep `lsm/agents/meta/meta.py` for `_tokens_used` — result must be empty
- Grep entire `lsm/agents/` for `query_embeddings` — result must be empty
- Review all `_bind_interaction_tools()` implementations: confirm `ask_user` binding works
  correctly after harness refactor; verify no fake harness stubs are broken
- Review workspace accessor usage: verify all agents use `_artifacts_dir()` and
  `_artifact_filename()` for file output; no hardcoded paths remain
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 6 changes
- Update `.agents/docs/architecture/development/AGENTS.md`: update all affected agent sections
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/AGENTS.md`

**Success criteria**: `pytest tests/ -v` passes. No direct harness, provider, or sandbox calls
remain in any agent file (outside of the harness itself and `base.py`). All documentation updated.
