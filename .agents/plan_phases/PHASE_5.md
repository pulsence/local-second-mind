# Phase 5: Academic Agents Migration

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1

Migrates `ResearchAgent`, `SynthesisAgent`, and `CuratorAgent` from manual execution
patterns to `_run_phase()`. This is the most substantial migration phase. `ResearchAgent`
also gains: `query_knowledge_base` integration (replacing `query_embeddings` + manual
synthesis), subtopic logging improvements (§4 of research plan), and artifact path corrections
(§2.7). `context_label` is used for per-subtopic collection contexts.

Reference: [RESEARCH_PLAN.md §2.9](../docs/RESEARCH_PLAN.md#29-scope-of-agent-migrations), [§3.4](../docs/RESEARCH_PLAN.md#34-chosen-design), [§4](../docs/RESEARCH_PLAN.md#4-research-agent-observability)

---

## 5.1: ResearchAgent Refactor

**Description**: Full refactor of `ResearchAgent` to use `_run_phase()`. Removes all direct
LLM and sandbox calls. Integrates `query_knowledge_base` for the collect phase. Adds
`context_label` per subtopic. Improves subtopic logging. Fixes artifact output path.

**Tasks**:

**Phase orchestration** — rewrite `run()`:
- Call `self._reset_harness()` at the top of `run()`
- Phase 1 — DECOMPOSE: `self._run_phase(system_prompt=RESEARCH_SYSTEM_PROMPT, user_message=f"Phase: DECOMPOSE. Topic: '{self.topic}'...", tool_names=[], max_iterations=1)`
- Parse subtopics from `result.final_text` using `self._parse_subtopics()`
- Log subtopics with names: `subtopic_lines = "\n".join(f"  [{i}] {st}" for i, st in enumerate(subtopics, 1)); self._log(f"Research iteration {iteration} — {len(subtopics)} subtopics:\n{subtopic_lines}")`
- Phase 2 — RESEARCH (per subtopic):
  - For each subtopic: `self._log(f"Collecting findings for subtopic: {subtopic}")`
  - Call `self._run_phase(..., tool_names=["query_knowledge_base"], max_iterations=3, context_label=f"subtopic:{subtopic}")`
  - Break loop if `result.stop_reason in ("budget_exhausted", "stop_requested")`
- Phase 3 — SYNTHESIZE: `self._run_phase(..., tool_names=[], max_iterations=1, context_label=None)`
- Phase 4 — REVIEW (if applicable): `self._run_phase(...)` in primary context

**Review suggestion logging**:
- Replace `self._log(f"Refining with {len(subtopics)} review suggestions.")` with:
  `suggestion_lines = "\n".join(f"  [{i}] {s}" for i, s in enumerate(suggestions, 1)); self._log(f"Refining with {len(suggestions)} review suggestions:\n{suggestion_lines}")`

**Artifact output**:
- Replace `output_path = self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}.md"` with:
  `output_path = self._artifacts_dir() / self._artifact_filename(safe_topic)`

**Removed methods** (delete entirely):
- `_select_tools()`
- `_collect_findings()`
- `_summarize_findings()`
- `_build_sources_block()`
- `_extract_sources_from_output()`
- Remove `_tokens_used` counter and all manual token tracking

**Remove**:
- `query_embeddings` from `tool_allowlist`
- Direct `provider.synthesize()` calls
- Direct `sandbox.execute()` calls
- `AgentHarness` import and instantiation

- Run relevant tests: `pytest tests/agents/academic/test_research_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/academic/research.py` — major rewrite
- `tests/agents/academic/test_research_agent.py` — update:
  - Test: phases execute in order DECOMPOSE → RESEARCH (per subtopic) → SYNTHESIZE → REVIEW
  - Test: `_run_phase()` called with `context_label=f"subtopic:{subtopic}"` for each RESEARCH phase
  - Test: `_run_phase()` called with `context_label=None` for SYNTHESIZE phase
  - Test: per-subtopic log entries contain subtopic name strings
  - Test: iteration log lists all subtopic names
  - Test: when `stop_reason="budget_exhausted"`, subtopic loop terminates and SYNTHESIZE still runs
  - Test: output file is written to `self._artifacts_dir()` with `_artifact_filename()` format
  - Test: `ResearchAgent` has no direct `sandbox.execute()` call
  - Test: `ResearchAgent` has no direct `provider.synthesize()` call
  - Test: `AgentHarness` is not directly instantiated in `research.py`
  - Update any `ResearchResult.output_path` assertions to the new `artifacts/` location

**Success criteria**: `ResearchAgent` uses `_run_phase()` exclusively. All deleted methods are
gone. `context_label` is used per subtopic. Subtopic names appear in logs. Artifact is in
`artifacts/` subdirectory. All tests pass.

---

## 5.2: SynthesisAgent Refactor

**Description**: Audit `SynthesisAgent` and replace all direct LLM and sandbox calls with
`_run_phase()`. Apply workspace accessor changes for any file output.

**Tasks**:
- Map `SynthesisAgent.run()` phases to `_run_phase()` calls
- For any multi-step workflow: use `context_label` if phases are conceptually independent;
  use the primary context (no label) for sequential phases that need prior context
- Remove direct `provider.synthesize()`, `provider.complete()`, or `sandbox.execute()` calls
- Remove `AgentHarness` direct instantiation
- Replace any hardcoded file output paths with `self._artifacts_dir()` / `self._artifact_filename()`
- Remove `_tokens_used` counter and manual token tracking
- Run relevant tests: `pytest tests/agents/academic/test_synthesis_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/academic/synthesis.py`
- `tests/agents/academic/test_synthesis_agent.py` — update:
  - Test: no direct `sandbox.execute()` or `provider.*` calls
  - Test: `AgentHarness` not directly instantiated
  - Test: file output goes to `_artifacts_dir()` if applicable

**Success criteria**: `SynthesisAgent` uses `_run_phase()` exclusively. All tests pass.

---

## 5.3: CuratorAgent Refactor

**Description**: Audit `CuratorAgent` and replace all direct LLM and sandbox calls with
`_run_phase()`. Apply workspace accessor changes for any file output.

**Tasks**:
- Same pattern as 5.2 for `CuratorAgent`
- Map each phase or LLM call in `run()` to a `_run_phase()` call
- Use `context_label` where independent contexts are beneficial
- Remove direct calls, manual token tracking, and hardcoded paths
- Run relevant tests: `pytest tests/agents/academic/test_curator_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/academic/curator.py`
- `tests/agents/academic/test_curator_agent.py` — update with same patterns as 5.2

**Success criteria**: `CuratorAgent` uses `_run_phase()` exclusively. All tests pass.

---

## 5.4: Phase 5 Code Review and Changelog

**Tasks**:
- Grep `lsm/agents/academic/` for `AgentHarness`, `sandbox.execute`, `provider.synthesize`,
  `provider.complete`, `_tokens_used` — all results must be empty
- Review `context_label` usage in `ResearchAgent`: confirm each subtopic gets its own label;
  confirm SYNTHESIZE uses primary context (no label)
- Review subtopic and suggestion log format: confirm names appear, not just counts
- Review artifact path: confirm `ResearchResult.output_path` reflects the new location
- Review tests: no stubs or mocks of internal methods; only `_run_phase()` and harness-level
  mocking permitted
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 5 changes
- Update `.agents/docs/architecture/development/AGENTS.md`: update Research Agent, Synthesis
  Agent, and Curator Agent workflow sections
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/AGENTS.md`

**Success criteria**: `pytest tests/ -v` passes. No manual execution patterns remain in the
three academic agents.
