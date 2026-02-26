# Phase 4: Simple Agent Migration

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1

Migrates `GeneralAgent`, `LibrarianAgent`, and `ManuscriptEditorAgent` from direct
`AgentHarness` instantiation to `self._run_phase()`. These agents have a single-phase pattern
(no custom orchestration loop) so migration is straightforward. Behavioral regression must be
confirmed by existing and new tests.

Reference: [RESEARCH_PLAN.md §2.9](../docs/RESEARCH_PLAN.md#29-scope-of-agent-migrations)

---

## 4.1: GeneralAgent Migration

**Description**: Remove direct `AgentHarness` instantiation from `GeneralAgent.run()`. Replace
with `self._run_phase()`. Apply workspace accessor changes.

**Tasks**:
- In `GeneralAgent.run()`:
  - Remove `harness = AgentHarness(...)` construction
  - Remove `harness.run(...)` call
  - Add `result = self._run_phase(system_prompt=..., user_message=task.objective, max_iterations=...)`
  - Consume `result.final_text` for any post-run logic (task result, logging)
  - Call `self._reset_harness()` (or rely on `run()` entry reset) at the top of `run()`
- Audit any file output in `GeneralAgent`: replace with `self._artifacts_dir()` and
  `self._artifact_filename()` if applicable
- Confirm `GeneralAgent` does NOT instantiate `AgentHarness` directly (remove import if unused)
- Run relevant tests: `pytest tests/agents/test_general_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/productivity/general.py`
- `tests/agents/test_general_agent.py` — update or extend:
  - Test: `run()` does NOT directly instantiate `AgentHarness` (spy on `AgentHarness.__init__`
    — it must not be called directly by `GeneralAgent`; it is called internally by `_run_phase`)
  - Test: `_run_phase()` is called with the expected `system_prompt` and `user_message`
  - Test: `PhaseResult.final_text` is correctly consumed by post-run logic
  - Regression: run `GeneralAgent` with a mocked `_run_phase()` and confirm behavior matches
    pre-migration expectations

**Success criteria**: `GeneralAgent.run()` uses `_run_phase()`. No direct `AgentHarness`
import or instantiation in `general.py`. All tests pass.

---

## 4.2: LibrarianAgent Migration

**Description**: Remove direct `AgentHarness` instantiation from `LibrarianAgent.run()`.
Replace with `self._run_phase()`. Apply workspace accessor changes.

**Tasks**:
- Same migration pattern as 4.1
- In `LibrarianAgent.run()`: replace harness construction and `run()` call with `_run_phase()`
- Audit file output paths — replace with `self._artifacts_dir()` / `self._artifact_filename()`
- Remove `AgentHarness` import from `librarian.py` if no longer needed
- Run relevant tests: `pytest tests/agents/test_librarian_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/librarian/librarian.py`
- `tests/agents/test_librarian_agent.py` — same test patterns as 4.1

**Success criteria**: Same as 4.1 for `LibrarianAgent`.

---

## 4.3: ManuscriptEditorAgent Migration

**Description**: Remove direct `AgentHarness` instantiation from `ManuscriptEditorAgent.run()`.
Replace with `self._run_phase()`. Apply workspace accessor changes.

**Tasks**:
- Same migration pattern as 4.1 and 4.2
- In `ManuscriptEditorAgent.run()`: replace harness construction and `run()` call with `_run_phase()`
- Audit file output paths — replace with workspace accessors
- Remove `AgentHarness` import from `manuscript_editor.py` if no longer needed
- Run relevant tests: `pytest tests/agents/test_manuscript_editor_agent.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/productivity/manuscript_editor.py`
- `tests/agents/test_manuscript_editor_agent.py` — same test patterns as 4.1

**Success criteria**: Same as 4.1 for `ManuscriptEditorAgent`.

---

## 4.4: Phase 4 Code Review and Changelog

**Tasks**:
- For each of the three agents: grep the source file for `AgentHarness` — result must be empty
- For each of the three agents: grep for `sandbox.execute()` and `provider.synthesize()` —
  results must be empty
- Review tests: no mocks or stubs; `_run_phase()` is mocked via the standard mock-at-harness
  pattern (patching `BaseAgent._run_phase`)
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 4 changes
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`

**Success criteria**: `pytest tests/ -v` passes. No direct harness, sandbox, or provider calls
remain in the three agent files.
