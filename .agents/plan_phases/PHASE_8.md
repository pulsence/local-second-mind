# Phase 8: Final Code Review and Release

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1

A final comprehensive review of the entire v0.7.1 implementation. Covers test suite cleanup,
user-facing documentation, a full cross-cutting code review, and the release commit.

---

## 8.1: Test Suite Cleanup

**Description**: Review the full test suite for tests that are now redundant or wrong given
the new architecture. Since LLM calls, tool dispatch, and file access are now centralized in
`BaseAgent._run_phase()` and `AgentHarness`, per-agent tests for these concerns may be
duplicative. However, each agent must still be tested to confirm it does NOT directly make
LLM or tool calls — the absence of direct calls is itself a critical property.

**Tasks**:
- Identify tests that test LLM behavior, tool execution, or file access at the per-agent
  level but are now fully covered by `BaseAgent` and `AgentHarness` tests — remove duplicates
- Confirm every agent file has at least one test asserting:
  - No direct `AgentHarness` instantiation (grep or import-level check)
  - No direct `provider.synthesize()` / `provider.complete()` call
  - No direct `sandbox.execute()` call (for LLM-driven agents)
  - No `_tokens_used` attribute
- Verify `query_embeddings` absence tests remain in place
- Verify `query_remote` (generic single-tool) absence tests remain in place
- Remove any test files whose entire purpose was to test a now-deleted method or class
  (e.g., tests for `_collect_findings()`, `_summarize_findings()`, `_select_tools()`)
- Run full test suite after cleanup: `pytest tests/ -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- Any test files in `tests/agents/` that contain now-redundant per-agent LLM/tool/file tests

**Success criteria**: `pytest tests/ -v` passes with the pruned test suite. Every agent has
absence-of-direct-calls tests in place.

---

## 8.2: docs/user-guide/AGENTS.md Update

**Description**: Update `docs/user-guide/AGENTS.md` so that each agent description specifies
the tools available to it by default, based on its `tool_allowlist` and `remote_source_allowlist`.

**Tasks**:
- For each agent, document:
  - Agent name, tier, and description
  - Default `tool_allowlist` (tools available to the LLM)
  - Default `remote_source_allowlist` (remote sources available, if applicable)
  - Brief workflow description
- Ensure the tool descriptions in the user guide match the actual tool names after Phase 3
  (`query_knowledge_base` instead of `query_embeddings`) and Phase 7 (per-source
  `query_<name>` instead of `query_remote`)
- Run full test suite: `pytest tests/ -v` (to confirm no doc-only changes broke anything)
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/user-guide/AGENTS.md`

**Success criteria**: Every agent in `docs/user-guide/AGENTS.md` has an accurate tool access
section reflecting the final v0.7.1 state.

---

## 8.3: Final Complete Code Review

**Description**: A cross-cutting final review of the entire v0.7.1 implementation — all phases,
all agents, all tools, all tests.

**Tasks**:
- Grep entire `lsm/agents/` tree for the following — all results must be empty outside of
  `harness.py` and `base.py`:
  - `AgentHarness(`
  - `sandbox.execute`
  - `provider.synthesize`
  - `provider.complete`
  - `_tokens_used`
  - `create_provider(`
- Grep entire project for `query_embeddings` — result must be empty
- Grep entire project for `"query_remote"` (string literal) — result must be empty outside
  tests that assert its absence
- Review `PhaseResult`: confirm no accounting fields have been added; confirm all `stop_reason`
  values are consistent across all agents
- Review `BaseAgent`: confirm security invariants hold — no `_check_budget_and_stop`, no
  `_tokens_used`, no provider imports
- Review `AgentHarness.run_bounded()`: confirm budget check precedes LLM call in all paths;
  confirm context isolation between labels; confirm global token tracking
- Review workspace accessor usage: grep `lsm/agents/` for hardcoded path construction
  patterns (e.g., `agents_folder / f"`) — all results must be empty outside of tests
- Review `InteractionChannel` two-phase timeout: confirm acknowledged requests wait indefinitely
  when `acknowledged_timeout_seconds=0`
- Review `_bind_interaction_tools()` in `CalendarAssistantAgent` and `EmailAssistantAgent`:
  confirm interaction channel binding is functional post-harness refactor
- Run full test suite with coverage: `pytest tests/ -v --cov=lsm --cov-report=html`
- Review coverage report: all new methods and branches in `harness.py`, `base.py`,
  `interaction.py`, and agent files must have meaningful coverage
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**: (review only — no changes unless a defect is found)

**Success criteria**: All grep checks pass. Full test suite passes with coverage. No defects
found requiring post-review fixes.

---

## 8.4: Final Documentation and Release

**Tasks**:
- Bump version to `0.7.1` in **all** of the following files — they must all match:
  - `pyproject.toml`: `version = "0.7.0"` → `"0.7.1"`
  - `lsm/__init__.py`: `__version__ = "0.7.0"` → `"0.7.1"` (the TUI help screen derives its
    version heading from this value automatically via `_WHATS_NEW_VERSION = f"v{__version__}"`)
  - `README.md`: version badge (`` `0.6.0` `` → `` `0.7.1` ``) and section heading
    `## What Is New in 0.6.0` → `## What Is New in 0.7.1`; update the "What Is New" body
    to summarise v0.7.1 changes
  - `lsm/ui/tui/screens/help.py`: replace the `_WHATS_NEW` tuple body (currently lines 144–151,
    which describes v0.7.0 features) with bullet lines summarising v0.7.1 changes; the version
    heading (`_WHATS_NEW_VERSION`) is dynamic and updates automatically from `__version__`
- Add a new `## 0.7.1 - <date>` entry to `docs/CHANGELOG.md` with a consolidated release
  summary covering all phases; do **not** modify earlier entries
- Update `.agents/docs/architecture/api-reference/CONFIG.md`: document
  `acknowledged_timeout_seconds` in the `agents.interaction` section
- Update `example_config.json`: verify `acknowledged_timeout_seconds` is present under
  `agents.interaction`
- Final review of `.agents/docs/architecture/development/AGENTS.md`: verify all sections
  reflect v0.7.1 state — `BaseAgent` section, interaction section, all agent workflow sections,
  `context_label` usage, workspace accessor methods
- Final review of `.agents/docs/architecture/packages/lsm.agents.md`: verify `PhaseResult`,
  `_run_phase()`, `run_bounded()`, `query_knowledge_base`, and per-source `QueryRemoteTool`
  are all documented
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `pyproject.toml`
- `lsm/__init__.py`
- `lsm/ui/tui/screens/help.py`
- `README.md`
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/api-reference/CONFIG.md`
- `.agents/docs/architecture/development/AGENTS.md`
- `.agents/docs/architecture/packages/lsm.agents.md`
- `example_config.json`

**Success criteria**: `pytest tests/ -v` passes. All documentation is accurate and complete.
`pyproject.toml`, `lsm/__init__.py`, `README.md`, and the TUI help screen all reflect `0.7.1`.
v0.7.1 changelog entry covers all phases. Release is committed and pushed.
