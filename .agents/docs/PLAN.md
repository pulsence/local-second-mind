# v0.7.1 Implementation Plan: Agent Foundation Refinement

**Version**: 0.7.1
**Research Plan**: [RESEARCH_PLAN.md](./RESEARCH_PLAN.md)

---

## Phases

| Phase | Title | File | Status |
|-------|-------|------|--------|
| 1 | Interaction Channel — Two-Phase Timeout | [PHASE_1.md](../plan_phases/PHASE_1.md) | Complete |
| 2 | AgentHarness and BaseAgent Infrastructure | [PHASE_2.md](../plan_phases/PHASE_2.md) | Complete |
| 3 | query_knowledge_base Tool | [PHASE_3.md](../plan_phases/PHASE_3.md) | Complete |
| 4 | Simple Agent Migration | [PHASE_4.md](../plan_phases/PHASE_4.md) | Complete |
| 5 | Academic Agents Migration | [PHASE_5.md](../plan_phases/PHASE_5.md) | |
| 6 | Remaining Agent Migration | [PHASE_6.md](../plan_phases/PHASE_6.md) | |
| 7 | query_remote Tool Redesign | [PHASE_7.md](../plan_phases/PHASE_7.md) | |
| 8 | Final Code Review and Release | [PHASE_8.md](../plan_phases/PHASE_8.md) | |

---

## Phase 1: Interaction Channel — Two-Phase Timeout

Adds an acknowledgment-based two-phase timeout to the agent interaction channel. Once an
interaction is displayed to the user in the TUI, it is acknowledged automatically; once
acknowledged, the channel waits indefinitely. Independent of all other phases.

→ [Full details: PHASE_1.md](../plan_phases/PHASE_1.md)

---

## Phase 2: AgentHarness and BaseAgent Infrastructure

Adds foundational infrastructure all subsequent phases depend on: `PhaseResult`,
`AgentHarness.run_bounded()` with multi-context support, `BaseAgent._run_phase()` with
`context_label`, and `BaseAgent` workspace accessor methods. No agents are migrated here.

→ [Full details: PHASE_2.md](../plan_phases/PHASE_2.md)

---

## Phase 3: query_knowledge_base Tool

Implements the `query_knowledge_base` tool that wraps `query_sync()` and calls the full
query pipeline. Removes and deletes `query_embeddings`.

→ [Full details: PHASE_3.md](../plan_phases/PHASE_3.md)

---

## Phase 4: Simple Agent Migration

Migrates `GeneralAgent`, `LibrarianAgent`, and `ManuscriptEditorAgent` from direct
`AgentHarness` instantiation to `self._run_phase()`. Single-phase agents with no custom
orchestration loop — straightforward migration.

→ [Full details: PHASE_4.md](../plan_phases/PHASE_4.md)

---

## Phase 5: Academic Agents Migration

Migrates `ResearchAgent`, `SynthesisAgent`, and `CuratorAgent`. `ResearchAgent` gains
`query_knowledge_base` integration, per-subtopic `context_label` usage, improved subtopic
logging, and corrected artifact output paths. Most substantial migration phase.

→ [Full details: PHASE_5.md](../plan_phases/PHASE_5.md)

---

## Phase 6: Remaining Agent Migration

Audits and refactors all remaining agents: `WritingAgent` (LLM-driven, full `_run_phase()`
migration), `AssistantAgent`, `NewsAssistantAgent`, `CalendarAssistantAgent`,
`EmailAssistantAgent` (data-pipeline agents, workspace accessor + token cleanup + tool-only
`_run_phase()` for tool dispatch), and `MetaAgent` (orchestration audit).

→ [Full details: PHASE_6.md](../plan_phases/PHASE_6.md)

---

## Phase 7: query_remote Tool Redesign

Redesigns `QueryRemoteTool` from a single generic tool to a factory-based pattern where each
configured remote source gets its own named tool instance (`query_<name>`). Adds
`remote_source_allowlist` to `BaseAgent`. Removes `_resolve_lsm_config()` registry hack
from news/calendar/email assistant agents.

→ [Full details: PHASE_7.md](../plan_phases/PHASE_7.md)

---

## Phase 8: Final Code Review and Release

Final comprehensive review: test suite cleanup, user-guide documentation update, full
cross-cutting code review with coverage, and the v0.7.1 release commit (version bump in all
files, changelog, TUI "What's New" content).

→ [Full details: PHASE_8.md](../plan_phases/PHASE_8.md)
