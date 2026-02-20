# Phase 10: Meta-Agent Evolution

**Why tenth:** Parallel execution builds on the existing serial meta-agent (already in `lsm/agents/meta.py`) and requires the full agent catalog (Phase 6) and communication assistants (Phase 9) to be available for orchestration.

**Depends on:** Phase 6 (core agents), Phase 9 (communication assistants)

| Task | Description | Depends On |
|------|-------------|------------|
| 10.1 | Parallel task graph planning | None |
| 10.2 | Parallel execution engine | 10.1 |
| 10.3 | General Meta-Agent | 10.2 |
| 10.4 | Assistant Meta-Agent | 10.2, 6.4 |
| 10.5 | Tests and documentation | 10.1–10.4 |

## 10.1: Parallel Task Graph Planning
- **Description:** Extend task graph planning to represent parallelizable work.
- **Tasks:**
  - Extend existing `task_graph.py` to support `parallel_group` nodes. A parallel group contains tasks that can execute concurrently. Dependency gates between groups enforce ordering.
  - Add deterministic ordering for parallel execution plans (within parallel groups for reproducible results).
  - Document graph serialization for meta-agent prompts.
- **Files:**
  - `lsm/agents/meta/task_graph.py`
  - `lsm/agents/meta/meta.py`
- **Success criteria:** Meta plans include explicit parallel groups with deterministic ordering.

## 10.2: Parallel Execution Engine
- **Description:** Execute sub-agents concurrently with resource limits and sandbox guarantees.
- **Tasks:**
  - Add concurrency controls using `asyncio` or `concurrent.futures` for parallel sub-agent execution.
  - Enforce sandbox monotonicity: sub-agent sandboxes must be subsets of the parent meta-agent sandbox.
  - Resource limits: respect `max_concurrent` from `AgentConfig`.
  - Collect artifacts concurrently and merge results deterministically (sorted by agent name, then task order).
- **Files:**
  - `lsm/agents/meta/meta.py`
  - `lsm/agents/tools/spawn_agent.py`
  - `lsm/agents/tools/await_agent.py`
  - `lsm/agents/tools/collect_artifacts.py`
- **Success criteria:** Parallel runs complete safely with predictable artifact aggregation.

## 10.3: General Meta-Agent
- **Description:** Provide a general meta-agent that orchestrates multiple agent types.
- **Tasks:**
  - Define meta-agent prompt and tool allowlist.
  - Implement sub-agent selection logic based on task types.
  - Emit final consolidated artifact (`final_result.md`).
- **Files:**
  - `lsm/agents/meta/meta.py`
  - `lsm/agents/factory.py`
- **Success criteria:** General meta-agent can coordinate multi-agent workflows end-to-end.

## 10.4: Assistant Meta-Agent
- **Description:** Provide a meta-agent dedicated to assistant-style summaries and compliance checks.
- **Tasks:**
  - Define assistant-meta prompt for reviews, security checks, and memory proposals.
  - Add optional validation passes over sub-agent outputs.
  - Emit summary artifacts and action recommendations.
- **Files:**
  - `lsm/agents/meta/meta.py`
  - `lsm/agents/assistants/assistant.py`
- **Success criteria:** Assistant meta-agent produces structured reviews and action items.

## 10.5: Tests and Documentation
- **Description:** Validate parallel meta-agent behavior and sandbox enforcement.
- **Tasks:**
  - Add tests for parallel plan execution and sandbox monotonicity.
  - Document meta-agent configuration and usage.
- **Files:**
  - `tests/test_agents_meta/`
  - `docs/`
- **Success criteria:** Meta-agent parallel execution is tested and documented.
