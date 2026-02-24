# Local Second Mind v0.7.0 Development Plan: Agents and Remote Sources

## Overview

v0.7.0 introduces a comprehensive agent system with multiple specialized agents, advanced file tooling, execution environments, and remote source providers. This plan organizes work into 11 phased releases for incremental delivery with proper dependency ordering.

---

## Validation Notes (Architecture Alignment)

- Agents are already implemented under `lsm/agents/` with built-in agents (`curator`, `meta`, `research`, `writing`, `synthesis`) and a registry in `lsm/agents/factory.py`; new agents should extend this framework rather than replace it.
- Tooling and runners live in `lsm/agents/tools/` (`docker_runner.py`, `runner.py`, `ask_user.py`, `source_map.py`, etc.); new file tools and runners should be added here and wired through the ToolRegistry.
- Remote providers already include `openalex`, `crossref`, `core`, and `oai_pmh` under `lsm/remote/providers/`; v0.7.0 should extend these and add missing providers instead of duplicating them.
- Remote provider interface and chain behavior are defined in `.agents/docs/architecture/api-reference/REMOTE.md`; new providers must conform to `BaseRemoteProvider` and chain output schemas.

---

## Standard Phase Completion Checklist

*Applies to every phase — not repeated as individual tasks:*

- Review that no backwards compatibility and dead code remains
- Ensure test coverage for all new code (TDD: tests first, then implementation)
- Ensure the documentation in docs/ and .agents/docs/ are properly updated for changes.
- Update `docs/CHANGELOG.md` with phase deliverables

---

## Phases

### [Phase 1: Core Infrastructure & Configuration](../plan_phases/PHASE_1.md) (COMPLETED)

Utils module, agent log format conversion, tiered model configuration, and agent/tool test harness.

### [Phase 2: File Graphing System](../plan_phases/PHASE_2.md) (COMPLETED)

Graph schema, code file grapher, text document grapher, PDF document grapher, HTML document grapher, tool integration hooks, and fixtures.

### [Phase 3: Agent Framework Overhaul](../plan_phases/PHASE_3.md) (COMPLETED)

Agent package restructure, workspace defaults, tool API standardization, and universal ask_user.

### [Phase 4: Advanced Tooling](../plan_phases/PHASE_4.md) (COMPLETED)

Line-hash editing engine, find file/section tools, read/outline enhancements, and benchmark comparisons.

### [Phase 5: Execution Environment](../plan_phases/PHASE_5.md)

Runner policy updates, Docker runner completion, WSL2 runner, bash/PowerShell tools with security testing.

### [Phase 6: Core Agent Catalog](../plan_phases/PHASE_6.md)

General, Librarian, Assistant, and Manuscript Editor agents.

### [Phase 7: AI Providers & Protocol Infrastructure](../plan_phases/PHASE_7.md)

OpenRouter provider, OAI-protocol consolidation, MCP host support, and generic RSS reader.

### [Phase 8: Remote Source Providers](../plan_phases/PHASE_8.md)

Provider sub-package restructure, structured output validation, scholarly/academic/cultural/news providers, and specialized protocols.

### [Phase 9: Communication Platform](../plan_phases/PHASE_9.md)

OAuth2 infrastructure, email/calendar providers, and email/calendar/news assistant agents.

### [Phase 10: Meta-Agent Evolution](../plan_phases/PHASE_10.md)

Parallel task graph planning, parallel execution engine, general and assistant meta-agents.

### [Phase 11: Release & Documentation](../plan_phases/PHASE_11.md)

Version bump, documentation audit, config examples update, and TUI WHATS_NEW.

---

## Cross-Phase Dependency Summary

```
Phase 1  ──>  Phase 2  ──>  Phase 4  ──>  Phase 6  ──>  Phase 10  ──>  Phase 11
  │    │         │                            │               ↑
  │    │         └────────────────────────────┘               │
  │    │                                                      │
  │    └──>  Phase 3  ──────────────────>  Phase 6            │
  │                                        │                  │
  │         Phase 5 (parallel w/ Phase 4)  └──>  Phase 9  ──>  Phase 10
  │                                                  ↑
  └──(1.4)──>  Phase 4  (benchmarks)       Phase 7  ──>  Phase 8
```

**Key gates:**

- Phase 1.4 (test harness) must finish before Phase 4 (benchmark comparisons)
- Phase 2 (graphing) must finish before Phase 4 (tooling that uses graphs)
- Phase 3 (restructure) must finish before Phase 6 (new agents placed in new packages)
- Phase 7 (OAI consolidation, RSS) must finish before Phase 8 (providers that use them)
- Phase 8.1 (structured output validation) must finish before all provider implementations
- Phase 9 (communication platform) needs both Phase 8 (provider patterns) and Phase 6 (agent patterns)
- Phase 10 (meta-agents) needs Phase 6 + Phase 9 (agents to orchestrate)

**Parallelizable work:**

- Phases 2 and 3 can run in parallel (no mutual dependencies)
- Phase 5 can run in parallel with Phase 4 (independent concerns)
- Phase 7 can start as soon as Phase 1.3 is done, in parallel with Phases 4-6
