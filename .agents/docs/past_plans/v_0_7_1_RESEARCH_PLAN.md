# v0.7.1 Research Plan: Agent Foundation Refinement

**Status**: Clarifications Resolved — Design Finalized
**Version Target**: 0.7.1
**Source**: `TODO` v0.7.1 section

This document records all design findings for v0.7.1. All clarifying questions have been
resolved. The next step is to produce an implementation plan based on these decisions.

---

## Table of Contents

1. [Release Overview](#1-release-overview)
2. [Core Architecture: Persistent Harness and Phase Execution](#2-core-architecture-persistent-harness-and-phase-execution)
   - 2.1 [The Problem: Two Divergent Execution Patterns](#21-the-problem-two-divergent-execution-patterns)
   - 2.2 [What Manual Execution Bypasses](#22-what-manual-execution-bypasses)
   - 2.3 [BaseAgent vs AgentHarness: Defined Responsibilities](#23-baseagent-vs-agentharness-defined-responsibilities)
   - 2.4 [Solution: Persistent Harness with _run_phase()](#24-solution-persistent-harness-with-_run_phase)
   - 2.5 [AgentHarness: run_bounded() and Budget Enforcement](#25-agentharness-run_bounded-and-budget-enforcement)
   - 2.6 [PhaseResult: The Phase Execution Contract](#26-phaseresult-the-phase-execution-contract)
   - 2.7 [Workspace and Artifact Management](#27-workspace-and-artifact-management)
   - 2.8 [Design Options Considered](#28-design-options-considered)
   - 2.9 [Scope of Agent Migrations](#29-scope-of-agent-migrations)
3. [Knowledge Base Access: query_knowledge_base Tool](#3-knowledge-base-access-query_knowledge_base-tool)
   - 3.1 [Problem: Research Agent Reimplements the Query Pipeline](#31-problem-research-agent-reimplements-the-query-pipeline)
   - 3.2 [What the Query Pipeline Provides](#32-what-the-query-pipeline-provides)
   - 3.3 [The Dependency Gap](#33-the-dependency-gap)
   - 3.4 [Chosen Design](#34-chosen-design)
   - 3.5 [query_embeddings Deprecation](#35-query_embeddings-deprecation)
   - 3.6 [Design Options Considered](#36-design-options-considered)
4. [Research Agent Observability](#4-research-agent-observability)
5. [Interaction Channel: Two-Phase Timeout](#5-interaction-channel-two-phase-timeout)
   - 5.1 [Problem Statement](#51-problem-statement)
   - 5.2 [Current Architecture](#52-current-architecture)
   - 5.3 [Root Cause](#53-root-cause)
   - 5.4 [Design Options](#54-design-options)
   - 5.5 [Chosen Design](#55-chosen-design)
6. [Implementation Plan](#6-implementation-plan)
   - 6.1 [Dependency Map](#61-dependency-map)
   - 6.2 [Implementation Order](#62-implementation-order)
   - 6.3 [Testing Strategy](#63-testing-strategy)
   - 6.4 [Documentation Updates](#64-documentation-updates)
7. [Resolved Decisions](#7-resolved-decisions)

---

## 1. Release Overview

v0.7.1 is an agent refinement release addressing behavioral and architectural problems found after
v0.7.0. Five areas of work are addressed in a single implementation pass.

| # | Area | Summary |
|---|------|---------|
| 1 | Architecture | `BaseAgent` should own all LLM and tool calls via a persistent harness; individual agents only orchestrate |
| 2 | Research Agent | Eliminate query pipeline duplication; route research queries through `query_knowledge_base` |
| 3 | Research Agent | Log each subtopic name during iterations, not just the count |
| 4 | Research Agent | Save output artifacts to the correct per-agent artifacts folder |
| 5 | General | Interaction channel should not time out when the user is actively viewing the agent prompt |

Items are ordered here by dependency. Item 1 (architecture) is the foundation on which Items 2, 3,
and 4 are built. Items 3 and 4 are absorbed into Item 1's agent migration work. Item 5 (interaction
timeout) is independent of the others.

---

## 2. Core Architecture: Persistent Harness and Phase Execution

### 2.1 The Problem: Two Divergent Execution Patterns

Looking across all agents, there are two distinct execution patterns:

**Pattern A — Harness-Delegating (correct pattern)**

Agents: `GeneralAgent`, `LibrarianAgent`, `ManuscriptEditorAgent`

These create an `AgentHarness` in `run()` and delegate all LLM calls and tool executions to it.
The harness handles: LLM resolution, tool selection, sandbox enforcement, state management,
memory injection, budget tracking, logging, and run summary generation. The agent's role is
purely configuration: tools, system prompt, LLM tier, and max iterations.

**Pattern B — Manual Execution (problematic)**

Agents: `ResearchAgent`, `SynthesisAgent`, `CuratorAgent`

These manage their own execution loops and make direct calls:
- `create_provider(self._resolve_llm_config(self.llm_registry))` → direct provider instantiation
- `provider.synthesize(prompt, context, mode=...)` → direct LLM invocation
- `self.sandbox.execute(tool, args)` → direct sandbox call

`WritingAgent` is a mixed case: it has both a custom outer phase loop and some harness delegation.
`AssistantAgent` uses its own pattern as well.

### 2.2 What Manual Execution Bypasses

When a manual agent calls `self.sandbox.execute(tool, args)` directly:
- ✅ Sandbox permission checks still run
- ❌ No `_is_tool_allowed()` check against the harness allowlist
- ❌ No per-run tool usage counts in `run_summary.json`
- ❌ No artifact auto-tracking via `_track_artifacts_from_sandbox()`
- ❌ No context compaction (`context_window_strategy`)

When a manual agent calls `provider.synthesize()` directly:
- ❌ No memory standing context injected before the LLM call
- ❌ No conversation history management
- ❌ No LLM server-side caching support
- ❌ No proper cost tracking (token estimates are manual with a 4-char heuristic)
- ❌ No structured per-run summary
- ❌ Agents manage their own `_tokens_used` counter independently

### 2.3 BaseAgent vs AgentHarness: Defined Responsibilities

**BaseAgent** (abstract class, [lsm/agents/base.py](../lsm/agents/base.py)):

`BaseAgent` is the **identity and capability specification** for an agent. It defines:

- What the agent IS: `name`, `description`, `agent_config`
- What the agent is ALLOWED to do: `tool_allowlist`, `_always_available_tools`
- Utility methods: `_log()`, `_resolve_llm_config()`, `_get_tool_definitions()`
- The abstract lifecycle method: `run()` — which subclasses must implement
- Workspace accessor methods (see §2.7)

`BaseAgent` does NOT execute anything. It has no LLM call logic, no tool dispatch, no execution
loop. It is the agent's "capability card."

**AgentHarness** (runtime engine, [lsm/agents/harness.py](../lsm/agents/harness.py)):

`AgentHarness` is the **basic LLM + tool execution loop** — the engine that drives the most
fundamental agent behavior:

1. Call the LLM with the current prompt and conversation history
2. Parse the LLM's tool requests
3. Validate each requested tool against the agent's allowlist
4. Execute permitted tools via the sandbox
5. Feed tool results back to the LLM
6. Repeat until the LLM signals completion, budget is exhausted, or stop is requested

Beyond the basic loop, `AgentHarness` also handles: standing memory context injection before each
LLM call, context compaction when windows fill, budget enforcement, run summary writing, and
artifact tracking.

`AgentHarness` IS what the user described as "the most basic agent": pass a prompt and tools,
execute those tools, and feed results back to the LLM.

**The Gap**

`AgentHarness` is designed as an **all-or-nothing executor**: it runs until the entire task is
complete. It cannot return intermediate results to a Python orchestration loop.

This is fine for simple agents (GeneralAgent): give it a task, it runs to completion, done.

But multi-phase agents (ResearchAgent: decompose → collect per subtopic → synthesize → review)
need to inspect intermediate results in Python and make decisions between phases. Because
`AgentHarness` cannot do this, these agents fall back to calling `sandbox.execute()` and
`provider.synthesize()` directly — bypassing all security and tracking.

### 2.4 Solution: Persistent Harness with _run_phase()

`BaseAgent` owns a single `AgentHarness` for the entire duration of `run()` and exposes
`_run_phase()` as the only way to invoke execution. Agents never instantiate `AgentHarness`
directly or call `sandbox`/`provider` directly.

**The key principles**:
- `BaseAgent` maintains ONE `AgentHarness` per `run()` lifecycle — created on first use, reused
  for every subsequent phase
- `BaseAgent._run_phase()` is the ONLY way agents invoke execution — it is the single control point
- Token counts, cost, and budget are internal to the harness — agents NEVER see raw numbers
- No agent may call `sandbox.execute()`, `provider.synthesize()`, or instantiate `AgentHarness`
- Budget checking, budget enforcement, and stop decisions are entirely internal to the harness;
  no budget-query method exists on `BaseAgent`

**Why a persistent harness (not a fresh harness per phase)**:

Creating a new `AgentHarness` per `_run_phase()` call has two significant problems:

1. **Lost context**: Each new harness starts with an empty conversation history. Later phases
   cannot build on earlier phases naturally — the LLM has no memory of the decomposition step
   when it reaches synthesis.

2. **Token leakage to agent code**: If `PhaseResult` carries `tokens_used` and `cost_usd`, agent
   code can read and act on financial data. Agents should not manage or inspect cost figures —
   the harness is the correct owner of this data. Keeping it internal prevents agents from
   making decisions based on cost exposure.

A persistent harness solves both: the conversation thread flows continuously across phases (the
LLM has full prior context when synthesizing), and all accounting stays inside the harness.

**`_run_phase()` signature**:

```python
def _run_phase(
    self,
    system_prompt: str,
    user_message: str,
    tool_names: Optional[list[str]] = None,  # subset of tool_allowlist; None = use all allowed
    max_iterations: int = 10,
    continue_context: bool = True,           # True: extend conversation; False: reset history
) -> PhaseResult:
    """
    Execute one bounded phase of LLM + tool interaction on the shared harness.

    On first call, initializes the shared harness with the given system_prompt.
    Subsequent calls reuse the same harness, preserving conversation context
    when continue_context=True. All tool execution, LLM calls, token tracking,
    and budget enforcement are internal to the harness.

    This is the ONLY method agents may use for LLM and tool activity.
    """
```

**How simple agents use this** (GeneralAgent pattern):

```python
# In GeneralAgent.run() — replaces current harness creation + harness.run():
result = self._run_phase(
    system_prompt=self._build_system_prompt(),
    user_message=task.objective,
    max_iterations=self._max_iterations,
)
# One phase that runs to completion. Budget handled internally by the harness.
```

Simple agents no longer instantiate `AgentHarness` directly. `BaseAgent._run_phase()` handles it.

**How orchestrating agents use this** (ResearchAgent pattern):

The system prompt is set once for the full run; phase directives arrive as structured user
messages. The persistent harness accumulates the full conversation, so the LLM has complete
context in later phases. Agents react to `PhaseResult.stop_reason` to decide loop control;
they never query budget state directly:

```python
# In ResearchAgent.run():

# Phase 1: Decompose (initializes the shared harness with the research system prompt)
result = self._run_phase(
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    user_message=f"Phase: DECOMPOSE. Topic: '{self.topic}'. List all subtopics to research.",
    tool_names=[],
    max_iterations=1,
)
subtopics = self._parse_subtopics(result.final_text)
self._log_subtopics(subtopics, iteration)

# Phase 2: Collect findings per subtopic
# The harness retains the decomposition in its conversation history.
# If budget is exhausted, _run_phase() returns immediately with a non-"done" stop_reason.
for subtopic in subtopics:
    self._log(f"Collecting findings for subtopic: {subtopic}")
    result = self._run_phase(
        system_prompt=RESEARCH_SYSTEM_PROMPT,  # same prompt; harness extends the conversation
        user_message=f"Phase: RESEARCH. Subtopic: '{subtopic}'. Use query_knowledge_base.",
        tool_names=["query_knowledge_base"],
        max_iterations=3,
    )
    if result.stop_reason in ("budget_exhausted", "stop_requested"):
        break

# Phase 3: Synthesize
# LLM has full context: decomposition + all per-subtopic findings = higher-quality outline.
result = self._run_phase(
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    user_message="Phase: SYNTHESIZE. Based on all findings above, write the research outline.",
    tool_names=[],
    max_iterations=1,
)
```

### 2.5 AgentHarness: run_bounded() and Budget Enforcement

`AgentHarness` gains a new `run_bounded()` method that `_run_phase()` delegates to:

```python
def run_bounded(
    self,
    user_message: str,
    tool_names: Optional[list[str]],
    max_iterations: int,
    continue_context: bool,
) -> PhaseResult:
    """
    Drive at most max_iterations of the LLM + tool loop and return.

    Budget is checked at the start of each iteration. If budget is exhausted
    before the first iteration begins, returns immediately with
    stop_reason="budget_exhausted" without making any LLM call.

    Token accumulation, cost tracking, budget enforcement, and run summary
    writing are all internal — nothing leaks out through PhaseResult.
    """
```

**Budget enforcement rules** (entirely internal to `run_bounded()`):

1. At the start of each iteration, check if the budget has been exceeded or a stop has been
   requested
2. If budget is exhausted, return `PhaseResult(stop_reason="budget_exhausted")` immediately,
   without making an LLM call
3. If stop was requested, return `PhaseResult(stop_reason="stop_requested")` immediately
4. If `max_iterations` is reached, return `PhaseResult(stop_reason="max_iterations")`
5. If the LLM signals done, return `PhaseResult(stop_reason="done")`
6. Token totals and cost figures are accumulated internally and NEVER placed in `PhaseResult`

**`run()` backward compatibility**: `AgentHarness.run()` becomes a wrapper that calls
`run_bounded()` in a loop until `stop_reason == "done"` or a terminal stop condition is met.
Observable behavior for single-phase agents is unchanged.

### 2.6 PhaseResult: The Phase Execution Contract

`PhaseResult` is the only object returned from `_run_phase()`. It contains operational output —
what happened during the phase — but no financial or resource data.

```python
@dataclass
class PhaseResult:
    final_text: str          # last LLM response text
    tool_calls: list[dict]   # all tool calls made during this phase
    stop_reason: str         # "done" | "max_iterations" | "budget_exhausted" | "stop_requested"
    # NOTE: no tokens_used, cost_usd, or any accounting data — these are internal to the harness
```

**Why `stop_reason` does not violate the financial isolation principle**:

`stop_reason` is a status code, not a financial figure. It tells the agent *why* execution
stopped, not *how much* was consumed. An agent seeing `"budget_exhausted"` knows to stop its
loop — the same decision it would make for any terminal stop signal. It cannot infer token
counts, remaining budget, or cost from this value. Raw numbers are never exposed.

### 2.7 Workspace and Artifact Management

`BaseAgent` gains workspace path accessor methods and a filename helper that all agents must use.
No agent should construct artifact, log, or memory paths by hand, and no agent should format
timestamps independently.

All accessor methods return `pathlib.Path` objects. `Path` is platform-aware by design — it uses
the correct separator on Windows, macOS, and Linux automatically. Agents must never join path
strings manually; they must always use these methods and let `Path` handle platform differences.

`ensure_agent_workspace()` ([lsm/agents/workspace.py](../lsm/agents/workspace.py)) creates:

```
<agents_folder>/<agent_name>/
├── logs/
├── artifacts/    ← all agent file output goes here
└── memory/
```

**New methods on `BaseAgent`**:

```python
def _workspace_root(self) -> Path:
    """Return the agent's workspace root, creating it if needed."""
    return ensure_agent_workspace(self.name, self.agent_config.agents_folder)

def _artifacts_dir(self) -> Path:
    """Return the artifacts directory for this agent, creating it if needed."""
    return self._workspace_root() / "artifacts"

def _logs_dir(self) -> Path:
    """Return the logs directory for this agent, creating it if needed."""
    return self._workspace_root() / "logs"

def _memory_dir(self) -> Path:
    """Return the memory directory for this agent, creating it if needed."""
    return self._workspace_root() / "memory"

def _artifact_filename(self, name: str, suffix: str = ".md") -> str:
    """Generate a platform-safe, consistently formatted filename for an artifact.

    Sanitizes the name descriptor and appends a UTC timestamp so all agents
    produce filenames in the same format regardless of the calling agent.

    Example:
        self._artifact_filename("neural networks overview")
        → "neural_networks_overview_20240315_143022.md"
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-]", "_", name).strip("_")
    return f"{safe_name}_{timestamp}{suffix}"
```

**Migration pattern**:

```python
# Current (ResearchAgent._save_outline):
output_path = self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}.md"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Proposed:
output_path = self._artifacts_dir() / self._artifact_filename(safe_topic)
```

**Scope** — applied to all agents:

- `ResearchAgent._save_outline()`: use `self._artifacts_dir()`
- `SynthesisAgent`, `CuratorAgent`: audit all file output calls; replace with `_artifacts_dir()`
- `WritingAgent`, `AssistantAgent`: same audit and replacement
- Any agent that constructs log or memory paths manually: replace with `_logs_dir()` / `_memory_dir()`

The `_save_log()` call in `ResearchAgent.run()` uses `save_agent_log()` which already targets
`<agents_folder>/<agent_name>/logs/` — verify this aligns with `_logs_dir()` and update if not.

Note: `ResearchResult.output_path` will point to the new location — tests asserting specific
path patterns must be updated.

### 2.8 Design Options Considered

**Option A — Add Protected Methods to `BaseAgent`**: Adds `_safe_execute_tool()` and
`_safe_call_llm()` to `BaseAgent` directly. Preserves loop structures but duplicates harness
logic and doesn't fix run summaries or cost tracking.

**Option B — Migrate Academic Agents to Harness-Driven**: Refactors academic agents to
`GeneralAgent` pattern. Clean long-term but loses explicit programmatic phases; large scope change.

**Option C — Hybrid: Custom Orchestration with Harness Calls**: Retains outer loops but uses
short-lived `AgentHarness` instances per call. Correct concept but harness state/budget across
multiple instances is complex.

**Option D — `BaseAgent` Gets Full LLM + Tool Infrastructure**: Adds full infrastructure to
`BaseAgent`; `AgentHarness` delegates to it. Single source of truth but ownership resolution
is complex.

**Chosen**: A refinement combining C and D — `BaseAgent` owns a single persistent `AgentHarness`
per `run()` lifecycle and exposes `_run_phase()` as the only execution entry point. Outer
orchestration stays in Python; inner execution (including budget enforcement and token tracking)
always goes through the harness.

### 2.9 Scope of Agent Migrations

All agents are in scope. The following table shows the current state and required action:

| Agent | File | Direct LLM Calls | Direct Sandbox Calls | Action |
|-------|------|-------------------|----------------------|--------|
| `ResearchAgent` | [research.py](../lsm/agents/academic/research.py) | Yes | Yes | Refactor to `_run_phase()` |
| `SynthesisAgent` | [synthesis.py](../lsm/agents/academic/synthesis.py) | Yes | Yes | Refactor to `_run_phase()` |
| `CuratorAgent` | [curator.py](../lsm/agents/academic/curator.py) | Yes | Yes | Refactor to `_run_phase()` |
| `WritingAgent` | [writing.py](../lsm/agents/productivity/writing.py) | Yes (partial) | Yes (partial) | Refactor to `_run_phase()` |
| `AssistantAgent` | [assistant.py](../lsm/agents/assistants/assistant.py) | TBD | TBD | Audit and refactor |
| `GeneralAgent` | [general.py](../lsm/agents/productivity/general.py) | No (harness) | No (harness) | Replace harness creation with `_run_phase()` |
| `LibrarianAgent` | [librarian.py](../lsm/agents/librarian/librarian.py) | No (harness) | No (harness) | Replace harness creation with `_run_phase()` |
| `ManuscriptEditorAgent` | [manuscript_editor.py](../lsm/agents/productivity/manuscript_editor.py) | No (harness) | No (harness) | Replace harness creation with `_run_phase()` |

---

## 3. Knowledge Base Access: query_knowledge_base Tool

### 3.1 Problem: Research Agent Reimplements the Query Pipeline

The `ResearchAgent` reimplements a simplified and incomplete version of the query pipeline. The
TODO states: "From the perspective of the Query API it should not care if an agent calls it or the
user. The Research Agent needs to call the query pipeline as if it was a user."

`ResearchAgent._collect_findings()` ([lsm/agents/academic/research.py:187](../lsm/agents/academic/research.py)):

1. `_select_tools()` — the LLM picks from `{query_embeddings, query_<provider>, query_remote_chain}`
2. For each selected tool: `self.sandbox.execute(tool, args)` → raw JSON string
3. Returns `[{"tool": name, "output": raw_json}, ...]`

`ResearchAgent._summarize_findings()` ([lsm/agents/academic/research.py:264](../lsm/agents/academic/research.py)):

1. `_build_sources_block()` — manually parses raw tool output JSON and builds `Sources:\n[S1]...`
2. `provider.synthesize(prompt, sources_block, mode="grounded")` — direct LLM call

The `query_embeddings` tool ([lsm/agents/tools/query_embeddings.py](../lsm/agents/tools/query_embeddings.py))
calls `retrieve_candidates()` directly (raw vector search, no reranking, no prefiltering).

### 3.2 What the Query Pipeline Provides

`lsm.query.api.query()` ([lsm/query/api.py](../lsm/query/api.py)) provides:

- Query prefiltering (path, extension, content-type filters)
- Candidate retrieval + semantic scoring
- LLM-based or score-based reranking
- Remote source integration
- Result caching (TTL-based)
- Full `QueryResult`: answer + candidates + sources_display + cost + remote_sources + debug_info

**What the Research Agent misses** by not using the pipeline:
- Reranking (each subtopic gets only the top-k raw embedding hits, no quality re-scoring)
- Prefiltering (path/extension filters from session config are ignored)
- Remote source chain integration alongside local results
- Caching (repeated subtopic queries are not cached)

### 3.3 The Dependency Gap

`lsm.query.api.query()` requires:

```python
async def query(
    question: str,
    config: LSMConfig,       # full config
    state: SessionState,     # session state with filters, model selection, etc.
    embedder,                # embedding model
    collection,              # vector DB collection
    progress_callback=None,
) -> QueryResult:
```

The `ResearchAgent` currently receives: `llm_registry`, `tool_registry`, `sandbox`, `agent_config`.
It does **not** receive `LSMConfig`, `SessionState`, `embedder`, or `collection`. These are
session-level objects that live above the agent layer.

However, the `query_embeddings` tool already has `collection` and `embedder` injected at
registration time via `create_default_tool_registry()`. The gap is `LSMConfig` and `SessionState`.

### 3.4 Chosen Design

**New `query_knowledge_base` tool** wrapping `query_sync()`, calling the **full pipeline**
including LLM synthesis.

The reasoning:
- The TODO explicitly states agents should call the query pipeline "as a user would" — the full
  pipeline IS the user-facing query API
- Per-subtopic synthesis improves quality: each subtopic gets a grounded summary before the
  Research Agent synthesizes across all subtopics
- The Research Agent's `_build_sources_block` and `_extract_sources_from_output` code can be
  removed since `QueryResult` returns structured `sources_display` and `answer`

**Tool construction**: `query_knowledge_base` is registered in `create_default_tool_registry()`
with `config`, `embedder`, and `collection` injected at registration time. It constructs a fresh
minimal `SessionState` per call using config defaults — agents do NOT receive or manage their own
`SessionState`. This keeps the agent layer clean and avoids mixing agent queries into the user's
interactive session history.

**Agent session state**: Agents do NOT receive `LSMConfig` or `SessionState`. The
`query_knowledge_base` tool encapsulates both at registration time. This is the same pattern
`query_embeddings` already uses for `embedder` and `collection`. No changes to agent constructors
are required.

**Integration with §2**: The Research Agent's collect phase calls `query_knowledge_base` via
`_run_phase(tool_names=["query_knowledge_base"])`. The harness enforces the allowlist; the tool
executes the full query pipeline. The Research Agent's manual `_select_tools()`,
`_collect_findings()`, `_build_sources_block()`, and `_extract_sources_from_output()` methods
are removed.

### 3.5 query_embeddings Deprecation

Once `query_knowledge_base` exists, `query_embeddings` is redundant for any meaningful knowledge
base query. Raw retrieval without synthesis is not a needed capability for current agents.
`query_embeddings` will be:

- Removed from `create_default_tool_registry()`
- Removed from all agent `tool_allowlist` definitions
- Deleted from `lsm/agents/tools/`

If a future raw-retrieval use case emerges, a dedicated tool can be introduced at that time.

### 3.6 Design Options Considered

**Option A — New `query_knowledge_base` Tool ✓ CHOSEN**: Wraps `query_sync()`. Registered in
`create_default_tool_registry()` with config, embedder, collection injected. Creates minimal
`SessionState` per call. Reusable by any agent. No changes to agent constructors.

**Option B — Retrieval-Only Tool**: Calls `build_combined_context_async()` without synthesis.
No LLM duplication but more complex plumbing; less aligned with "call it as a user would."

**Option C — Inject Dependencies into Research Agent**: Pass `LSMConfig` + `embedder` +
`collection` directly into `ResearchAgent`. Direct but tight dependency from agent layer into
query layer; doesn't create a reusable tool.

**Option D — Harness-Driven Research Agent with Query Tool**: Migrate `ResearchAgent` to
`GeneralAgent` pattern. Removes custom loop code but loses explicit deterministic phase control.

---

## 4. Research Agent Observability

These logging improvements are absorbed into the §2 agent migration work and applied when the
`ResearchAgent` orchestration loop is updated to use `_run_phase()`.

**Subtopic iteration logging** — `ResearchAgent.run()` ([lsm/agents/academic/research.py:114](../lsm/agents/academic/research.py)):

```python
# Current:
self._log(f"Research iteration {iteration} with {len(subtopics)} subtopics.")

# Proposed:
subtopic_lines = "\n".join(f"  [{i}] {st}" for i, st in enumerate(subtopics, 1))
self._log(f"Research iteration {iteration} — {len(subtopics)} subtopics:\n{subtopic_lines}")
```

**Review suggestion logging**:

```python
# Current:
self._log(f"Refining with {len(subtopics)} review suggestions.")

# Proposed:
suggestion_lines = "\n".join(f"  [{i}] {s}" for i, s in enumerate(subtopics, 1))
self._log(f"Refining with {len(subtopics)} review suggestions:\n{suggestion_lines}")
```

**Per-subtopic progress log** added inside the collection loop:

```python
self._log(f"Collecting findings for subtopic: {subtopic}")
```

If log verbosity becomes a concern with large subtopic lists, this can be gated behind a verbose
mode via `_log_verbosity()`. For now, listing subtopics inline is the correct default.

---

## 5. Interaction Channel: Two-Phase Timeout

### 5.1 Problem Statement

When an agent posts a permission or clarification request through the `InteractionChannel`, the
harness blocks on `post_request()` for at most `timeout_seconds` (default 300 seconds). After
the timeout the request is auto-denied or auto-approved depending on `timeout_action`.

The problem: the timeout fires even when the user has the interaction panel open in the TUI and
is actively reading or typing a response. The user experiences the request disappearing mid-reply.

### 5.2 Current Architecture

**`InteractionChannel.post_request()`** ([lsm/agents/interaction.py:122](../lsm/agents/interaction.py)):

```python
if event.wait(timeout=float(self.timeout_seconds)):
    # response received
else:
    # timeout: apply timeout_action (deny or approve)
```

There is a single `threading.Event.wait(timeout=...)` call. Once the timeout expires nothing can
extend it. There is no concept of "user has seen this" in the channel.

**TUI interaction polling** ([lsm/ui/tui/screens/agents.py:1176](../lsm/ui/tui/screens/agents.py)):

The `_refresh_interaction_panel()` method runs on a timer (default every 1 second). When it
finds a pending interaction it displays the request fields to the user. However it sends no signal
back to the `InteractionChannel` indicating "this is now visible to the user."

**Config** (`agents.interaction`): One timeout setting (`timeout_seconds = 300`) with no concept
of a pre-view or post-view state.

### 5.3 Root Cause

There is no feedback path from the UI back to the `InteractionChannel` indicating user engagement.
The channel only knows two states: "response received" and "timed out." It has no "acknowledged
by user" state.

### 5.4 Design Options

**Option A — Heartbeat Reset**: TUI sends periodic heartbeats to the channel while a pending
interaction is displayed. Channel tracks last heartbeat time and resets deadline on each heartbeat.
Pros: Transparent, works while user is reading. Cons: Requires polling loop; heartbeat could miss
if TUI timer is slow.

**Option B — Two-Phase Timeout ✓ CHOSEN**: Channel gains `acknowledge_request(request_id)`. When
UI first displays a pending interaction it calls this method. Channel transitions from
"pre-acknowledge timeout" to "post-acknowledge timeout" (long or infinite). Config gains
`acknowledged_timeout_seconds` (default `0` meaning infinite).
Pros: Clean two-state model; easy to reason about; UI only calls acknowledge once per request.
Cons: Requires UI to call acknowledge correctly; "acknowledged" means "displayed" not "typing."

**Option C — Focus Detection**: TUI detects when reply input or buttons have keyboard focus and
calls a `heartbeat()` method. Pros: Most accurate. Cons: Textual focus events are unreliable;
doesn't cover shell path.

**Option D — Config-Only**: Increase default timeout or document manual setting.
Pros: Simplest. Cons: Does not address the problem.

### 5.5 Chosen Design

**Option B (Two-Phase Timeout)** is chosen:

- **Acknowledge trigger**: TUI having displayed the request is sufficient — no requirement for
  the user to have typed or clicked. `_refresh_interaction_panel()` detects and displays the
  request; this event triggers the acknowledgment signal.
- **Post-acknowledge timeout**: `acknowledged_timeout_seconds = 0` (infinite). Once the TUI
  has shown the request to the user, the channel waits indefinitely for a response.

**Required changes:**

- `InteractionChannel` ([lsm/agents/interaction.py](../lsm/agents/interaction.py)):
  - Add `_acknowledged: bool` state and `_acknowledged_at: Optional[datetime]`
  - Add `acknowledge_request(request_id: str) -> None` method; validates request_id matches
    pending request; sets `_acknowledged = True`
  - Modify `post_request()` to use a polling loop: wait in chunks, check acknowledged state,
    switch to `acknowledged_timeout_seconds` once acknowledged; a value of `0` means no timeout
  - Add thread-safety for `_acknowledged` under `_lock`

- `InteractionConfig` dataclass in config models:
  - Add `acknowledged_timeout_seconds: int = 0` (0 = infinite once acknowledged)

- `AgentRuntimeManager` ([lsm/ui/shell/commands/agents.py](../lsm/ui/shell/commands/agents.py)):
  - Add `acknowledge_interaction(agent_id: str, request_id: str)` method that forwards to
    the correct run's `InteractionChannel`

- TUI `_refresh_interaction_panel()` ([lsm/ui/tui/screens/agents.py](../lsm/ui/tui/screens/agents.py)):
  - When a new pending request is detected and displayed, call
    `manager.acknowledge_interaction(agent_id, request_id)` once per unique `request_id`
  - Track `_acknowledged_interaction_ids: set[str]` to avoid duplicate acknowledgments

- Shell interaction commands: When `/agent interact` displays an interaction, call acknowledge.

---

## 6. Implementation Plan

### 6.1 Dependency Map

```
§2 — Core Architecture: Persistent Harness and Phase Execution
├── enables → §3 (Research Agent uses query_knowledge_base via _run_phase)
├── absorbs → §4 (subtopic logging updated as part of research loop refactor)
└── absorbs → §2.7 (BaseAgent workspace accessors applied during agent migration)

§5 — Interaction Channel: Two-Phase Timeout
└── Independent of §2–§4; can be developed in parallel
```

### 6.2 Implementation Order

All items are bundled into a single implementation pass. Recommended order within that pass:

1. **§5** (interaction timeout): Implement first as it is fully independent. Changes are
   isolated to `InteractionChannel`, `InteractionConfig`, `AgentRuntimeManager`, and TUI.

2. **§2 infrastructure** (`_run_phase()`, `PhaseResult`, `run_bounded()`, workspace accessors):
   Implement `_run_phase()` on `BaseAgent`, `PhaseResult` dataclass, and `run_bounded()` on
   `AgentHarness`. Add workspace accessor methods. No agents are migrated yet — this establishes
   the infrastructure that all subsequent steps depend on.

3. **§3** (`query_knowledge_base` tool): Implement the new tool. Remove `query_embeddings` from
   `create_default_tool_registry()`. Verify `query_sync()` is callable from tool context.

4. **§2 migration — simple agents** (`GeneralAgent`, `LibrarianAgent`, `ManuscriptEditorAgent`):
   Replace direct `AgentHarness` instantiation with `self._run_phase()`. Verify behavior is
   unchanged with existing tests.

5. **§2 migration — academic agents** (`ResearchAgent`, `SynthesisAgent`, `CuratorAgent`):
   Refactor each to use `_run_phase()`. `ResearchAgent` uses `query_knowledge_base`. Remove all
   direct `provider.synthesize()` and `sandbox.execute()` calls. Apply §2.7 workspace accessors
   and §4 logging improvements simultaneously.

6. **§2 migration — remaining agents** (`WritingAgent`, `AssistantAgent`): Audit and refactor.
   Apply workspace accessor changes.

### 6.3 Testing Strategy

All changes follow the TDD approach: tests are written before implementation. The test suite
requires updates at multiple layers. Since `AgentHarness` is being reinforced as the sole
execution control point, tests must verify both the harness internals and the invariant that no
agent bypasses it.

#### Layer 1: InteractionChannel (§5)

- **Two-phase state machine**:
  - `post_request()` without acknowledgment times out after `timeout_seconds`
  - `acknowledge_request()` transitions channel to acknowledged state
  - After acknowledgment, `post_request()` waits indefinitely (`acknowledged_timeout_seconds=0`)
  - Acknowledging with a mismatched `request_id` does not acknowledge the pending request
  - Thread-safety: concurrent acknowledge and timeout polling do not race
- **TUI acknowledgment signal**:
  - `_refresh_interaction_panel()` calls `acknowledge_interaction()` exactly once per unique
    `request_id` regardless of how many timer ticks fire while the request is pending
  - A second refresh for the same `request_id` does NOT call acknowledge again

#### Layer 2: AgentHarness — run_bounded() (§2 infrastructure)

- **Loop mechanics**:
  - Returns `PhaseResult(stop_reason="max_iterations")` after exactly `max_iterations`
  - Returns `PhaseResult(stop_reason="done")` when LLM signals completion before the limit
  - Returns `PhaseResult(stop_reason="stop_requested")` when a stop is signaled mid-loop

- **Budget enforcement — fully internal**:
  - If budget is already exhausted on entry: returns `PhaseResult(stop_reason="budget_exhausted")`
    immediately without making any LLM call or tool call
  - After budget exhaustion, a subsequent `run_bounded()` call returns immediately the same way
  - Budget is checked at the start of each iteration, before the LLM call
  - `PhaseResult` has NO `tokens_used` field and NO `cost_usd` field — assert these attributes
    do not exist on the `PhaseResult` dataclass (type-level check)

- **Context continuity**:
  - `continue_context=True`: conversation history from a prior phase is present in the LLM
    context for the next phase (verify via messages captured in the mock LLM call)
  - `continue_context=False`: conversation history is reset; LLM receives no prior context

- **Tool allowlist subset enforcement**:
  - `tool_names=["tool_a"]` causes only `tool_a` to be offered to the LLM
  - A tool call for `tool_b` (in the agent's allowlist but not in `tool_names`) is rejected
  - `tool_names=None` allows all tools in the agent's allowlist
  - `tool_names=[]` allows no tools (LLM still executes but cannot call tools)

- **Run summary and artifact tracking**:
  - `_track_artifacts_from_sandbox()` is called for each tool execution within a phase
  - `run_summary.json` is written with cumulative token totals across all phases after `run()`
    completes — not after each individual `run_bounded()` call
  - Token totals in `run_summary.json` are the sum over all phases of the run

- **`run()` backward compatibility**:
  - `AgentHarness.run()` wraps `run_bounded()` and produces identical observable behavior to
    the pre-refactor implementation for single-phase agents

#### Layer 3: BaseAgent._run_phase() (§2 infrastructure)

- **Harness lifecycle**:
  - The first `_run_phase()` call creates exactly one `AgentHarness` instance (spy on
    `AgentHarness.__init__` and assert call count = 1 after multiple `_run_phase()` calls)
  - A second call to `BaseAgent.run()` creates a new `AgentHarness` instance (fresh per run)

- **Security invariants on BaseAgent**:
  - `BaseAgent` has NO `_check_budget_and_stop()` method — assert `AttributeError` on access
  - `BaseAgent` has NO `_tokens_used` attribute — assert `AttributeError` on access
  - `BaseAgent` imports NO token counting or cost utilities directly
  - `_run_phase()` returns a `PhaseResult` with no token/cost fields (type check)

#### Layer 4: Workspace and Artifact Management (§2.7)

- `_artifacts_dir()` returns `Path(<agents_folder>/<agent_name>/artifacts/)`
- `_logs_dir()` returns `Path(<agents_folder>/<agent_name>/logs/)`
- `_memory_dir()` returns `Path(<agents_folder>/<agent_name>/memory/)`
- `_workspace_root()` creates the full directory tree when the directories do not yet exist
- `_artifact_filename("some topic name")` → `"some_topic_name_YYYYMMDD_HHMMSS.md"`
- `_artifact_filename()` converts spaces, slashes, colons, and other non-word characters to `_`
- `_artifact_filename()` strips leading and trailing underscores from the sanitized name
- `_artifact_filename("name", suffix=".json")` respects the `suffix` parameter
- `ResearchAgent._save_outline()` writes to `self._artifacts_dir() / self._artifact_filename(...)`
  — existing path-assertion tests must be updated to the new location

#### Layer 5: query_knowledge_base Tool (§3)

- `execute()` calls `query_sync()` with the injected config, embedder, and collection
- Each call constructs a **fresh** `SessionState` — verify session state is not shared between
  calls (e.g., history from a prior call does not appear in a subsequent call's state)
- Output is serialized to JSON containing at minimum: `answer`, `sources_display`, and relevant
  candidate fields
- `create_default_tool_registry()` registers `query_knowledge_base` and does NOT register
  `query_embeddings` — assert `query_embeddings` is absent from the returned registry
- Input schema: `query` is required; missing or blank `query` raises `ValueError`
- `lsm/agents/tools/query_embeddings.py` is deleted — assert the file does not exist

#### Layer 6: Agent Migration — Security Invariants and Regression (§2.9)

These tests verify that all agents route execution through the harness and that behavioral
regressions are caught.

**No-bypass assertions** — applied to every agent module:

  - No agent module directly calls `sandbox.execute()` — enforce via AST inspection test or
    a monkey-patch test that raises `AssertionError` if sandbox is invoked outside the harness
  - No agent module directly calls any `provider.*` method — same enforcement approach
  - No agent module contains a direct `AgentHarness(...)` instantiation — assert via AST or
    import-time check

**Single harness per run** — for each refactored agent:

  - Spy on `AgentHarness.__init__`; call `agent.run(mock_task)`; assert `__init__` was called
    exactly once during the run

**Simple agent regression** (`GeneralAgent`, `LibrarianAgent`, `ManuscriptEditorAgent`):

  - Mock at the `_run_phase()` level; capture the `system_prompt`, `user_message`, and
    `tool_names` arguments; assert they match what was previously passed to `harness.run()`
  - Assert `PhaseResult.final_text` is correctly consumed by each agent's post-run logic

**ResearchAgent phase flow**:

  - Assert phases execute in order: DECOMPOSE → RESEARCH (per subtopic) → SYNTHESIZE → REVIEW
  - Assert each per-subtopic log entry contains the subtopic name string
  - Assert `_run_phase()` is called with `tool_names=["query_knowledge_base"]` for RESEARCH
    phases and `tool_names=[]` for DECOMPOSE and SYNTHESIZE phases
  - When `run_bounded()` returns `stop_reason="budget_exhausted"`, assert the subtopic loop
    terminates early and SYNTHESIZE still executes with whatever findings were collected
  - Assert output file is written under `self._artifacts_dir()` with a filename matching the
    `_artifact_filename()` format (`name_YYYYMMDD_HHMMSS.md`)

**Existing test suite updates required**:

  - Tests that monkeypatch `sandbox.execute()` or `provider.synthesize()` on agent instances
    must be rewritten to mock at the `_run_phase()` or `run_bounded()` level
  - Tests asserting `ResearchResult.output_path` must be updated to reflect the new
    `artifacts/` subdirectory location
  - Tests asserting `_tokens_used` or `_cost` attributes on agent instances must be removed
  - Tests asserting `AgentHarness` is directly instantiated inside an agent's `run()` method
    must be inverted — they should now assert it is NOT directly instantiated

### 6.4 Documentation Updates

- `AGENTS.md`: Update interaction section (§5 two-phase timeout); update `BaseAgent` section
  with `_run_phase()`, `PhaseResult`, and workspace accessors; update Research Agent section
  with new workflow and `query_knowledge_base`; note artifact layout
- `lsm.agents.md` package doc: Document new `BaseAgent` methods; document `PhaseResult`;
  document `AgentHarness.run_bounded()`
- `CONFIG.md`: Document new `acknowledged_timeout_seconds` config field
- `CHANGELOG.md`: All changes documented per standard phase checklist

---

## 7. Resolved Decisions

All clarifying questions from the discovery phase have been answered.

**§5 — Interaction Channel**

| # | Question | Decision |
|---|----------|----------|
| Q1 | What level of user engagement triggers timeout suspension? | TUI displaying the request to the user (interaction panel has shown it). No requirement for user to have typed or clicked. Option B (two-phase timeout). |
| Q2 | Post-acknowledge timeout behavior? | No timeout — infinite wait once acknowledged. `acknowledged_timeout_seconds = 0`. |

**§3 — Knowledge Base Access**

| # | Question | Decision |
|---|----------|----------|
| Q3 | Full pipeline or retrieval only? | Full pipeline including LLM synthesis. Each subtopic gets a pre-synthesized answer via `query_sync()`. |
| Q4 | Built-in tool or direct dependency? | New built-in tool (`query_knowledge_base`). Reusable by any agent. |
| Q5 | Is `query_embeddings` still needed after `query_knowledge_base` exists? | No. Deprecated and removed from default tool registry. Deleted. |
| Q6 | Should agents receive `LSMConfig` and their own `SessionState`? | No. The `query_knowledge_base` tool encapsulates both at registration time. Agent constructors do not change. |

**§2 — Core Architecture**

| # | Question | Decision |
|---|----------|----------|
| Q7 | Which refactoring approach? | Persistent harness: `BaseAgent` gains `_run_phase()` as the single execution control point. All agents call `_run_phase()`; no agent touches `AgentHarness`, sandbox, or provider directly. |
| Q8 | Which agents are in scope? | All agents — complete consistency pass. |
| Q9 | Bundle or implement separately? | Everything bundled into a single implementation pass. §4 and §2.7 are absorbed into the §2 migration work. |
| Q10 | Should workspace paths be platform-formatted? Should filenames be consistent? | Yes. All workspace accessor methods return `pathlib.Path` objects (platform-aware by design). `BaseAgent` gains `_artifact_filename(name, suffix)` to produce consistently timestamped filenames; no agent formats its own timestamps. |
| Q11 | Should the harness be recreated per `_run_phase()` call or persist for the run? | Persist. `BaseAgent` owns one `AgentHarness` per `run()` lifecycle. Token tracking, cost, and budget stay fully internal to the harness. |
| Q12 | Does the agent need a budget-check method between phases? | No. Budget checking, enforcement, and stop decisions are entirely internal to the harness. No budget-query API exists on `BaseAgent`. Agents react to `PhaseResult.stop_reason` to control their loops — this conveys status only, not financial data. |
