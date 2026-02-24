# Agents Guide

This guide covers the agent system: architecture, configuration, tools, sandboxing, memory, scheduler, and meta-agent orchestration usage.

## Overview

The agent runtime is implemented in `lsm/agents/` and integrated into both:

- TUI (`lsm/ui/tui/screens/agents.py`)
- Shell commands (`/agent ...` via `lsm/ui/shell/commands/agents.py`)

The shell runtime manager supports concurrent runs and assigns each started run a unique `agent_id`.

Built-in agent:

- `curator`
- `meta`
- `research`
- `writing`
- `synthesis`

## Architecture

- `lsm/agents/base.py`: `BaseAgent`, `AgentState`, lifecycle status model
- `lsm/agents/harness.py`: runtime loop, tool-calling execution, budget/iteration guards, state persistence, per-run summaries
- `lsm/agents/interaction.py`: thread-safe request/response channel for runtime-to-UI interaction handshakes
- `lsm/agents/models.py`: runtime message/log/response models
- `lsm/agents/log_formatter.py`: log formatting and serialization helpers
- `lsm/agents/factory.py`: registry + `create_agent(...)`
- `lsm/agents/academic/research.py`: built-in research workflow agent
- `lsm/agents/productivity/writing.py`: built-in grounded writing workflow agent
- `lsm/agents/academic/synthesis.py`: built-in synthesis workflow agent
- `lsm/agents/academic/curator.py`: built-in corpus curation workflow agent
- `lsm/agents/scheduler.py`: recurring schedule engine for harness-driven agent runs
- `lsm/agents/meta/task_graph.py`: task graph datamodels and dependency-order helpers for meta orchestration
- `lsm/agents/meta/meta.py`: built-in meta-agent orchestrator with shared-workspace execution and final synthesis
- `lsm/agents/memory/models.py`: memory and memory-candidate dataclasses
- `lsm/agents/memory/store.py`: memory store abstraction + SQLite/PostgreSQL backends
- `lsm/agents/memory/migrations.py`: SQLite <-> PostgreSQL migration helpers
- `lsm/agents/memory/api.py`: memory lifecycle operations and ranked retrieval helpers
- `lsm/agents/memory/context_builder.py`: standing-context builder for memory prompt injection

Tooling:

- `lsm/agents/tools/base.py`: `BaseTool`, `ToolRegistry`
- `lsm/agents/tools/sandbox.py`: `ToolSandbox` permission enforcement + runner policy
- `lsm/agents/tools/runner.py`: base runner interface + `LocalRunner`
- `lsm/agents/tools/docker_runner.py`: `DockerRunner` foundation
- `lsm/agents/tools/spawn_agent.py`: meta-system tool for sub-agent spawning
- `lsm/agents/tools/await_agent.py`: meta-system tool for sub-agent completion waits
- `lsm/agents/tools/collect_artifacts.py`: meta-system tool for sub-agent artifact collection
- `lsm/agents/tools/ask_user.py`: clarification tool for runtime user interaction
- `lsm/agents/tools/*.py`: built-in tool implementations
- Tool metadata includes `risk_level`, `preferred_runner`, and `needs_network`

## Agent Config

Agents are configured in top-level `agents`:

```json
"agents": {
  "enabled": true,
  "agents_folder": "Agents",
  "max_tokens_budget": 200000,
  "max_iterations": 25,
  "max_concurrent": 5,
  "log_stream_queue_limit": 500,
  "context_window_strategy": "compact",
  "sandbox": {
    "allowed_read_paths": ["./notes", "./docs"],
    "allowed_write_paths": ["./notes", "./Agents"],
    "allow_url_access": false,
    "require_user_permission": {},
    "force_docker": false,
    "tool_llm_assignments": {}
  },
  "memory": {
    "enabled": true,
    "storage_backend": "auto",
    "sqlite_path": "memory.sqlite3",
    "postgres_connection_string": null,
    "postgres_table_prefix": "agent_memory",
    "ttl_project_fact_days": 90,
    "ttl_task_state_days": 7,
    "ttl_cache_hours": 24
  },
  "interaction": {
    "timeout_seconds": 300,
    "timeout_action": "deny"
  },
  "agent_configs": {
    "research": {
      "max_iterations": 30
    }
  },
  "schedules": [
    {
      "agent_name": "curator",
      "params": {"topic": "--mode memory", "force_docker": true},
      "interval": "daily",
      "enabled": false,
      "concurrency_policy": "skip",
      "confirmation_mode": "auto"
    }
  ]
}
```

### Key Fields

- `enabled`: turns agent features on/off
- `agents_folder`: where run state/log files are written
- `max_tokens_budget`: approximate token cap per run
- `max_iterations`: max action loop iterations
- `max_concurrent`: max concurrently running agents (used by multi-agent runtime managers)
- `log_stream_queue_limit`: max buffered live-log entries per agent before oldest entries are dropped
- `context_window_strategy`: `compact` or `fresh`
- `interaction.timeout_seconds`: wait timeout for an interaction response (default `300`)
- `interaction.timeout_action`: timeout fallback (`deny` or `approve`, default `deny`)
- `memory`: persistent memory backend config and TTL caps

## Workspace Layout

Each agent has a default workspace rooted at:

`<agents_folder>/<agent_name>/`

The harness creates this structure on first run:

- `logs/` (run summaries, state snapshots, per-run workspaces)
- `artifacts/` (agent-generated outputs)
- `memory/` (agent memory data files)

Relative paths passed to `read_file`/`write_file` resolve against the per-agent workspace root.
- `agent_configs`: per-agent overrides
- `agent_configs` can include `llm_tier`, `llm_service`, or `llm_provider` + `llm_model` to select models
- `schedules`: optional scheduled runs (`hourly`, `daily`, `weekly`, `<seconds>s`, or cron intervals)

## Interaction Channel

`lsm/agents/interaction.py` defines a thread-safe bridge between background harness threads and UI threads.

- `InteractionRequest`: request envelope with `request_id`, `request_type`, `tool_name`, `risk_level`, `reason`, `args_summary`, `prompt`, and `timestamp`
- `InteractionResponse`: response envelope with `request_id`, `decision`, and optional `user_message`
- `InteractionChannel`:
  - `post_request(...)` blocks until UI responds or timeout is reached
  - `get_pending_request()` and `has_pending()` support non-blocking UI polling
  - `post_response(...)` fulfills a pending request
  - `cancel_pending(...)` and `shutdown(...)` unblock waiting runtime calls safely
  - stores per-session tool approvals for `approve_session` flows

Timeout behavior is configurable via `agents.interaction`:

- `timeout_action="deny"` raises `PermissionError` on timeout (safe default)
- `timeout_action="approve"` auto-approves timed-out requests

Runtime usage:

- `ToolSandbox` posts `"permission"` requests through the channel when a tool requires confirmation
- `AgentHarness` posts clarification requests (for example through `ask_user`) and transitions `RUNNING -> WAITING_USER -> RUNNING`
- `AgentHarness.stop()` cancels any pending request so blocked tool execution unblocks deterministically
- `AgentRuntimeManager` keeps one interaction channel per running agent and forwards UI responses to the correct run by `agent_id`

## Sandbox Model

Sandbox rules are defined in `agents.sandbox` and enforced by `ToolSandbox`.

- `allowed_read_paths`: readable locations for file tools
- `allowed_write_paths`: writable locations for file tools
- `allow_url_access`: permits/disallows URL tools
- `require_user_permission`: per-tool interactive gate
- `require_permission_by_risk`: per-risk interactive gate (`read_only`, `writes_workspace`, `network`, `exec`)
- `execution_mode`: runner policy (`local_only` or `prefer_docker`)
- `force_docker`: require Docker execution for all tool risks; block when Docker is unavailable
- `limits`: execution limits (`timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`)
- `docker`: docker runner settings (`enabled`, `image`, `network_default`, `cpu_limit`, `mem_limit_mb`, `read_only_root`)
- `tool_llm_assignments`: optional per-tool service mapping

The sandbox is deny-by-default outside configured paths.
Permission precedence is:
`require_user_permission[tool]` -> `require_permission_by_risk[risk_level]` -> `tool.requires_permission` -> allow.
Execution flow is:
permission checks -> runner selection -> environment scrubbing -> runner execution -> output redaction.
Runner selection policy is:
`read_only`/`writes_workspace` -> local runner (unless `force_docker=true`);
`network`/`exec` in `prefer_docker` mode -> docker runner when available, otherwise confirmation-required block.
When `force_docker=true`, all risks require Docker and local fallback is blocked.

## Memory Storage

Memory storage is configured in `agents.memory` and implemented in `lsm/agents/memory/`.

- `storage_backend`: `auto`, `sqlite`, or `postgresql`
- `sqlite_path`: SQLite DB file path for memory storage
- `postgres_connection_string`: optional PostgreSQL override for memory storage
- `postgres_table_prefix`: table prefix used by PostgreSQL memory tables
- TTL caps: `ttl_project_fact_days`, `ttl_task_state_days`, `ttl_cache_hours`

Backend selection in `auto` mode:

- Uses PostgreSQL memory storage when `vectordb.provider` is `postgresql`
- Uses SQLite memory storage for `chromadb` (or any non-PostgreSQL vector backend)

Path resolution:

- If `agents.memory.sqlite_path` is relative, it resolves under `agents_folder`

Lifecycle model:

- Memory candidates are created as `pending`
- Candidates can be `promoted` or `rejected`
- Promoted memories are searchable
- Expired memories are removed via TTL cleanup
- `memory_search()` applies recency scoring with pin weighting
- `last_used_at` is updated only for memories actually injected into standing context

Runtime integration:

- `MemoryContextBuilder` runs before each LLM call in `AgentHarness`
- Standing memory context is injected as a separate prompt block (not merged into tool definitions)
- Zero-memory injection is valid and simply omits the standing-context block
- Memory tools (`memory_put`, `memory_search`) are registered only when a memory backend is initialized and passed into the tool registry
- `memory_put` supports both creating pending memory candidates and updating existing memory records by `memory_id`

Migration helpers are provided in `lsm/agents/memory/migrations.py` for SQLite-to-PostgreSQL and PostgreSQL-to-SQLite moves.

## Built-In Tools

Default tool registry (`create_default_tool_registry`) includes:

- `read_file` (`risk_level=read_only`)
- `read_folder` (`risk_level=read_only`)
- `file_metadata` (`risk_level=read_only`)
- `hash_file` (`risk_level=read_only`)
- `source_map` (`risk_level=read_only`)
- `query_embeddings` (`risk_level=read_only`) (registered only when vector DB provider + embedder are available)
- `extract_snippets` (`risk_level=read_only`) (registered only when vector DB provider + embedder are available)
- `similarity_search` (`risk_level=read_only`) (registered only when vector DB provider is available)
- `write_file` (`risk_level=writes_workspace`)
- `append_file` (`risk_level=writes_workspace`)
- `create_folder` (`risk_level=writes_workspace`)
- `memory_put` (`risk_level=writes_workspace`, `requires_permission=true`) (registered when memory is enabled and a memory backend is provided to the default registry)
- `memory_remove` (`risk_level=writes_workspace`, `requires_permission=true`) (registered when memory is enabled and a memory backend is provided to the default registry)
- `load_url` (`risk_level=network`, `needs_network=true`)
- `query_llm` (`risk_level=network`, `needs_network=true`)
- `query_remote` (`risk_level=network`, `needs_network=true`)
- `query_remote_chain` (`risk_level=network`, `needs_network=true`)
- `memory_search` (`risk_level=read_only`) (registered when memory is enabled and a memory backend is provided to the default registry)
- `ask_user` (`risk_level=read_only`) (always available in harness/base-agent allowlist filtering)
- `spawn_agent` (`risk_level=exec`, `requires_permission=true`)
- `await_agent` (`risk_level=exec`, `requires_permission=true`)
- `collect_artifacts` (`risk_level=exec`, `requires_permission=true`)

`ToolRegistry` also supports risk-based inspection with:

- `list_by_risk(risk_level)`
- `list_network_tools()`

## Runtime Loop

`AgentHarness` loop:

1. Prepare standing memory context (if available) + filtered tool definitions (`tool_allowlist` when provided)
2. Call LLM
3. Parse strict JSON action response
4. Execute tool via sandbox when requested (reject unlisted tools; interactive permission checks block on channel responses when configured)
5. Append tool output to context
6. Stop on `DONE`, stop request, budget exhaustion, or iteration cap

When waiting on a user interaction request, harness status is set to `WAITING_USER` and restored to `RUNNING` after a response (or `COMPLETED` when a stop request arrives during the wait).

## Runtime Manager

`AgentRuntimeManager` (in `lsm/ui/shell/commands/agents.py`) tracks active agent sessions and recent completed history.

- Active run records include `agent_id`, `agent_name`, `agent`, `thread`, `harness`, `channel`, `started_at`, and `topic`
- Concurrency is limited by `agents.max_concurrent` (default `5`)
- Control APIs accept optional `agent_id` and target all active runs only when unambiguous (single-run compatibility)
- Completed-run retention is bounded (default last `10` runs)
- `shutdown()` cancels pending interaction requests and joins active run threads
- Interaction helpers:
  - `get_pending_interactions()` returns active run prompts waiting for user input
  - `respond_to_interaction(agent_id, response)` posts an `InteractionResponse` to the run channel
- Agents TUI (`lsm/ui/tui/screens/agents.py`) now consumes these APIs with:
  - a polling Running Agents table (`agent_id`, agent, topic, status, duration)
  - row-based log targeting for concurrent sessions
  - an interaction request panel that supports permission decisions and clarification replies per `agent_id`
  - urgency indicator styling for pending interaction requests
  - launch-first panel order (launch panel, `Running Agents`, then status/control panel)
  - live-log follow behavior that only auto-scrolls when already at the end, with `Log` forcing a jump to latest output

Per-run workspace is created as:

- `<agents_folder>/<agent_name>_<timestamp>/workspace/`

State is persisted under the same run folder as:

- `<agents_folder>/<agent_name>_<timestamp>/<agent_name>_<timestamp>_state.json`

Each run also emits a summary artifact:

- `<agents_folder>/<agent_name>_<timestamp>/run_summary.json`

`run_summary.json` includes agent/topic metadata, tool usage, approval/denial counts, artifacts, run outcome, duration, token usage, and extracted user constraints.

## Research Agent

The `research` agent decomposes a topic, queries available sources/tools, iteratively synthesizes findings, and writes structured output.

Use when you need multi-step retrieval + synthesis rather than a single query turn.

## Writing Agent

The `writing` agent gathers grounding evidence from local tools, builds an outline, drafts prose, self-reviews, and writes a final markdown deliverable.

Use when you need a polished grounded write-up from your local knowledge base.

## Synthesis Agent

The `synthesis` agent selects scope, gathers candidate sources, synthesizes compact output in a target format (bullets, outline, narrative, or QA), and writes:

- `synthesis.md`
- `source_map.md`

Use when you need concise cross-document distillation with explicit source coverage.

## Curator Agent

The `curator` agent inventories files, collects metadata, detects exact and near-duplicates, applies staleness/quality heuristics, and writes:

- `curation_report.md`

Use when you want actionable maintenance recommendations for corpus quality.

Curator also supports memory distillation mode via `--mode memory` in the topic (or `agent_configs.curator.mode=memory`), which scans recent `run_summary.json` files and writes:

- `memory_candidates.md`
- `memory_candidates.json`

## Meta Agent

The `meta` agent converts a high-level goal into a dependency-aware task graph for sub-agents (`research`, `writing`, `synthesis`, `curator`), executes that graph, and writes consolidated run artifacts.

Core (Phase 6.1) behavior:

- builds deterministic orchestration graphs with dependency-safe ordering
- supports structured JSON goals with explicit tasks/dependencies
- supports deterministic fallback planning from plain-text goals

Meta-system tooling (Phase 6.2):

- `spawn_agent` starts sub-agent harness runs
- `await_agent` blocks on sub-agent completion
- `collect_artifacts` returns sub-agent artifact paths (optionally filtered by glob)
- sub-agent sandboxes are derived as monotone subsets of the parent sandbox (paths, network, permission gates, runner policy, and limits cannot be widened)

Shared workspace + synthesis (Phase 6.3):

- per-run layout:
  - `<agents_folder>/meta_<timestamp>/workspace/` (shared read workspace)
  - `<agents_folder>/meta_<timestamp>/sub_agents/<agent_name>_<NNN>/` (per-sub-agent write workspace)
  - `<agents_folder>/meta_<timestamp>/final_result.md`
  - `<agents_folder>/meta_<timestamp>/meta_log.md`
- sub-agents can read the shared `workspace/` and write only to their own `sub_agents/<agent_name>_<NNN>/` directory
- final synthesis attempts an LLM-generated consolidated result and falls back to a deterministic markdown summary when synthesis is unavailable

## TUI Usage

Open the **Agents** tab:

- in the launch panel, choose an agent, enter a topic, and press `Enter` to start
- in **Running Agents**, pick the active run row (`F6`/`F7` also move selection)
- in the status panel, use status/pause/resume/stop/log controls for the selected run
- inspect live status/log output (auto-follows only when scrolled to the bottom)
- press **Log** to reload persisted run log output and jump to the log end
- when a run is stopping, the stop flow waits for in-flight work to finish, then persists run logs/artifacts before reporting completion
- approve/deny/reply in **Interaction Request** when permission/clarification prompts appear
- review the Meta panel task graph table
- review per-sub-agent run status/workspaces in the Meta panel
- inspect meta artifact paths (`final_result.md`, `meta_log.md`) from the Meta panel
- review schedules in a table with runtime status
- add/remove schedules from the panel
- enable/disable schedules from the panel
- review pending memory candidates
- approve/reject candidates
- edit candidate TTL (days)

## Shell Commands

- `/agent start <name> <topic>`
- `/agent list`
- `/agent interact [agent_id]`
- `/agent approve <agent_id>`
- `/agent deny <agent_id> [reason]`
- `/agent approve-session <agent_id>`
- `/agent reply <agent_id> <message>`
- `/agent queue [agent_id] <message>`
- `/agent select <agent_id>`
- `/agent status [agent_id]`
- `/agent pause [agent_id]`
- `/agent resume [agent_id] [message]`
- `/agent stop [agent_id]`
- `/agent log [agent_id]`
- `/agent meta start <goal>`
- `/agent meta status`
- `/agent meta log`
- `/agent schedule add <agent_name> <interval> [--params '{"topic":"..."}'] [--concurrency_policy skip|queue|cancel] [--confirmation_mode auto|confirm|deny]`
- `/agent schedule list`
- `/agent schedule enable <schedule_id>`
- `/agent schedule disable <schedule_id>`
- `/agent schedule remove <schedule_id>`
- `/agent schedule status`
- `/memory candidates [pending|promoted|rejected|all]`
- `/memory promote <candidate_id>`
- `/memory reject <candidate_id>`
- `/memory ttl <candidate_id> <days>`

`/agent queue` lets you inject follow-up user instructions into a running agent context.  
`/agent resume ... [message]` queues that message first, then resumes the selected/targeted agent.

## Scheduler Engine

`AgentScheduler` (`lsm/agents/scheduler.py`) executes configured `agents.schedules` entries on an interval and persists runtime status in:

- `<agents_folder>/schedules.json`

Tracked fields include `last_run_at`, `next_run_at`, `last_status`, and `last_error`.

Concurrency handling per schedule:

- `skip`: skip overlapping due runs
- `queue`: queue one or more follow-up runs
- `cancel`: request stop on current run and queue a replacement run

Safety defaults for scheduled runs:

- read-only tool risk only by default
- network disabled (`allow_url_access=false`)
- write/network/exec risks require explicit schedule param opt-in (`allow_writes`, `allow_network`, `allow_exec`)
- when network or exec is allowed, sandbox execution mode is forced to `prefer_docker`
- schedule param `force_docker=true` forces Docker execution for all tool risks in that run

## Notes

- Agents are optional and disabled unless `agents.enabled = true`.
- Keep sandbox paths minimal and explicit.
- Prefer dedicated low-cost service models for tool-heavy runs using `llms.services`.
