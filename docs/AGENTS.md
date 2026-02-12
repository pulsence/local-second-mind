# Agents Guide

This guide covers the agent system added in `0.4.0`: architecture, configuration, tools, sandboxing, and runtime usage.

## Overview

The agent runtime is implemented in `lsm/agents/` and integrated into both:

- TUI (`lsm/ui/tui/screens/agents.py`)
- Shell commands (`/agent ...` via `lsm/ui/shell/commands/agents.py`)

Built-in agent:

- `curator`
- `research`
- `writing`
- `synthesis`

## Architecture

- `lsm/agents/base.py`: `BaseAgent`, `AgentState`, lifecycle status model
- `lsm/agents/harness.py`: runtime loop, tool-calling execution, budget/iteration guards, state persistence, per-run summaries
- `lsm/agents/models.py`: runtime message/log/response models
- `lsm/agents/log_formatter.py`: log formatting and serialization helpers
- `lsm/agents/factory.py`: registry + `create_agent(...)`
- `lsm/agents/research.py`: built-in research workflow agent
- `lsm/agents/writing.py`: built-in grounded writing workflow agent
- `lsm/agents/synthesis.py`: built-in synthesis workflow agent
- `lsm/agents/curator.py`: built-in corpus curation workflow agent
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
  "context_window_strategy": "compact",
  "sandbox": {
    "allowed_read_paths": ["./notes", "./docs"],
    "allowed_write_paths": ["./notes", "./Agents"],
    "allow_url_access": false,
    "require_user_permission": {},
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
  "agent_configs": {
    "research": {
      "max_iterations": 30
    }
  }
}
```

### Key Fields

- `enabled`: turns agent features on/off
- `agents_folder`: where run state/log files are written
- `max_tokens_budget`: approximate token cap per run
- `max_iterations`: max action loop iterations
- `context_window_strategy`: `compact` or `fresh`
- `memory`: persistent memory backend config and TTL caps
- `agent_configs`: per-agent overrides

## Sandbox Model

Sandbox rules are defined in `agents.sandbox` and enforced by `ToolSandbox`.

- `allowed_read_paths`: readable locations for file tools
- `allowed_write_paths`: writable locations for file tools
- `allow_url_access`: permits/disallows URL tools
- `require_user_permission`: per-tool interactive gate
- `require_permission_by_risk`: per-risk interactive gate (`read_only`, `writes_workspace`, `network`, `exec`)
- `execution_mode`: runner policy (`local_only` or `prefer_docker`)
- `limits`: execution limits (`timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`)
- `docker`: docker runner settings (`enabled`, `image`, `network_default`, `cpu_limit`, `mem_limit_mb`, `read_only_root`)
- `tool_llm_assignments`: optional per-tool service mapping

The sandbox is deny-by-default outside configured paths.
Permission precedence is:
`require_user_permission[tool]` -> `require_permission_by_risk[risk_level]` -> `tool.requires_permission` -> allow.
Execution flow is:
permission checks -> runner selection -> environment scrubbing -> runner execution -> output redaction.
Runner selection policy is:
`read_only`/`writes_workspace` -> local runner;
`network`/`exec` in `prefer_docker` mode -> docker runner when available, otherwise confirmation-required block.

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

`ToolRegistry` also supports risk-based inspection with:

- `list_by_risk(risk_level)`
- `list_network_tools()`

## Runtime Loop

`AgentHarness` loop:

1. Prepare standing memory context (if available) + filtered tool definitions (`tool_allowlist` when provided)
2. Call LLM
3. Parse strict JSON action response
4. Execute tool via sandbox when requested (reject unlisted tools)
5. Append tool output to context
6. Stop on `DONE`, stop request, budget exhaustion, or iteration cap

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

## TUI Usage

Open the **Agents** tab:

- choose an agent
- enter a topic
- start/pause/resume/stop
- inspect live status/log output
- review pending memory candidates
- approve/reject candidates
- edit candidate TTL (days)

## Shell Commands

- `/agent start <name> <topic>`
- `/agent status`
- `/agent pause`
- `/agent resume`
- `/agent stop`
- `/agent log`
- `/memory candidates [pending|promoted|rejected|all]`
- `/memory promote <candidate_id>`
- `/memory reject <candidate_id>`
- `/memory ttl <candidate_id> <days>`

## Notes

- Agents are optional and disabled unless `agents.enabled = true`.
- Keep sandbox paths minimal and explicit.
- Prefer dedicated low-cost service models for tool-heavy runs using `llms.services`.
