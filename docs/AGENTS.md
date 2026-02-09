# Agents Guide

This guide covers the agent system added in `0.4.0`: architecture, configuration, tools, sandboxing, and runtime usage.

## Overview

The agent runtime is implemented in `lsm/agents/` and integrated into both:

- TUI (`lsm/ui/tui/screens/agents.py`)
- Shell commands (`/agent ...` via `lsm/ui/shell/commands/agents.py`)

Built-in agent:

- `research`

## Architecture

- `lsm/agents/base.py`: `BaseAgent`, `AgentState`, lifecycle status model
- `lsm/agents/harness.py`: runtime loop, tool-calling execution, budget/iteration guards, state persistence
- `lsm/agents/models.py`: runtime message/log/response models
- `lsm/agents/log_formatter.py`: log formatting and serialization helpers
- `lsm/agents/factory.py`: registry + `create_agent(...)`
- `lsm/agents/research.py`: built-in research workflow agent

Tooling:

- `lsm/agents/tools/base.py`: `BaseTool`, `ToolRegistry`
- `lsm/agents/tools/sandbox.py`: `ToolSandbox` permission enforcement
- `lsm/agents/tools/*.py`: built-in tool implementations

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
- `agent_configs`: per-agent overrides

## Sandbox Model

Sandbox rules are defined in `agents.sandbox` and enforced by `ToolSandbox`.

- `allowed_read_paths`: readable locations for file tools
- `allowed_write_paths`: writable locations for file tools
- `allow_url_access`: permits/disallows URL tools
- `require_user_permission`: per-tool interactive gate
- `tool_llm_assignments`: optional per-tool service mapping

The sandbox is deny-by-default outside configured paths.

## Built-In Tools

Default tool registry (`create_default_tool_registry`) includes:

- `read_file`
- `read_folder`
- `write_file`
- `create_folder`
- `load_url`
- `query_llm`
- `query_remote`
- `query_remote_chain`
- `query_embeddings` (registered only when vector DB provider + embedder are available)

## Runtime Loop

`AgentHarness` loop:

1. Prepare context + tool definitions
2. Call LLM
3. Parse strict JSON action response
4. Execute tool via sandbox when requested
5. Append tool output to context
6. Stop on `DONE`, stop request, budget exhaustion, or iteration cap

State is persisted to `agents_folder` as:

- `<agent_name>_<timestamp>_state.json`

## Research Agent

The `research` agent decomposes a topic, queries available sources/tools, iteratively synthesizes findings, and writes structured output.

Use when you need multi-step retrieval + synthesis rather than a single query turn.

## TUI Usage

Open the **Agents** tab:

- choose an agent
- enter a topic
- start/pause/resume/stop
- inspect live status/log output

## Shell Commands

- `/agent start <name> <topic>`
- `/agent status`
- `/agent pause`
- `/agent resume`
- `/agent stop`
- `/agent log`

## Notes

- Agents are optional and disabled unless `agents.enabled = true`.
- Keep sandbox paths minimal and explicit.
- Prefer dedicated low-cost service models for tool-heavy runs using `llms.services`.
