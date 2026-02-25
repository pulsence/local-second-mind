# Agent System

Local Second Mind includes a built-in agent runtime that lets specialized LLM-driven agents work on long-running tasks: researching topics, synthesizing documents, curating knowledge, writing drafts, or orchestrating other agents. Agents run concurrently alongside your normal query workflow and can interact with you at runtime when they need input.

## How Agents Work

Each agent is an autonomous loop:

1. **Receives a topic or goal** from you at launch.
2. **Plans and executes tool calls** (read files, search the knowledge base, write outputs, etc.).
3. **Asks for clarification** when it hits ambiguity — you can approve, deny, or reply from the Agents tab or shell.
4. **Writes artifacts** (reports, notes, documents) to its workspace when done.

Agents run in a sandboxed environment. They can only read and write paths you authorize and cannot access the network unless explicitly allowed.

## Running Agents

### TUI

Switch to the Agents tab with **Ctrl+G**. From the launch panel:

1. Select an agent from the dropdown.
2. Enter a topic or goal in the topic field.
3. Press **Enter** to launch.

The **Running Agents** panel shows all active runs. Select a run to see its live log stream and status. Use the interaction panel to respond to any pending requests.

**Agents tab keyboard shortcuts:**

| Shortcut | Action |
|----------|--------|
| Enter (topic input) | Launch selected agent |
| F6 / F7 | Select previous/next running agent |
| F8 | Approve pending interaction |
| F9 | Approve-session (approve all future requests this run) |
| F10 | Deny pending interaction |
| F11 | Reply to a clarification request |

### Shell Commands

All agent operations are also available as interactive shell commands:

```text
/agent start <name> "<topic>"    Launch an agent
/agent list                      List all running agents
/agent status [agent_id]         Show status of a run
/agent log [agent_id]            Stream or view run log
/agent pause [agent_id]          Pause a running agent
/agent resume [agent_id]         Resume a paused agent
/agent stop [agent_id]           Stop a running agent
/agent interact [agent_id]       Show pending interaction request
/agent approve <agent_id>        Approve a pending request
/agent approve-session <agent_id> Approve all future requests this run
/agent deny <agent_id> [reason]  Deny a pending request
/agent reply <agent_id> <msg>    Send a reply to a clarification request
/agent queue <agent_id> <msg>    Queue a message for the next request
/agent select <agent_id>         Set the active agent for subsequent commands
```

## Available Agents

Agents are organized into four themes.

### Academic

| Name | Launch As | Purpose |
|------|-----------|---------|
| Research | `research` | Deep-dives a topic using your knowledge base and remote sources. Produces a structured research outline with citations. |
| Synthesis | `synthesis` | Reads research outputs or raw documents and writes a synthesized, coherent summary in Markdown. |
| Curator | `curator` | Reviews and filters your knowledge base for a given topic. Flags duplicates, gaps, and outdated material; produces a curation report. |

### Productivity

| Name | Launch As | Purpose |
|------|-----------|---------|
| General | `general` | General-purpose agent for multi-step tasks using local tools. |
| Librarian | `librarian` | Explores the knowledge base and builds idea graphs with metadata summaries. |
| Manuscript Editor | `manuscript_editor` | Iteratively edits manuscripts and produces revision logs. |
| Writing | `writing` | Takes a topic or outline and produces a polished long-form document (report, essay, technical writeup). |

### Meta

| Name | Launch As | Purpose |
|------|-----------|---------|
| Meta | `meta` | Orchestrator. Breaks a complex goal into a task graph and spawns sub-agents (`research`, `synthesis`, `curator`, `writing`, assistants) to execute each task, then consolidates their artifacts into a final output. |
| Assistant Meta | `assistant_meta` | Runs assistant-oriented sub-agents, validates their outputs, and produces a review summary plus action recommendations. |

Meta agents execute independent tasks in parallel (up to `agents.max_concurrent`) and write consolidated artifacts into their run workspace:

- `final_result.md`
- `meta_log.md`
- `assistant_meta_summary.md` (assistant meta only)
- `assistant_meta_summary.json` (assistant meta only)

### Assistants

| Name | Launch As | Purpose |
|------|-----------|---------|
| Assistant | `assistant` | Summarizes recent agent activity and proposes memory updates. |

## Available Tools

Every agent has access to a set of tools governed by its sandbox configuration. The tools below are registered in the default registry.

### File Navigation

| Tool | Description |
|------|-------------|
| `read_file` | Read a file's full text or a specific section by name, heading, or graph node ID. Supports `max_depth` outline, `include_hashes` for line-level hashes, and JSON structured output. |
| `read_folder` | List the contents of a directory. |
| `find_file` | Search for files by name pattern or content pattern with optional regex. Returns matching paths with structural outlines. |
| `find_section` | Locate a function, class, or heading within one or more files by name or node ID. Returns content and per-line hashes ready for `edit_file`. |
| `file_metadata` | Return size, modification time, and extension for one or more paths. Pass `include_graph: true` to include the full file graph. |
| `hash_file` | Compute a SHA-256 hash of a file's contents. |
| `source_map` | Build a source-centric map from evidence items. Returns per-source structural outlines and collects graph `node_id` references from evidence. |

### File Editing

| Tool | Description |
|------|-------------|
| `write_file` | Write or overwrite a file. |
| `append_file` | Append text to an existing file. |
| `create_folder` | Create a directory. |
| `edit_file` | Apply a precise line-range edit using start/end line hashes. Returns a diagnostic error (with context and suggestions) on hash mismatch, and the refreshed file graph outline on success. Use `find_section` or `read_file` with `include_hashes: true` to obtain the hashes first. |

### Knowledge Base

| Tool | Description |
|------|-------------|
| `query_embeddings` | Semantic search over the ingested knowledge base. Returns ranked chunks with scores. |
| `similarity_search` | Low-level vector similarity search against the active collection. |
| `extract_snippets` | Extract the most relevant text snippets for a query from the knowledge base. |

### Remote and LLM

| Tool | Description |
|------|-------------|
| `query_llm` | Call the configured LLM directly and return the response text. |
| `query_remote` | Query a configured remote provider (Brave Search, arXiv, Semantic Scholar, etc.) and return structured results. |
| `query_remote_chain` | Run a remote provider query and pass the results through the LLM for synthesis. |
| `load_url` | Fetch the text content of a URL. |

### Memory

| Tool | Description |
|------|-------------|
| `memory_put` | Store a key/value entry in the agent's persistent memory. |
| `memory_search` | Search agent memory by semantic similarity. |
| `memory_remove` | Delete a memory entry by key. |

### Meta / Orchestration

| Tool | Description |
|------|-------------|
| `spawn_agent` | Spawn a sub-agent run (used by the `meta` agent). |
| `await_agent` | Wait for a spawned sub-agent to complete and retrieve its result. |
| `collect_artifacts` | Collect artifact files written by a completed sub-agent run. |
| `ask_user` | Request clarification or approval from the user at runtime. Always available to every agent regardless of sandbox configuration. |

### Execution

| Tool | Description |
|------|-------------|
| `bash` | Execute a bash command string. Respects sandbox command allow/deny lists and blocks command chaining. |
| `powershell` | Execute a PowerShell command string. Respects sandbox command allow/deny lists and blocks command chaining. |

## Agent Configuration

Agents are configured under the `"agents"` section of your `config.json`:

```json
{
  "agents": {
    "enabled": true,
    "agents_folder": "agents",
    "max_iterations": 50,
    "interaction": {
      "auto_continue": false
    },
    "memory": {
      "enabled": false,
      "storage_backend": "auto"
    },
    "sandbox": {
      "execution_mode": "local_only",
      "allowed_read_paths": [],
      "allowed_write_paths": [],
      "allow_url_access": false
    }
  }
}
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `false` | Master switch for the agent system. |
| `agents_folder` | `"agents"` | Root directory for agent workspaces, logs, and schedules. |
| `max_iterations` | `50` | Maximum tool-call iterations per agent run before forced stop. |
| `interaction.auto_continue` | `false` | When `true`, `ask_user` requests are automatically approved with "Continue with your best judgment." Useful for unattended runs. |
| `memory.enabled` | `false` | Enable persistent agent memory across runs. |
| `memory.storage_backend` | `"auto"` | Memory backend: `auto` (SQLite), `sqlite`, or `postgresql`. |

### Per-Agent Overrides

Override `max_iterations` or disable specific agents via `agent_configs`:

```json
{
  "agents": {
    "agent_configs": {
      "research": { "max_iterations": 100 },
      "curator": { "enabled": false }
    }
  }
}
```

## Workspace Structure

Each agent run gets an isolated workspace directory:

```
<agents_folder>/
  <agent_name>/
    logs/         Run logs (JSON)
    artifacts/    Output files written by the agent
    memory/       Persistent memory storage (if enabled)
```

File tools default to the agent's workspace root when given relative paths. Absolute paths are also accepted if they fall within the sandbox's allowed read/write paths.

## Sandbox and Permissions

Each agent runs inside a `ToolSandbox` that enforces path and network boundaries:

- **Read paths** — Directories the agent may read from. Add your knowledge base roots here if agents need to read your documents directly.
- **Write paths** — Directories the agent may write to. The workspace directory is always included.
- **URL access** — Disabled by default. Enable to allow `load_url` and remote provider tools.
- **Shell command lists** — Use `command_allowlist`/`command_denylist` to permit or block specific commands for `bash`/`powershell`.
- **Tool allowlist** — Restrict which tools a given agent can call. `ask_user` is always available regardless of the allowlist.
- **Permission prompts** — Tools marked `writes_workspace` (e.g. `write_file`, `edit_file`) will prompt for approval unless `require_permission_by_risk` is set to `false`.

When an agent requests a tool that is not in its allowlist or outside its sandbox boundaries, the request is denied and the agent receives an error it can report back to you.

## Artifacts and Logs

After a run completes:

- **Artifacts** are written to `<agents_folder>/<agent_name>/artifacts/`. Common outputs include `research_outline.md`, `deliverable.md`, and `curation_report.md`.
- **Logs** are stored in `<agents_folder>/<agent_name>/logs/` as structured JSON. View them with `/agent log <agent_id>` or open the file directly.
- A **run summary** is emitted at the end of every run with tool usage counts, approval/denial tallies, duration, and outcome.

## Scheduling Agents

Agents can be scheduled to run at fixed intervals. Schedule configuration is stored in `<agents_folder>/schedules.json` and managed from the Agents tab in the TUI or via shell commands.

## See Also

- [Getting Started](GETTING_STARTED.md) — First run walkthrough including launching your first agent.
- [CLI Usage](CLI_USAGE.md) — Full shell command reference.
- [Configuration](CONFIGURATION.md) — All configuration options including the `agents` section.
- [Remote Sources](REMOTE_SOURCES.md) — Setting up remote providers for agents to query.
