# TUI Command Reference

LSM uses the TUI as the primary interactive interface. This document lists commands in Query, Ingest, and Agent workflows, along with keyboard shortcuts and side effects.

## TUI (Textual User Interface)

Launch with:

```bash
lsm
```

### TUI Structure

```text
lsm/ui/tui/
|-- __init__.py
|-- app.py                # Main LSMApp class
|-- styles/               # Split Textual CSS files (base + per-screen + widgets)
|-- completions.py        # Autocomplete logic
|-- screens/
|   |-- main.py
|   |-- query.py          # Query interface
|   |-- ingest.py         # Ingest management
|   |-- remote.py         # Remote provider workflows
|   |-- agents.py         # Interactive multi-agent workflows
|   |-- settings.py       # Configuration panel
|   `-- help.py           # Help modal
`-- widgets/
    |-- results.py
    |-- input.py
    `-- status.py
```

### TUI Keyboard Shortcuts

**Global:**
- `Ctrl+N` - Switch to Ingest tab
- `Ctrl+Q` - Switch to Query tab
- `Ctrl+R` - Switch to Remote tab
- `Ctrl+G` - Switch to Agents tab
- `Ctrl+S` - Switch to Settings tab
- `F1` - Show help modal
- `Ctrl+C` - Quit application

**Settings View:**
- `F2` - Settings Global sub-tab
- `F3` - Settings Ingest sub-tab
- `F4` - Settings Query sub-tab
- `F5` - Settings LLM sub-tab
- `F6` - Settings Vector DB sub-tab
- `F7` - Settings Modes sub-tab
- `F8` - Settings Remote sub-tab
- `F9` - Settings Chats/Notes sub-tab

**Query View:**
- `Enter` - Submit query
- `Up/Down` - Command history
- `Tab` - Autocomplete
- `Ctrl+E` - Expand selected citation
- `Ctrl+O` - Open source file
- `Escape` - Clear input

**Ingest View:**
- `Ctrl+B` - Run build
- `Ctrl+T` - Run tagging
- `Ctrl+Shift+R` - Refresh stats
- `Up/Down` - Command history
- `Tab` - Autocomplete
- `Escape` - Clear input

**Agents View:**
- `Enter` (topic input) - Start selected agent
- `F6` - Select previous running agent
- `F7` - Select next running agent
- `F8` - Approve pending interaction
- `F9` - Approve pending interaction for session
- `F10` - Deny pending interaction
- `F11` - Reply to pending clarification/feedback interaction

## Ingest Commands (TUI)

- `/info`:
  - show collection name, ID, and chunk count
- `/stats`:
  - compute detailed stats, file distributions, and error report summary
- `/explore [query]`:
  - list indexed files; supports substrings, extensions, or glob patterns
  - `--full-path` shows full prefixes instead of compact tree
- `/show <path>`:
  - show chunks for a specific file path
- `/search <query>`:
  - search metadata by path substring
- `/build [--force]`:
  - run ingest pipeline (incremental by default)
  - `--force` clears the manifest and reprocesses all files
- `/tag [--max N]`:
  - run AI tagging on untagged chunks
- `/tags`:
  - list all unique AI and user tags
- `/wipe`:
  - delete all chunks after confirmation
- `/help`:
  - show ingest help
- `/exit`:
  - exit the TUI

Side effects:

- `/build` writes to the vector DB and updates the manifest.
- `/wipe` deletes all data in the collection.
- `/tag` writes tags into chunk metadata.

## Query Commands (TUI)

- `/help`:
  - show help
- `/exit`:
  - exit the TUI
- `/show S#`:
  - show the cited chunk text
- `/expand S#`:
  - show the full chunk text without truncation
- `/open S#`:
  - open the source file using the OS default application
- `/models [provider]`:
  - list models available to the configured provider(s)
- `/model`:
  - show current model
- `/model <task> <provider> <model>`:
  - set model for a task
- `/providers`:
  - list available LLM providers
- `/provider-status`:
  - show provider health
- `/vectordb-providers`:
  - list available vector DB providers
- `/vectordb-status`:
  - show vector DB status
- `/remote-providers`:
  - list remote source providers
- `/remote-search <provider> <query>`:
  - test a remote provider
- `/remote-search-all <query>`:
  - search all configured providers
- `/mode`:
  - show current mode
- `/mode <name>`:
  - switch mode for this session
- `/mode set <setting> <on|off>`:
  - toggle mode settings
- `/note`:
  - open the last query in an editor and save as a note
- `/load <path>`:
  - pin all chunks from a file for the next query
- `/load clear`:
  - clear all pinned chunks
- `/set path_contains <substring> [more...]`:
  - set path filters for this session
- `/set ext_allow .md .pdf`:
  - set allowlist extensions
- `/set ext_deny .txt`:
  - set denylist extensions
- `/clear path_contains|ext_allow|ext_deny`:
  - clear a filter
- `/costs`:
  - show session cost summary
- `/costs export <path>`:
  - export cost data to CSV
- `/budget set <amount>`:
  - set session budget
- `/cost-estimate <query>`:
  - estimate cost without running a query
- `/export-citations [format] [note_path]`:
  - export citations (bibtex|zotero)
- `/debug`:
  - show last query diagnostics

Side effects:

- `/note` writes a Markdown file to the notes directory.
- `/model` overrides the model for the current session only.
- `/mode` overrides the mode for the current session only.
- `/load` pins chunk IDs for the next query.

## Agent Commands (TUI/REPL)

- `/agent start <name> <topic>`:
  - start an agent run
- `/agent list`:
  - list running agents and recent completed runs
- `/agent select <agent_id>`:
  - set selected agent for no-id control/status/log commands
- `/agent status [agent_id]`:
  - show run status
- `/agent pause [agent_id]`:
  - pause run
- `/agent resume [agent_id] [message]`:
  - resume run; optional message is queued first
- `/agent stop [agent_id]`:
  - request stop and wait for graceful completion
- `/agent log [agent_id]`:
  - show persisted run log
- `/agent queue [agent_id] <message>`:
  - queue user message into active run context
- `/agent interact [agent_id]`:
  - show pending interaction requests
- `/agent approve <agent_id>`:
  - approve pending permission request
- `/agent approve-session <agent_id>`:
  - approve pending permission and remember for this manager session
- `/agent deny <agent_id> [reason]`:
  - deny pending permission request
- `/agent reply <agent_id> <message>`:
  - reply to clarification/feedback request
- `/agent schedule add <agent_name> <interval> [--params '{...}'] [--concurrency_policy skip|queue|cancel] [--confirmation_mode auto|confirm|deny]`:
  - add schedule entry
- `/agent schedule list|enable|disable|remove|status`:
  - manage schedules
- `/agent meta start <goal>`:
  - start meta-agent orchestration
- `/agent meta status|log`:
  - inspect active/recent meta-agent run state

Side effects:

- `/agent start` creates run workspaces and persisted state/log artifacts under `agents.agents_folder`.
- `/agent approve-session` caches tool approval for later matching requests in the same runtime-manager session.
- live log lines stream to the Agents tab log panel via per-agent bounded queues.

## Exit Codes and Errors

- Most commands print user-facing error messages on failure.
- API errors during query fallback to local excerpts when possible.
