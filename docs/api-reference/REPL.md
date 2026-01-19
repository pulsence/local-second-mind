# TUI Command Reference

LSM uses the TUI as the only interactive interface. This document lists the available commands in the Query and Ingest tabs, along with keyboard shortcuts and side effects.

## TUI (Textual User Interface)

Launch with:

```bash
lsm
```

### TUI Structure

```
lsm/gui/shell/tui/
|-- __init__.py           # Module exports (LSMApp, run_tui)
|-- app.py                # Main LSMApp class
|-- styles.tcss           # Textual CSS styling
|-- completions.py        # Autocomplete logic
|-- screens/
|   |-- main.py           # Main layout screen
|   |-- query.py          # Query interface
|   |-- ingest.py         # Ingest management
|   |-- settings.py       # Configuration panel
|   `-- help.py           # Help modal
`-- widgets/
    |-- results.py        # Results display with citations
    |-- input.py          # Command input with history
    `-- status.py         # Status bar
```

### TUI Keyboard Shortcuts

**Global:**
- `Ctrl+I` - Switch to Ingest tab
- `Ctrl+Q` - Switch to Query tab
- `Ctrl+S` - Switch to Settings tab
- `F1` - Show help modal
- `Ctrl+C` - Quit application

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
- `Ctrl+R` - Refresh stats
- `Up/Down` - Command history
- `Tab` - Autocomplete
- `Escape` - Clear input

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
  - search all enabled providers
- `/remote-provider enable|disable|weight <name> [value]`:
  - configure remote providers
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

## Exit Codes and Errors

- Most commands print user-facing error messages on failure.
- API errors during query fallback to local excerpts when possible.
