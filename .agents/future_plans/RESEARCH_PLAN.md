# v0.9.0 UI Rework — Research Plan

## Overview

v0.9.0 is a significant UX overhaul with four distinct concerns:

1. **General housekeeping** — consolidate duplicated logging/path modules and clarify module boundaries.
2. **Web UI** — introduce a browser-based primary interface backed by an HTTP server.
3. **TUI simplification** — strip the TUI down to an admin/operator-focused command-prompt experience.
4. **CLI + TUI unification** — eliminate the separate CLI shell layer; everything routes through the TUI.

Each concern is analysed independently, then integration considerations are drawn at the end.

---

## 1. General Housekeeping

### 1.1 Logging Consolidation

**Current state**

Two separate logging implementations exist today:

| Module | Purpose |
|--------|---------|
| `lsm/logging.py` | Python `logging`-based structured logger. Used everywhere (`get_logger`, `setup_logging`, `ColoredFormatter`). |
| `lsm/utils/logger.py` | Lightweight `PlainTextLogger` / `LogVerbosity` for scripts and ingest pipelines that write plain text to a stream or file. |

These serve genuinely different use cases — structured Python logging vs. simple plain-text progress output — but the names and locations make the split non-obvious and have started to create confusion about which one to reach for.

**Consolidation options**

- **Option A (recommended):** Keep the structured `logging`-based implementation as the authoritative system logger under `lsm.logging`. Move `PlainTextLogger` into `lsm.logging` as a companion utility class (e.g., `lsm.logging.plain`). Delete `lsm/utils/logger.py`. Update all imports.
- **Option B:** Promote `lsm.utils.logger` to `lsm.logging.plain_text` and leave the Python `logging` wrapper in `lsm.logging.structured`. Adds a subpackage but preserves the distinction more explicitly.
- **Option C:** Delete `PlainTextLogger` entirely and replace its usages with a thin wrapper around standard Python logging that writes to `sys.stdout` without formatting.

**Log persistence to `<GLOBAL_FOLDER>/Logs/`**

`lsm/logging.py:setup_logging()` already supports an optional `log_file` argument, but the caller (`__main__.py:configure_logging_from_args`) receives the file path from CLI `--log-file`. There is no automatic routing to `<GLOBAL_FOLDER>/Logs/`.

Changes needed:
- `ensure_global_folders()` in `lsm/paths.py` should also create `<GLOBAL_FOLDER>/Logs/`.
- `setup_logging()` should accept a `global_folder` parameter and derive a timestamped or rotated log file path automatically when no explicit `log_file` is given and a global folder exists.
- Consider using Python's `logging.handlers.RotatingFileHandler` or `TimedRotatingFileHandler` to avoid unbounded log growth.
- The TUI app startup should call `setup_logging(global_folder=config.global_settings.global_folder)`.

### 1.2 Path Consolidation

**Current state**

Two path helper modules exist:

| Module | Contents |
|--------|---------|
| `lsm/paths.py` | `get_global_folder`, `get_chats_folder`, `get_notes_folder`, `resolve_relative_path`, `ensure_global_folders` — global folder awareness. |
| `lsm/utils/paths.py` | `resolve_path`, `resolve_paths`, `canonical_path`, `safe_filename` — generic, stateless path utilities. |

The split is logical — the first is domain-specific, the second is generic — but having two modules named `paths` creates import confusion.

**Consolidation options**

- **Option A (recommended):** Merge both into `lsm/utils/paths.py` (which already exists), adding the global-folder functions there. Delete `lsm/paths.py` and redirect any imports. `lsm.paths` becomes an alias shim for one version, then removed.
- **Option B:** Keep `lsm/paths.py` as the single authoritative paths module (domain-aware) and merge the generic helpers from `lsm/utils/paths.py` into it. Delete `lsm/utils/paths.py`.
- **Option C:** Rename `lsm/paths.py` → `lsm/global_paths.py` to make domain specificity explicit, keep `lsm/utils/paths.py` as the generic layer.

**User Feedback:** Variant of option a. Merge both into `lsm/utils/paths.py` and migrate ALL references to `lsm/utils/paths.py`. There should be no alias for `lsm/paths.py`.

Migration scope: `get_global_folder` is imported in `lsm/config/loader.py`, `lsm/ui/tui/app.py`, `lsm/db/migration.py`, and likely several test files. A global grep is needed.

### 1.3 `lsm.vectordb` → `lsm.db` Subpackage

**Current state**

`lsm.vectordb` is a standalone package (evident from its documentation and test infrastructure). `lsm.db` already exists and manages schema, migration, enrichment, clustering, and job status.

**Proposed change:** `lsm.vectordb` becomes `lsm.db.vectordb`. The existing `lsm.db` package becomes a proper parent with `lsm.db.vectordb`, `lsm.db.migration`, `lsm.db.enrichment`, etc.

**Migration concerns:**
- All `from lsm.vectordb import ...` usages must be updated.
- Public API re-export can be provided from `lsm.db.vectordb.__init__` to smooth the transition.
- A deprecation shim at `lsm/vectordb/__init__.py` can forward imports with a `DeprecationWarning` for one release.
- Test suite has `tests/vectordb/` — would need renaming or path adjustment.

**User Feedback:** No re-exporting or shims should be used. This is all internal code, just fix all the references.

### 1.4 `lsm.finetune` Purpose Review

From the CLI (`__main__.py`) and `lsm/ui/shell/cli.py`, `lsm.finetune` provides:

- `extract_training_pairs(conn)` — mines query↔chunk pairs from the corpus.
- `finetune_embedding_model(pairs, base_model, output_path, epochs)` — runs Sentence-Transformers fine-tuning.
- `register_model`, `set_active_model`, `get_active_model`, `list_models` — model registry in SQLite.

It is exposed via `lsm finetune train|list|activate` CLI subcommands.

**Verdict:** This is legitimate core functionality for improving retrieval quality by customising the embedding model on the user's own corpus. It does not appear to be dead code. Whether it belongs under a different subpackage (e.g., `lsm.db.finetune` if it is heavily coupled to the DB, or `lsm.ml.finetune`) is a housekeeping question rather than a removal question.

---

## 2. Web UI

### 2.1 Goals and Non-Goals

**Goals:**
- Become the primary everyday interface for query and agent interaction.
- Standardised chat UI: scrollable message panel in the centre, fixed input box at the bottom, collapsible navigation sidebar on the left.
- Four top-level navigation sections: **Query** (chat), **Ingest**, **Agents**, **Settings** (includes Remote Sources and Remote Chains configuration).
- A dedicated **Remote Sources & Chains** query screen.
- The server must also expose an API endpoint usable by a future Obsidian plugin.

**Non-goals (for v0.9.0):**
- Remote access / authentication (v1.1.0).
- Mobile responsiveness (can be deferred).

### 2.2 Technology Stack Options

The choice of server framework and frontend approach are the most consequential decisions.

#### 2.2.1 Backend Server

| Option | Pros | Cons |
|--------|------|------|
| **FastAPI** | Modern async Python, automatic OpenAPI docs, WebSocket and SSE support, excellent type integration, widely used | New dependency, requires some JS for a reactive frontend |
| **Flask** | Simpler, smaller, mature | Sync by default, no built-in SSE/WebSocket ergonomics |
| **Starlette (bare)** | FastAPI's foundation, minimal | No automatic docs, FastAPI is just as easy |
| **aiohttp** | Full async, but more verbose | Steeper learning curve |

**Recommendation:** FastAPI. It is async-native (matching the existing Textual/asyncio patterns), produces automatic API docs that will be needed for the Obsidian plugin, and handles streaming responses via SSE or WebSockets naturally.

**User Feedback:** FastAPI seems to be the best, we are staying in Python which is a major bonus.

#### 2.2.2 Frontend Approach

| Option | Pros | Cons |
|--------|------|------|
| **HTMX + Jinja2 templates** | No build toolchain, Python-rendered, small JS surface, server-side streaming via SSE/chunked | Less reactive, SSE partial updates require HTMX extensions |
| **React (Vite/CRA)** | Mature ecosystem, component model, streaming via SSE hooks, easy chat UI patterns | Requires Node.js build step, separate dev/prod workflow |
| **Svelte (SvelteKit)** | Smaller bundle, simpler reactivity, good SSE support | Smaller community |
| **Vue 3** | Familiar to many devs, Composition API, good SSE handling | Middle ground — not as lightweight as HTMX, not as full-featured as React |
| **Reflex / NiceGUI** | Pure Python, no JS | Heavy opinion layer, may restrict UI flexibility |

**Recommendation:** Two viable paths:
- **Path A (lower complexity):** HTMX + Jinja2 + FastAPI. Server renders templates, HTMX handles partial-page swaps and SSE streaming. No Node.js build step. Well-suited for a local tool with one user.
- **Path B (higher capability):** FastAPI backend + lightweight SPA frontend (Vue 3 or Svelte). Better component isolation, easier streaming UX, natural chat message rendering. Requires a build step.

**User Feedback:** We will go with HTMX + Jinja2 + FastAPI

Given this is a local-first tool and future API exposure (Obsidian plugin) is an explicit goal, **Path A (HTMX + Jinja2)** keeps the codebase purely Python and avoids a JS build toolchain. However, **Path B** will be more maintainable as the UI grows.

The choice should be resolved before implementation begins (see Clarifications section).

#### 2.2.3 Real-Time Streaming

Query responses are streamed from LLMs. The Web UI must handle this gracefully.

Options:
- **Server-Sent Events (SSE):** Simple, HTTP/1.1 compatible, browser built-in `EventSource`, one-directional (server → client). Good for streaming query responses. FastAPI supports SSE via `StreamingResponse` with `text/event-stream` content type.
- **WebSockets:** Bi-directional, slightly more complex. Needed if the browser sends incremental input (e.g., live typing suggestions). For a chat UI, SSE is sufficient.

**Recommendation:** SSE for query streaming responses. WebSockets can be added later if live bi-directional features are needed.

**User Feedback:** If SSE is suffient then no need to make it complicated.

### 2.3 Web Server Lifecycle

**How does the server start?**

Currently `python -m lsm` (no subcommand) launches the TUI. With v0.9.0, this needs to accommodate the web server.

Options:
- `lsm serve` — new subcommand that starts the FastAPI server. The TUI remains the admin UI, launched via `lsm tui`.
- `lsm` with no arguments starts the server (browser is primary UI now).
- Configuration flag `ui = "web" | "tui"` in config.json.

**User Feedback:** No ui config flag in config.json, since this handles launching the overall application. 
Instead there should be a `lsm cli` subcommand that starts the TUI.

The server would bind on `127.0.0.1` (localhost only) and a configurable port (default: 8080). Browser auto-open is optional.

**Co-running TUI and server:**

The TODO notes the TUI should be able to run alongside the server. This means:
- The HTTP server runs in a background asyncio task or a separate thread.
- The TUI continues to run in the foreground Textual event loop.
- Shared state (config, query provider, agent runtime) must be thread-safe, following existing `AppState` patterns.

Alternatively, the server and TUI are separate processes coordinated via the shared SQLite database, which is already the pattern for persistence.

**User Feedback:** The alternative server/TUI separate processes is the prefered way. 

### 2.4 Web UI Screen Design

#### Query Screen (Primary Chat Interface)

```
+--sidebar--+----main panel--------------+
| Navigation|  [conversation history]    |
|           |  User: what is X?          |
|  > Query  |  LSM: [answer text]        |
|  Ingest   |  [citation cards]          |
|  Agents   |                            |
|  Settings |  [streaming indicator]     |
|           |-----------------------------|
| [collapse]|  [input box]  [send btn]   |
+-----------+-----------------------------+
```

Features:
- Chat history persisted per mode (existing `Chats/` folder).
- Mode selector (grounded / insight / hybrid / chat).
- Citation cards rendered inline, expandable.
- Cost display.

#### Ingest Screen

- Show ingest stats (chunk count, last run, files indexed).
- Trigger ingest build/tag via button (POST to `/api/ingest/build`).
- Progress display via SSE.

**User Feedback:** ALL ingest related commands should be available here.


#### Agents Screen

- List agent schedules and running agents.
- Start / pause / stop controls.
- Streaming log view via SSE.

#### Settings Screen

- Sections matching existing TUI tabs: Global, LLM, VectorDB, Query, Modes, Ingest, Notes/Chats, Remote Providers, Remote Chains.
- Live validation feedback.
- Save/reset controls.

#### Remote Sources & Chains Query Screen

- Select remote providers/chains.
- Execute queries against them.
- Display results.

#### Admin Screen
**User Feedback:** Need to also consider how to implement things like running evaluation testing, migration
and all the other commands available commands and see the system health and statistics.

### 2.5 API Surface

A minimal REST API that both the Web UI and the future Obsidian plugin will consume:

```
POST   /api/query                     # Submit a query, returns SSE stream
GET    /api/query/candidates          # Get last result candidates
POST   /api/ingest/build              # Trigger ingest build, returns SSE stream
POST   /api/ingest/tag
GET    /api/ingest/stats
GET    /api/agents                    # List agents
POST   /api/agents/{name}/start
POST   /api/agents/{name}/stop
GET    /api/agents/{name}/logs        # SSE log stream
GET    /api/config                    # Read current config
PUT    /api/config                    # Write config changes
GET    /api/health                    # System health
```

All endpoints return JSON. Streaming endpoints use `text/event-stream`.

### 2.6 Static Asset Serving

FastAPI can serve static files via `StaticFiles`. Templates go under `lsm/ui/web/templates/`, static assets (CSS, JS) under `lsm/ui/web/static/`. This keeps the web UI self-contained within the existing `lsm.ui.web` package.

### 2.7 Dependencies to Add

Minimum new dependencies:
```
fastapi>=0.110
uvicorn[standard]>=0.29    # ASGI server
jinja2>=3.1                # If using server-side templates
```

For HTMX path, no additional JS dependencies. For SPA path, a `package.json` in `lsm/ui/web/frontend/` with a build step.

---

## 3. TUI Simplification

### 3.1 Design Direction

The TODO specifies a single three-zone display:

```
----------------------------------------
| Formatted Text Content               |
| for the current screen               |
|                                      |
|--------------------------------------|
| Prompt for user type commands        |
|--------------------------------------|
| Short cuts for current display       |
----------------------------------------
```

All user input comes from the command prompt. All output goes to the text display. No buttons.

### 3.2 Current TUI vs Target TUI

| Aspect | Current | Target |
|--------|---------|--------|
| Primary interaction | Mouse + keyboard hybrid | Keyboard-first REPL |
| Buttons | Present (Build, Tag, etc.) | None |
| Tabs | 5 tabs (Query, Ingest, Remote, Agents, Settings) | Settings tab + single REPL screen |
| Query/Ingest/Agents/Remote | Separate tab widgets | Folded into one `CommandScreen` REPL |
| Settings | Tab-based GUI | Retained (keyboard-nav only, no mouse buttons) |

### 3.3 Target Screen Architecture

#### 3.3.1 The `CommandScreen` (replaces Query/Ingest/Remote/Agents tabs)

A unified REPL widget:

```python
class CommandScreen(ManagedScreenMixin, Widget):
    """
    Unified command REPL screen.

    Processes all commands and displays results.
    Replaces the separate Query, Ingest, Remote, and Agents tabs.
    """
```

- Top area: `ScrollableContainer` with `RichLog` or `Static` for output.
- Middle: `CommandInput` (existing widget, reused).
- Bottom: `Static` shortcut bar (context-sensitive, updates on each command execution).

Commands routed through a single dispatch table:
- Query commands (existing `execute_query_command`).
- Ingest commands (existing `handle_ingest_command`).
- Agent commands.
- Remote commands.
- New `health` command.
- New `log` command.

#### 3.3.2 `SettingsScreen` Retained

The Settings screen is the only tab retained because it has genuine visual form structure that benefits from widget composition. Settings should:
- Remove any remaining Button widgets and replace with keyboard-nav `Input` fields.
- Bind F2-F9 for tab navigation (already partially done).
- Make all actions triggerable by keyboard.

#### 3.3.3 Tab Structure

From 5 tabs → 2 tabs:

| Old Tab | New Tab |
|---------|---------|
| Query | `Command` (REPL) |
| Ingest | (folded into `Command`) |
| Remote | (folded into `Command`) |
| Agents | (folded into `Command`) |
| Settings | `Settings` (retained) |

Keybindings: `Ctrl+C` → Command screen, `Ctrl+S` → Settings screen (or similar).
**User Feedback:** Use `Ctrl+H` for Command screen

### 3.4 New TUI Commands

#### `health`

Shows status/health of all systems:

```
$ health
System Health
=============
Database       [OK]   SQLite @ ~/Local Second Mind/lsm.db (1,234,567 chunks)
Embedding      [OK]   all-MiniLM-L6-v2 (384d), device=cpu
LLM Provider   [OK]   openai/gpt-4o
Remote         [OK]   3 providers configured
Agents         [OK]   0 running, 2 scheduled
Logs           [OK]   ~/Local Second Mind/Logs/lsm-2026-03-02.log
```

Implementation: call `lsm.db.health.check_db_health`, provider health checks, embedding model probe.

**User Feedback:** Should also test polling the server to see its status

#### `log`

Shows recent log entries from `<GLOBAL_FOLDER>/Logs/`:

```
$ log
$ log 50          # last 50 lines
$ log --level WARNING
$ log --follow    # tail -f style (poll loop until Ctrl+C)
```

Reads from the persisted log file, not just the in-memory TUI buffer.

### 3.5 Removing Buttons

Current button inventory (to audit and remove):

| Screen | Button | Replacement |
|--------|--------|-------------|
| Ingest | Build | `/build` command |
| Ingest | Tag | `/tag` command |
| Ingest | Wipe | `/wipe` command |
| Settings | Save | Enter key / `/save` command |
| Settings | Reset | `/reset` command |
| Agents | Start | `/agent start <name>` |
| Agents | Pause | `/agent pause` |
| Agents | Stop | `/agent stop` |

### 3.6 Co-running with Web Server

The TUI is reduced to admin scope. If the web server is running, the TUI can connect to the same SQLite database and shared state without conflict because:
- The web server handles user queries via its own asyncio loop.
- The TUI reads logs and shows health without modifying query state.
- Only Settings writes config — a file write, not a DB write.

If the web server and TUI need to coordinate (e.g., TUI triggers an ingest while web server is active), this can be mediated through the DB job status table.

---

## 4. CLI + TUI Unification

### 4.1 Current Separation

Currently:
- `lsm` (no args) → TUI (`lsm.ui.tui.app`)
- `lsm ingest build` → CLI shell (`lsm.ui.shell.cli`)
- `lsm query` → no separate subcommand exists (query happens in TUI or via REPL)

The `lsm.ui.shell` package contains `cli.py` (ingest/db/migrate commands) and `commands/` (agent commands shared with TUI). The TUI internally uses the same handlers.

### 4.2 Unification Target

With v0.9.0:
- Single entry point: `lsm` → starts web server (or TUI with `--tui` flag).
- All single-shot operations (`ingest build`, `ingest tag`, `db prune`, etc.) remain as CLI subcommands — these are non-interactive batch operations that naturally belong on the CLI.
- The interactive "query REPL" that was previously a hidden mode is folded into the simplified TUI.
- No separate "CLI shell" layer exists for interactive use.

**Concrete changes:**
- `lsm.ui.shell.unified` (currently the REPL entry) can be deleted or folded into the TUI's `CommandScreen`.
- `lsm.ui.shell.commands/` command handlers are shared between the TUI's `CommandScreen` and the CLI (no duplication needed — already factored through `lsm.ui.helpers.commands`).
- The `--tui` flag on `lsm serve` (or a `lsm tui` subcommand) starts the Textual admin TUI.

---

## 5. Module Structure After v0.9.0

```
lsm/
├── __main__.py              # Entry point: `lsm serve` | `lsm tui` | `lsm ingest ...`
├── logging.py               # Consolidated (absorbs utils/logger.py plain-text helpers)
├── paths.py                 # Kept (or merged into utils/paths.py)
├── utils/
│   ├── paths.py             # Generic path helpers (merged or kept)
│   ├── text_processing.py
│   └── file_graph.py
├── config/                  # Unchanged
├── ingest/                  # Unchanged
├── query/                   # Unchanged
├── providers/               # Unchanged
├── remote/                  # Unchanged
├── db/                      # Expanded
│   ├── vectordb/            # Moved from lsm.vectordb
│   ├── migration.py
│   ├── enrichment.py
│   └── ...
├── agents/                  # Unchanged
├── finetune/                # Unchanged (or moved to db/finetune or ml/finetune)
└── ui/
    ├── web/
    │   ├── app.py           # FastAPI application factory
    │   ├── routes/
    │   │   ├── query.py
    │   │   ├── ingest.py
    │   │   ├── agents.py
    │   │   ├── config.py
    │   │   └── health.py
    │   ├── templates/       # Jinja2 templates (if HTMX path)
    │   ├── static/          # CSS, JS, HTMX
    │   └── __init__.py
    ├── tui/
    │   ├── app.py           # Simplified: 2 tabs
    │   ├── screens/
    │   │   ├── command.py   # New unified REPL screen
    │   │   ├── settings.py  # Retained, keyboard-nav cleaned up
    │   │   ├── help.py      # Retained
    │   │   └── base.py      # Retained
    │   ├── widgets/         # Existing, mostly retained
    │   ├── state/           # Retained
    │   ├── presenters/      # May simplify
    │   └── styles/          # 2 CSS files remain (command.tcss, settings.tcss)
    ├── shell/
    │   ├── cli.py           # Retained: batch CLI commands only
    │   └── commands/        # Retained: shared command handlers
    └── helpers/
        └── commands/        # Retained: shared command parsing
```

**User Feedback:** lsm.ui.shell must be entirely rolled into lsm.ui.tui.

---

## 6. Testing Considerations

### 6.1 Web UI Tests

- **Unit tests:** FastAPI route handlers tested with `httpx.AsyncClient` and `TestClient`.
- **SSE tests:** Mock the streaming generator; verify event format.
- **Contract tests:** Verify API request/response schemas match the documented surface.
- **Integration tests:** Spin up a real in-memory server, issue a real query, verify streaming response.

### 6.2 TUI Tests

- Existing test patterns (fake widget doubles) can adapt to the simplified `CommandScreen`.
- The REPL dispatch table is the primary test surface — test each command class.
- `health` command: mock `check_db_health`, verify output formatting.
- `log` command: mock file read, verify tail/follow behaviour.

### 6.3 Logging Merge Tests

- Test that `setup_logging(global_folder=...)` creates the log file under `<GLOBAL_FOLDER>/Logs/`.
- Test that `PlainTextLogger` (merged) still works when imported from new location.
- Verify no import breakage (via a smoke test that imports every `lsm.*` module).

### 6.4 Path Merge Tests

- All existing path helper tests must pass from the merged module location.
- Verify deprecation shim (if used) emits `DeprecationWarning`.

---

## 7. Implementation Sequencing

Suggested order within v0.9.0:

1. **Logging consolidation** (lowest risk, enables log persistence for everything else).
2. **Log persistence** to `<GLOBAL_FOLDER>/Logs/`.
3. **Path consolidation** (low risk, pure rename/merge).
4. **Web server scaffold** — FastAPI app factory, `lsm serve` subcommand, health endpoint.
5. **Web UI query screen** — SSE streaming, chat history.
6. **Web UI ingest screen** — trigger ingest, SSE progress.
7. **Web UI agents screen** — list, start/stop, SSE logs.
8. **Web UI settings screen** — config read/write.
9. **TUI simplification** — collapse 5 tabs → 2 tabs, `CommandScreen` REPL.
10. **Remove TUI buttons** — audit and replace with commands.
11. **`health` command** in TUI.
12. **`log` command** in TUI.
13. **`lsm.vectordb` → `lsm.db.vectordb` rename** (high blast radius, do last).
14. **`lsm.finetune` placement decision** (review after 13).

---

## 8. Risk Areas

| Risk | Mitigation |
|------|-----------|
| `lsm.vectordb` rename touches many import sites | Do last; use deprecation shim for one release cycle |
| Web UI framework choice locks in a technology | Decide before any implementation; document the choice |
| Streaming responses in web UI are tricky to test | Write SSE test helpers early; standardise event format |
| TUI simplification may break existing TUI tests | Run full TUI test suite after each screen removal |
| Log file growth with `RotatingFileHandler` | Cap file size at 10 MB, keep 5 rotations |
| Web server port conflict | Make port configurable; default 8080 |
| TUI + web server thread safety | Use existing `AppState`/snapshot patterns; no shared mutable objects across threads |

---

## Clarifications Required

1. **Web UI framework choice:** Do you prefer the pure-Python HTMX + Jinja2 approach (no JS build step) or a SPA frontend (Vue 3 / Svelte / React) with a FastAPI JSON backend? The choice affects maintainability, the feel of the UI, and the developer workflow significantly.

**User Feedback:** See comments above.

2. **Web server startup mode:** When the user runs `lsm` with no arguments in v0.9.0, should it start the web server and open a browser, start the TUI, or show a help message? What is the default launch experience?

**User Feedback:** Start server and show address to access server once fully loaded

3. **`lsm serve` vs `lsm tui` entry points:** Should `lsm` (no args) become `lsm serve` and the TUI be accessed via a separate `lsm tui` command? Or should `lsm` remain the TUI launcher and the server be explicitly started?

**User Feedback:** `lsm` should default to starting the server

4. **TUI + web server co-running:** Should the simplified TUI be capable of starting the embedded web server in a background thread (so one `lsm tui` run gives you both), or are they always separate processes?

**User Feedback:** Always separate processes

5. **Settings screen in Web UI:** The TODO says Remote Sources and Remote Chains configuration should be in the Web UI Settings screen. Should the Web UI Settings screen completely replace the TUI Settings screen as the authoritative config editor, or should both be kept in sync?

**User Feedback:** Both should read and write from the same file. It does not matter which config editor the
user uses.

6. **Logging merge strategy:** Should `PlainTextLogger` from `lsm.utils.logger` be absorbed into `lsm.logging` (Option A), promoted to `lsm.logging.plain` (Option B), or deleted and replaced with standard logging (Option C)?

**User feedback:** What is the benefit of having two separate loggers? It duplicates code, it makes it complicated
to debug when logging messages are inconsistent. Having a single logger that takes in logs and then enables those
logs to be output via CLI/File/Consummable Emiter/etc seems to be the best long-term strategy.

7. **Path merge strategy:** Should `lsm/paths.py` and `lsm/utils/paths.py` be merged into `lsm/utils/paths.py` (Option A), into `lsm/paths.py` (Option B), or separated with a clearer naming convention (Option C)?

**User Feedback:** See answers above.

8. **`lsm.vectordb` → `lsm.db.vectordb` timing:** This is a high-blast-radius rename. Should it be done within v0.9.0 or deferred to the v0.10.0 refactor cycle?

**User Feedback:** No deferring features.

9. **`lsm.finetune` location:** Now that you've reviewed what it does (embedding fine-tuning + model registry), does it stay at `lsm.finetune`, move to `lsm.db.finetune`, or to an `lsm.ml` package?

**User Feedback:** Keep as it is.

10. **Web UI chat history persistence:** Should the Web UI use the existing `<GLOBAL_FOLDER>/Chats/` folder structure (markdown transcripts) for chat history, or introduce a DB-backed conversation store?

**User Feedback:** Which long term will be more maintainable? Moving the chats into a db or keeping them as
flat files. Eventually we will want to include the converstations in the query pipeline.

11. **Port and host defaults for the web server:** Is `localhost:8080` an acceptable default, or is there a preference for a different port? Should there be an `--open` flag to auto-open the browser on startup?

**User Feedback:** No need for auto-open. That is an acceptable default, the user should then specify if they
care otherwise in a config `server` object.

12. **Authentication / access control:** For v0.9.0 (localhost only), is no authentication acceptable? The server will only bind to `127.0.0.1`, but if the user has untrusted processes on the same machine this could be a concern.
**User Feedback:** This is fine for right now.

**User Feedback:** No shims or backwards compatability code must be retained. There is no reason for their to
ever be shims or backwards compatability unless the user explicitly states so.