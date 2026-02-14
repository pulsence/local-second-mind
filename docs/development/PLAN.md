# Local Second Mind v0.6.0 Development Plan: TUI Improvements

## Context

v0.5.0 delivered a comprehensive agent system, but the TUI has accumulated significant technical debt:
the Settings screen is a 1,310-line monolith mixing UI rendering with config mutation; the Agent screen
has no mechanism for interactive agent communication (permission prompts crash the agent); CSS sizing is
too large for standard Windows command prompts; and only one agent can run at a time. This version
addresses all of these issues plus several additional UX improvements.

**Ordering rationale**: CSS/scaling fixes come first since they affect every screen and are
low-risk. Settings MVC refactoring is second as it's the largest structural change but doesn't affect
runtime behavior. The interactive agent system is third because it requires changes to the harness,
sandbox, runtime manager, and TUI simultaneously. Additional improvements are last as polish items.

---

## Phase 1: UI Scaling & Compact Layout (COMPLETED)

The TUI uses oversized defaults (3-line tabs, 3-line min-height fields, generous padding) that consume
too much vertical space on a standard 24-30 row Windows command prompt. This phase reduces element sizes
and adds a compact layout mode.

### 1.1: Audit and Reduce Default Sizes (COMPLETED)

Reduce the size of all oversized elements in the CSS. The goal is to make the TUI comfortably usable in
an 80x24 terminal window.

**Tasks:**
- Reduce `ContentTabs` height from `3` to `2`
- Reduce `Tab` height from `3` to `2`, reduce padding from `1 2` to `0 2`
- Reduce `TabPane` padding from `1 2` to `0 1`
- Reduce `.settings-field` min-height from `3` to `2`
- Reduce `.settings-field Input` height from `3` to `1`
- Reduce `.settings-field Select` height from `3` to `1`
- Reduce `.settings-actions` height from `3` to `auto`
- Reduce `CommandInput` min-height from `3` to `1`
- Reduce `#query-command-input` min-height from `3` to `1`
- Reduce `#settings-screen` padding from `2` to `1`
- Reduce `.settings-panel` padding from `1 2` to `0 1`
- Reduce `.settings-section` margin-bottom from `2` to `1`, padding from `1` to `0 1`
- Reduce `.command-input` min-height from `3` to `1`
- Reduce `.bottom-pane` min-height from `3` to `1`
- Reduce `ProgressBar` padding from `1` to `0 1`
- Reduce `#remote-controls Input` and `Select` height from `3` to `1`
- Scope compact `height: 1` rules to compact containers/classes only (do not apply globally)
- Verify Textual minimum widget sizes don't conflict (some widgets need at least height 3
  for border rendering — test each change)

**Files to modify:**
- `lsm/ui/tui/styles.tcss`

**Post-block:** Manual testing on 80x24 terminal, adjust any clipping. Tests in
`tests/test_ui/tui/test_compact_layout.py`, run `pytest tests/ -v`, update docs, commit.

### 1.1.1: Add Density Mode (Compact vs Comfortable vs Auto) (COMPLETED)

The plan should provide a true mode toggle, not only hard-coded compact defaults.

**Tasks:**
- Add a density setting (`auto` | `compact` | `comfortable`) for TUI layout behavior
- Support runtime toggle command (for example `/ui density auto|compact|comfortable`)
- In `auto` mode, detect terminal size and apply compact mode when dimensions are below thresholds
  (for example width <= 100 or height <= 32), otherwise use comfortable mode
- Re-evaluate density on terminal resize events with hysteresis/debounce to avoid rapid flip-flopping
- Persist density selection in config (or explicitly document it as session-only if not persisted)
- Apply compact-only size reductions through a root class/state, leaving comfortable defaults available
- Ensure manual mode selection (`compact`/`comfortable`) overrides auto-detection until switched back to `auto`
- Add terminal width/height breakpoint rules for narrow terminals (single-column fallback where needed)

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/styles.tcss` (or split style files after Phase 1.3)
- `lsm/config/models/` and `lsm/config/loader.py` (if persisted)
- `example_config.json` (if persisted)

**Post-block:** Add tests for density toggle, auto-detection, resize behavior, and responsive fallback,
run `pytest tests/ -v`, commit.

### 1.2: Add CSS for Agents Screen (COMPLETED)

The agents screen currently has no dedicated CSS section in `styles.tcss` (unlike query, ingest,
settings, remote which all have sections). This causes reliance on generic styles and inconsistent
appearance.

**Tasks:**
- Add `#agents-layout`, `#agents-top`, `#agents-left`, `#agents-control-panel` layout rules
- Add `#agents-log-panel` styling (match `#query-log-panel` pattern)
- Add `.agents-section-title`, `.agents-label` styling
- Add `#agents-buttons`, `#agents-schedule-buttons`, `#agents-memory-buttons`, `#agents-meta-buttons`
  button row styling (horizontal layout, compact height)
- Add `#agents-status-panel`, `#agents-meta-panel`, `#agents-schedule-panel`,
  `#agents-memory-panel` section styling
- Add DataTable styling for schedule and meta tables
- Size the left panel (control/schedule/memory) at `1fr` and log panel at `2fr`
  with `min-width` constraints

**Files to modify:**
- `lsm/ui/tui/styles.tcss`

**Post-block:** Visual verification, tests, commit.

### 1.3: Split CSS into Per-Screen Files (COMPLETED)

The monolithic 960-line `styles.tcss` makes it hard to find and modify screen-specific styles.
Split into modular files that are loaded by the main app.

**Tasks:**
- Create `lsm/ui/tui/styles/` directory
- Extract global/shared styles to `lsm/ui/tui/styles/base.tcss`
- Extract query styles to `lsm/ui/tui/styles/query.tcss`
- Extract ingest styles to `lsm/ui/tui/styles/ingest.tcss`
- Extract settings styles to `lsm/ui/tui/styles/settings.tcss`
- Extract remote styles to `lsm/ui/tui/styles/remote.tcss`
- Extract agents styles to `lsm/ui/tui/styles/agents.tcss`
- Extract widget styles to `lsm/ui/tui/styles/widgets.tcss`
- Update `LSMApp.CSS_PATH` in `app.py` to load all split files (Textual supports
  `CSS_PATH` as a list of paths)
- Delete original `styles.tcss` after migration
- Verify no style regressions

**Files to create:**
- `lsm/ui/tui/styles/base.tcss`
- `lsm/ui/tui/styles/query.tcss`
- `lsm/ui/tui/styles/ingest.tcss`
- `lsm/ui/tui/styles/settings.tcss`
- `lsm/ui/tui/styles/remote.tcss`
- `lsm/ui/tui/styles/agents.tcss`
- `lsm/ui/tui/styles/widgets.tcss`

**Files to modify:**
- `lsm/ui/tui/app.py` — update CSS loading

**Files to delete:**
- `lsm/ui/tui/styles.tcss`

**Post-block:** Visual verification on all screens, tests, commit.

### 1.4: Phase 1 Changelog (COMPLETED)

Summarize Phase 1 changes into `docs/development/CHANGELOG.md`.

---

## Phase 2: Settings Screen MVC Refactoring

The Settings screen is a 1,310-line monolith that directly mutates config objects from UI event handlers.
This phase breaks it into an MVC architecture:
- **Model**: Existing config dataclasses (`GlobalConfig`, `IngestConfig`, etc.) — no changes needed
- **Controller**: `SettingsScreen` (slimmed) — routes events, coordinates tabs, manages save/reset
- **Views**: 8 separate tab widget files — each handles composing and refreshing its own fields

### 2.1: Create Settings Utilities Module (COMPLETED)

Extract shared field-creation helpers and widget-query utilities from `settings.py` into a reusable
module that all tab widgets will import.

**Tasks:**
- Extract `_field()` (creates Input or Switch based on type)
- Extract `_select_field()` (creates Select dropdown)
- Extract `_set_input()`, `_set_switch()`, `_set_select_value()`, `_set_select_options()`
- Extract `_save_reset_row()` (creates Save/Reset button pair)
- Extract `_replace_container_children()` (mounts/remounts dynamic widgets)
- Create a `BaseSettingsTab` class (extends `Widget`) with:
  - Common field helpers inherited from above
  - Abstract `refresh_fields(config)` method
  - Abstract `apply_update(field_id, value, config)` method
  - `_is_refreshing` guard logic
  - Status message posting via parent controller

**Files to create:**
- `lsm/ui/tui/widgets/settings_base.py` — `BaseSettingsTab` class + field helpers

**Post-block:** Tests for `BaseSettingsTab` utilities, run `pytest tests/ -v`, commit.

### 2.2: Create Tab Widget Files (Views) (COMPLETED)

Extract each tab's compose and refresh logic into its own widget file. Each tab widget:
- Extends `BaseSettingsTab`
- Implements `compose()` for its field layout
- Implements `refresh_fields(config)` to populate values from config
- Implements `apply_update(field_id, value, config)` for live config mutation
- Handles its own dynamic list management (roots, providers, services, chains)

**Tasks:**
- Create `GlobalSettingsTab` — 5 text inputs (embed_model, device, batch_size,
  embedding_dimension, global_folder)
- Create `IngestSettingsTab` — 14 inputs + 7 switches + 1 select + dynamic roots list
  (add/remove root, per-root path/tags/content_type)
- Create `QuerySettingsTab` — 12 inputs + 3 switches + 2 selects
- Create `LLMSettingsTab` — dynamic providers list + dynamic services dict
  (add/remove/rename with cascading updates)
- Create `VectorDBSettingsTab` — 10 inputs + 1 select (show/hide provider-specific fields)
- Create `ModesSettingsTab` — 4 read-only displays (mode browser)
- Create `RemoteSettingsTab` — dynamic providers + chains with nested links
- Create `ChatsNotesSettingsTab` — 8 inputs + 4 switches (two sub-configs)

**Files to create:**
- `lsm/ui/tui/widgets/settings_global.py`
- `lsm/ui/tui/widgets/settings_ingest.py`
- `lsm/ui/tui/widgets/settings_query.py`
- `lsm/ui/tui/widgets/settings_llm.py`
- `lsm/ui/tui/widgets/settings_vectordb.py`
- `lsm/ui/tui/widgets/settings_modes.py`
- `lsm/ui/tui/widgets/settings_remote.py`
- `lsm/ui/tui/widgets/settings_chats_notes.py`

**Post-block:** Per-tab tests in `tests/test_ui/tui/test_settings_tabs.py`, run `pytest tests/ -v`,
commit.

### 2.3: Refactor SettingsScreen as Controller (COMPLETED)

Slim `settings.py` down to ~150-200 lines. It becomes the controller that:
- Composes the `TabbedContent` with 8 tab widgets
- Routes `on_input_changed`, `on_switch_changed`, `on_select_changed` events to the active tab's
  `apply_update()`
- Handles cross-tab synchronization (mode change updates Modes tab, provider rename cascades)
- Manages Save/Reset by delegating to config loader
- Posts status messages from tab validation errors

**Tasks:**
- Remove all tab-specific compose logic (delegated to tab widgets)
- Remove `_refresh_settings()` monolith (each tab has `refresh_fields()`)
- Remove `_apply_live_update_inner()` giant if/elif chain (each tab has `apply_update()`)
- Remove dynamic list management methods (moved to respective tabs)
- Keep: tab activation routing, save/reset orchestration, cross-tab sync, status display
- Wire up tab widget events to controller dispatch

**Files to modify:**
- `lsm/ui/tui/screens/settings.py` — major rewrite (1,310 → ~150-200 lines)

**Post-block:** Update `tests/test_ui/tui/test_settings_screen.py` to test controller behavior
with tab widgets, run `pytest tests/ -v`, commit.

### 2.5: AppState/ViewModel for Settings + Global UI State

Introduce a typed UI state layer so screens render from immutable snapshots/state transitions
instead of directly mutating runtime config objects.

**Tasks:**
- Create `SettingsViewModel` with explicit state buckets:
  - `persisted_config` (last loaded/saved state)
  - `draft_config` (editable working copy)
  - `dirty_fields`/`dirty_tabs`
  - `validation_errors`
- Add typed update actions (`update_field`, `add_item`, `remove_item`, `rename_key`, `reset_tab`,
  `reset_all`, `save`) with centralized validation and normalization
- Ensure Settings tabs dispatch actions to ViewModel; tabs should not mutate config dataclasses directly
- Make `SettingsScreen` render from ViewModel state snapshots
- Add global UI state container for cross-screen concerns (active context, density mode, notifications,
  selected agent id) with typed read/write APIs
- Keep save boundary explicit: only ViewModel/controller writes through config loader/serializer

**Files to create:**
- `lsm/ui/tui/state/settings_view_model.py`
- `lsm/ui/tui/state/app_state.py`

**Files to modify:**
- `lsm/ui/tui/screens/settings.py`
- `lsm/ui/tui/widgets/settings_base.py`
- `lsm/ui/tui/app.py`
- `lsm/config/loader.py` (integration points for explicit save/serialize boundaries)

**Post-block:** Add tests for ViewModel action flows, validation, dirty tracking, save/reset,
and cross-tab synchronization. Run `pytest tests/ -v`, commit.

### 2.6: Phase 2 Changelog

Summarize Phase 2 changes into `docs/development/CHANGELOG.md`.

---

## Phase 3: Interactive Agent System

Currently, when an agent needs user permission, `ToolSandbox._enforce_tool_permissions()` raises
`PermissionError` and the agent crashes. The `WAITING_USER` status exists in `AgentStatus` but is
never used. This phase adds bidirectional communication and multi-agent support.

### 3.1: Create Interaction Channel

A thread-safe communication channel between the agent harness (background thread) and the UI
(main thread). The harness posts requests; the UI posts responses.

**Tasks:**
- Create `InteractionRequest` dataclass: `request_id`, `request_type` (`"permission"` | `"clarification"` |
  `"feedback"` | `"confirmation"`), `tool_name`, `risk_level`, `reason`, `args_summary`, `prompt`,
  `timestamp`
- Create `InteractionResponse` dataclass: `request_id`, `decision` (`"approve"` | `"deny"` |
  `"approve_session"` | `"reply"`), `user_message`
- Create `InteractionChannel` class:
  - `post_request(request) -> InteractionResponse` - blocks calling thread until response arrives
    (with configurable timeout)
  - `get_pending_request() -> Optional[InteractionRequest]` - non-blocking, for UI polling
  - `post_response(response)` - UI thread sends decision
  - `has_pending() -> bool` - quick check for UI
  - `cancel_pending(reason)` - unblocks waiters when agent is stopped or app is shutting down
  - `shutdown(reason)` - marks channel closed and safely rejects new waits
  - Thread-safe via `threading.Event` and `threading.Lock`
  - Stores session-level approvals (tool_name set) so repeated approvals are not needed
- Add `timeout_seconds` parameter (default `300`) for request expiry
- Add `timeout_action` parameter (default `"deny"`) configurable via `agents.interaction`.
  When timeout expires: `"deny"` raises `PermissionError` (safe default), `"approve"` auto-approves
- Add `InteractionConfig` dataclass to `lsm/config/models/agents.py`:
  `timeout_seconds: int = 300`, `timeout_action: str = "deny"`
- Add `max_concurrent: int = 5` to `AgentConfig`
- Add config loader/serializer support for `agents.interaction` and `agents.max_concurrent`
  (parse + `config_to_raw(...)` round-trip)
- Update `example_config.json` with interaction and concurrency examples

**Files to create:**
- `lsm/agents/interaction.py` - `InteractionChannel`, `InteractionRequest`, `InteractionResponse`

**Files to modify:**
- `lsm/config/models/agents.py` - add `InteractionConfig` and `max_concurrent` to `AgentConfig`
- `lsm/config/loader.py` - parse/serialize interaction + concurrency fields
- `example_config.json` - document interaction + concurrency defaults

**Post-block:** `tests/test_agents/test_interaction_channel.py` (thread safety, timeout,
session approvals, cancellation/shutdown), plus config round-trip tests in
`tests/test_config/test_agents_config.py`, run `pytest tests/ -v`, commit.

### 3.2: Integrate Channel into Sandbox, Harness, and Clarification Flow

When `requires_confirmation=True`, instead of raising `PermissionError`, the sandbox pauses and
sends a request through the channel.

**Tasks:**
- Add `interaction_channel: Optional[InteractionChannel]` parameter to `ToolSandbox.__init__()`
- Modify `_enforce_tool_permissions()`:
  - When `decision.requires_confirmation` and channel exists:
    1. Check session approvals first (skip prompt if already approved for this tool)
    2. Post `InteractionRequest` to channel
    3. Block until response (or timeout)
    4. If approved, record session approval and continue
    5. If denied, raise `PermissionError` with user's reason
  - When `decision.requires_confirmation` and no channel: raise `PermissionError` (existing behavior)
- Add `interaction_channel` parameter to `AgentHarness.__init__()`
- Pass channel through to sandbox
- Set `AgentStatus.WAITING_USER` while waiting for response, restore `RUNNING` after
- Ensure channel is propagated to sub-agent sandboxes (for meta-agent)
- Add stop-safe behavior: stopping an agent while waiting must cancel pending requests and unblock
  waiting calls deterministically
- Add app-shutdown-safe behavior: runtime manager shutdown cancels all pending requests before joining threads
- Implement clarification flow for all agents:
  - Create a built-in `ask_user` tool that raises a `"clarification"` interaction request and returns user reply text
  - Register `ask_user` in the default tool registry for all agents
  - Ensure harness/system allowlist handling never strips `ask_user` (always available)

**Files to create:**
- `lsm/agents/tools/ask_user.py` - clarification interaction tool implementation

**Files to modify:**
- `lsm/agents/tools/sandbox.py` - add channel integration to `_enforce_tool_permissions()`
- `lsm/agents/harness.py` - accept/pass channel and manage `WAITING_USER` transitions
- `lsm/agents/tools/__init__.py` - register `ask_user` by default
- `lsm/agents/base.py` and/or `lsm/agents/tools/base.py` - ensure `ask_user` is always available to agents

**Post-block:** `tests/test_agents/test_sandbox_interaction.py` (approval flow, denial flow,
timeout, session approval caching, stop/shutdown cancellation), plus
`tests/test_agents/test_ask_user_tool.py` for clarification flow, security tests still pass,
run `pytest tests/ -v`, commit.

### 3.3: Multi-Agent Runtime Manager

Replace the single-agent tracking in `AgentRuntimeManager` with a dictionary-based registry
that can track multiple concurrent agents.

**Tasks:**
- Replace `_active_agent`, `_active_thread`, `_active_name` with
  `_agents: Dict[str, AgentRunEntry]` where `AgentRunEntry` is a dataclass holding:
  `agent_id` (uuid), `agent_name`, `agent`, `thread`, `harness`, `channel`, `started_at`, `topic`
- Enforce configurable max concurrent agents (default 5) via `agents.max_concurrent` config field.
  Reject start if limit reached with clear message.
- `start()` generates unique agent_id, creates `InteractionChannel`, passes to harness,
  stores entry, returns agent_id in output message
- `status(agent_id=None)` — if no id, show all agents; if id, show specific
- `pause(agent_id)`, `resume(agent_id)`, `stop(agent_id)` — target specific agent
- `log(agent_id)` — get log for specific agent
- `list_running()` returns list of `AgentRunEntry` summaries
- `get_pending_interactions()` returns all agents with pending interaction requests
- `respond_to_interaction(agent_id, response)` forwards response to channel
- Keep backward compatibility: if only one agent running, commands without agent_id target it
- Clean up completed agent entries after configurable retention (default: keep last 10)
- Add deterministic lifecycle handling:
  - stop path cancels pending interaction and joins agent thread with timeout
  - shutdown path cancels all pending interactions before joining all active threads
  - completed-run pruning keeps bounded history and removes stale queues/channels
  - race-safe transitions for start/stop/cleanup under lock

**Files to modify:**
- `lsm/ui/shell/commands/agents.py` — refactor `AgentRuntimeManager`
- `lsm/config/models/agents.py` - add `max_concurrent: int = 5` to `AgentConfig` (if not done in 3.1)

**Post-block:** `tests/test_ui/shell/test_multi_agent_manager.py` including stop/shutdown/join race cases, run `pytest tests/ -v`, commit.

### 3.4: Agent Interaction UI in TUI

Add UI elements to the Agents screen for handling interaction requests and showing running agents.

**Tasks:**
- Add "Running Agents" DataTable at top of agents-left panel:
  - Columns: ID (short), Agent, Topic, Status, Duration
  - Row selection changes which agent's log is displayed
  - Auto-refresh via timer (every 2 seconds)
- Add "Interaction Request" panel (initially hidden, shown when pending):
  - Displays: request type, tool name (if applicable), risk level, reason/description, args summary
  - Permission requests: "Approve", "Approve for Session", "Deny" buttons + optional deny reason
  - Clarification/feedback requests: text reply input + "Send Reply" action
  - Panel highlights with warning color when active
  - Auto-polls for pending requests via timer
- Add notification badge/indicator when interaction is pending (visual urgency)
- Wire agent selection to log panel (clicking agent row shows its log)
- Update Start button to not block if another agent is running
- Update Status to show all agents summary
- Add keyboard shortcuts for interaction actions (approve/deny/reply) and running-agent row navigation

**Files to modify:**
- `lsm/ui/tui/screens/agents.py` - add running agents table, interaction panel, timers, and keyboard actions
- `lsm/ui/tui/styles/agents.tcss` - styling for new panels and urgency states

**Post-block:** `tests/test_ui/tui/test_agent_interaction.py`, run `pytest tests/ -v`, commit.

### 3.5: Agent Interaction CLI Commands

Add CLI commands for responding to agent interaction requests.

**Tasks:**
- Add `/agent list` - show all running agents with status
- Add `/agent interact` - show pending interaction requests
- Add `/agent approve <agent_id>` - approve pending request for agent
- Add `/agent deny <agent_id> [reason]` - deny pending request
- Add `/agent approve-session <agent_id>` - approve and remember for session
- Add `/agent reply <agent_id> <message>` - respond to clarification/feedback request
- Add `/agent select <agent_id>` - set active agent for status/log/stop commands
- Update existing commands to work with multi-agent (use selected or only agent)

**Files to modify:**
- `lsm/ui/shell/commands/agents.py` - add interaction commands
- `lsm/ui/tui/completions.py` - add completion for new commands

**Post-block:** `tests/test_ui/shell/test_agent_interaction_commands.py`, run `pytest tests/ -v`, commit.

### 3.6: Real-Time Agent Log Streaming

Currently the agent log is only visible when explicitly refreshed. Add a streaming mechanism
so the UI shows log entries as they happen.

**Tasks:**
- Add `log_callback: Optional[Callable[[AgentLogEntry], None]]` to `AgentHarness.__init__()`
- Call callback in `_append_log()` after appending entry
- In `AgentRuntimeManager.start()`, wire callback to push entries to a thread-safe queue per agent
- In `AgentsScreen`, add a Textual timer (every 500ms) that drains the log queue and appends to `RichLog`
- Add bounded queue/backpressure policy for each agent log stream (default max entries), with
  dropped-message counter and truncation notice in UI
- Format log entries with actor-colored prefixes: `[LLM]`, `[TOOL]`, `[AGENT]`, `[USER]`
- Show tool execution in real-time: tool name + args summary -> result summary

**Files to modify:**
- `lsm/agents/harness.py` - add log callback
- `lsm/ui/shell/commands/agents.py` - wire callback in manager
- `lsm/ui/tui/screens/agents.py` - add streaming timer and queue drain
- `lsm/config/models/agents.py` and `lsm/config/loader.py` - optional log queue limit config
- `example_config.json` - optional log queue limit example

**Post-block:** Tests for log streaming and queue pressure handling, run pytest tests/ -v, commit.

### 3.7: Phase 3 Changelog

Summarize Phase 3 changes into `docs/development/CHANGELOG.md`.

---

## Phase 4: TUI UX and Interaction Polish

This phase focuses on user-facing usability and interaction improvements that directly affect
day-to-day TUI operation.

### 4.1: Toast Notification System

Add a non-blocking notification system for background events (agent started/completed,
ingest finished, errors).

**Tasks:**
- Leverage Textual's built-in `self.notify()` method (available since Textual 0.40+)
- Add notifications for:
  - Agent started / completed / failed
  - Agent waiting for user interaction (high priority)
  - Ingest build completed
  - Config saved successfully
  - Schedule triggered
- Add notification severity levels: info, warning, error
- Configure notification timeout (default 5s, errors 10s)

**Files to modify:**
- `lsm/ui/tui/app.py` - add `notify_event()` helper method
- `lsm/ui/tui/screens/agents.py` - emit notifications on agent events
- `lsm/ui/tui/screens/ingest.py` - emit notification on build complete
- `lsm/ui/tui/screens/settings.py` - emit notification on save

**Post-block:** Tests, run `pytest tests/ -v`, commit.

### 4.2: Context-Sensitive Help

Replace the static help modal with context-sensitive help that shows commands relevant to the
active tab.

**Tasks:**
- Modify `HelpScreen` to accept a `context` parameter ("query", "ingest", "remote", "agents",
  "settings")
- Show only commands relevant to current context, with "All Commands" expandable section
- Add keyboard shortcut hints inline (show keybinding next to action)
- Add a "What's New in v0.6.0" section highlighting new agent interaction features

**Files to modify:**
- `lsm/ui/tui/screens/help.py` — context-aware rendering
- `lsm/ui/tui/app.py` — pass current context to help modal

**Post-block:** Tests, run `pytest tests/ -v`, commit.

### 4.3: UI Command Helpers Extraction

The query screen is 1,515 lines with a massive command dispatch handler. Extract command
parsing/dispatch helpers into a shared UI helpers package so command behavior is reusable and
consistent across TUI and shell surfaces.

**Tasks:**
- Create `lsm/ui/helpers/commands/` as the standard command-helper package in `lsm.ui`
- Add `lsm/ui/helpers/commands/query.py` with handler functions for each query command group:
  - `handle_mode_commands()` - mode get/set/list
  - `handle_model_commands()` - model, models, providers, provider-status
  - `handle_results_commands()` - show, expand, open, export-citations
  - `handle_agent_commands()` - agent, memory (delegates to shell commands)
  - `handle_filter_commands()` - set, clear, load, context
  - `handle_cost_commands()` - costs, budget, cost-estimate
  - `handle_remote_commands()` - remote-search, remote-providers
  - `handle_note_commands()` - note, notes
- Add shared parsing/validation helpers in `lsm/ui/helpers/commands/common.py` for tokenization,
  argument normalization, and error formatting
- Make query screen `_execute_query_command()` a thin dispatcher that delegates to helper handlers
- Reuse command helpers from shell paths where appropriate to reduce parser drift

**Files to create:**
- `lsm/ui/helpers/__init__.py`
- `lsm/ui/helpers/commands/__init__.py`
- `lsm/ui/helpers/commands/common.py`
- `lsm/ui/helpers/commands/query.py`

**Files to modify:**
- `lsm/ui/tui/screens/query.py` - slim down from ~1,515 to ~600-700 lines
- `lsm/ui/shell/commands/agents.py` - reuse shared parser/error helpers where applicable
- `lsm/ui/tui/completions.py` - align completion behavior with helper-level parser contracts

**Post-block:** All existing query tests pass, command-contract tests stay green, run `pytest tests/ -v`, commit.

### 4.4: Agent Panel Refresh Controls and Log Following

Improve usability and performance by giving users explicit control over refresh behavior.

**Tasks:**
- Add refresh toggle and interval controls for running-agents and interaction polling timers
- Add "follow selected agent log" toggle in Agents screen
- Add unread log counters per running agent when not selected
- Reset unread counters when an agent is selected or manually cleared

**Files to modify:**
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/styles/agents.tcss`

**Post-block:** `tests/test_ui/tui/test_agent_interaction.py` updates for refresh/follow/unread behavior, run `pytest tests/ -v`, commit.

### 4.5: Responsive Layout Fallbacks for Narrow Terminals

Strengthen terminal compatibility with explicit breakpoint behavior.

**Tasks:**
- Add breakpoint-based layout fallbacks for narrow terminals (for example 80x24 and smaller)
- Collapse split panes to single-column flow when width is constrained
- Ensure focused input and primary action controls remain reachable without deep scrolling

**Files to modify:**
- `lsm/ui/tui/styles/base.tcss` and per-screen style files (or `styles.tcss` before split)
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/query.py`
- `lsm/ui/tui/screens/ingest.py`

**Post-block:** Manual verification in Windows command prompt and tests for narrow-layout behavior.

### 4.6: Phase 4 Changelog

Summarize Phase 4 changes into `docs/development/CHANGELOG.md`.

---

## Phase 5: TUI Reliability and Architecture

This phase hardens runtime safety and evolves the UI architecture for maintainability.

### 5.1: TUI Structural Regression Tests

Add lightweight structural assertions to catch UI regressions early.

**Tasks:**
- Add tests asserting key widgets/IDs exist for each major screen
- Add lightweight snapshot/golden tests for key screen structures and command output formatting
- Add tests for timer lifecycle safety (start/stop without leaks)
- Add smoke tests for interaction panel mode switching (permission vs clarification)

**Files to create:**
- `tests/test_ui/tui/test_layout_structure.py`

**Files to modify:**
- `tests/test_ui/tui/test_agent_interaction.py`
- `tests/test_ui/tui/test_schedule_screen.py`
- `tests/test_ui/tui/test_meta_screen.py`
- `tests/test_ui/tui/test_completions.py`

**Post-block:** Run `pytest tests/ -v`, commit.

### 5.2: Thread-Safe UI Event Model

Ensure all background work communicates with the UI through a single event/message pattern.

**Tasks:**
- Define typed UI event/message models for background -> UI communication
- Ensure background workers and manager threads never mutate widgets directly
- Route all UI changes through main-thread handlers (`post_message`/queued events)
- Add debug assertions/helpers to detect unsafe off-thread widget mutation

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/shell/commands/agents.py`

**Post-block:** Add thread-safety tests for event-driven updates, run `pytest tests/ -v`, commit.

### 5.3: Worker Lifecycle Standards

Standardize long-running operation handling across TUI surfaces.

**Tasks:**
- Create a consistent worker lifecycle wrapper (start, cancel, timeout, join)
- Require cancellation support for all long-running TUI-triggered operations
- Ensure app shutdown and screen unmount cancel/join outstanding workers deterministically
- Add timeout defaults and explicit timeout error handling paths

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/ingest.py`
- `lsm/ui/tui/screens/query.py`

**Post-block:** Add worker timeout/cancel tests, run `pytest tests/ -v`, commit.

### 5.4: Timer Lifecycle Safety

Prevent leaked timers and duplicate polling loops.

**Tasks:**
- Add timer registration helpers with idempotent start/stop semantics
- Ensure each screen starts timers in mount and stops them in unmount
- Add duplicate-timer guards for repeated context switches
- Add tests for timer lifecycle and teardown under rapid screen changes

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/query.py`
- `lsm/ui/tui/screens/ingest.py`

**Post-block:** Add timer lifecycle tests, run `pytest tests/ -v`, commit.

### 5.5: UI Error Boundary and Recovery

Keep the TUI resilient to screen-level exceptions.

**Tasks:**
- Add a global UI error boundary to catch non-fatal screen exceptions
- Render a recoverable error state/panel instead of terminating the app
- Add "Return to safe screen" action (for example Query/Main)
- Ensure full traceback is logged while user-facing message stays concise

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/screens/help.py`
- `lsm/logging.py`

**Post-block:** Add tests for recoverable screen errors, run `pytest tests/ -v`, commit.

### 5.6: Dirty-State and Unsaved-Change Guards

Prevent accidental configuration/data loss during navigation.

**Tasks:**
- Track dirty state per Settings tab and screen-level aggregate dirty status
- Add dirty indicators in tab labels/status area
- Prompt for confirmation on tab/context/app exit when unsaved changes exist
- Add explicit save/discard flow for bulk reset actions

**Files to modify:**
- `lsm/ui/tui/screens/settings.py`
- `lsm/ui/tui/widgets/settings_base.py`
- `lsm/ui/tui/styles/settings.tcss`

**Post-block:** Add tests for dirty-state transitions and confirmation prompts, run `pytest tests/ -v`, commit.

### 5.7: Keyboard-First Interaction Parity

Ensure all major TUI workflows are fully keyboard operable.

**Tasks:**
- Audit all primary actions and add keybinding parity for non-mouse operation
- Add bindings for agent interaction actions (approve/deny/reply/focus panels)
- Add bindings for schedule/memory panel actions and row navigation
- Update help screen to display action-level keyboard shortcuts

**Files to modify:**
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/query.py`
- `lsm/ui/tui/screens/settings.py`
- `lsm/ui/tui/screens/help.py`

**Post-block:** Add keybinding coverage tests, run `pytest tests/ -v`, commit.

### 5.8: TUI Performance Budgets and Polling Guardrails

Keep refresh loops performant and bounded under load.

**Tasks:**
- Define polling/refresh budget defaults (max frequency, max rows per tick, queue caps)
- Add adaptive refresh backoff when no updates are pending
- Cap per-tick render work to prevent UI stalls under high event volume
- Add debug counters for drops/backpressure/render lag

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/query.py`
- `lsm/config/models/agents.py` and `lsm/config/loader.py` (if exposed as config)

**Post-block:** Add performance guardrail tests, run `pytest tests/ -v`, commit.

### 5.9: Command Parsing Contract Tests

Stabilize command behavior while refactoring handlers.

**Tasks:**
- Add command grammar/contract tests for `/agent`, `/memory`, `/ui`, and query command groups
- Assert consistent parse/validation/error-message behavior across CLI and TUI command paths
- Add completion-contract tests to keep autocomplete aligned with parser behavior

**Files to create:**
- `tests/test_ui/shell/test_command_contracts.py`

**Files to modify:**
- `tests/test_ui/tui/test_completions.py`
- `lsm/ui/helpers/commands/query.py`
- `lsm/ui/shell/commands/agents.py`

**Post-block:** Run command-contract suite + full tests, `pytest tests/ -v`, commit.

### 5.10: TUI Architecture Guideline Documentation

Document architectural conventions so future TUI work remains consistent.

**Tasks:**
- Create a TUI architecture guide covering state, events, workers, timers, errors, and testing
- Define explicit "do/don't" rules for thread access and widget mutation
- Document lifecycle expectations (mount/unmount, worker cancellation, timer teardown)
- Link the guide from contributor/dev documentation

**Files to create:**
- `docs/development/TUI_ARCHITECTURE.md`

**Files to modify:**
- `CONTRIBUTING.md`
- `CLAUDE.md`
- `docs/development/TESTING.md`

**Post-block:** Documentation review and consistency pass, commit.

### 5.11: Screen Presenter/Controller Decomposition

Reduce large screen complexity by splitting rendering, action handling, and async coordination into
focused collaborators.

**Tasks:**
- For large screens (`query`, `agents`, `settings`), split logic into:
  - screen container (composition + routing)
  - panel presenters/controllers (per-panel rendering + interaction logic)
  - service adapters (calls into shell/runtime/config operations)
- Move per-panel refresh and state-derivation logic out of screen classes
- Keep screen classes focused on navigation, message routing, and high-level orchestration
- Add clear ownership boundaries so each panel can be tested independently

**Files to create:**
- `lsm/ui/tui/presenters/query/` (presenter modules for query panels)
- `lsm/ui/tui/presenters/agents/` (presenter modules for agents panels)
- `lsm/ui/tui/presenters/settings/` (presenter modules for settings panels)

**Files to modify:**
- `lsm/ui/tui/screens/query.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/settings.py`

**Post-block:** Add focused presenter/controller tests and regression checks, run `pytest tests/ -v`, commit.

### 5.12: Shared Base Screen and UI Helpers Layer

Create shared screen scaffolding so lifecycle, status, notifications, worker wiring, and timer wiring
are implemented once and reused consistently.

**Tasks:**
- Add a base TUI screen abstraction with standardized hooks for:
  - timer registration/teardown
  - worker registration/cancel/join
  - status + notification publishing
  - common error-handling wrapper
- Move repeated screen helper logic into `lsm/ui/helpers/` modules
- Ensure concrete screens override minimal extension points instead of duplicating plumbing
- Add guidance for when logic belongs in screen vs presenter vs helper

**Files to create:**
- `lsm/ui/tui/screens/base.py`
- `lsm/ui/helpers/lifecycle.py`
- `lsm/ui/helpers/notifications.py`

**Files to modify:**
- `lsm/ui/tui/screens/query.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/tui/screens/ingest.py`
- `lsm/ui/tui/screens/settings.py`
- `lsm/ui/tui/app.py`

**Post-block:** Add tests for shared lifecycle helpers and base-screen behavior, run `pytest tests/ -v`, commit.

### 5.13: UI Test Architecture and Reusable Fixtures

Align UI tests with TUI module boundaries and reduce duplicate setup logic.

**Tasks:**
- Mirror UI module structure in tests (screen/presenter/helper level separation)
- Add shared fixture/harness utilities for mounting app/screens and driving timer/worker events
- Separate contract tests (command parsing), structure tests (widget layout), and behavior tests
  (interaction and workflows) into clear test modules
- Add explicit naming conventions for UI tests to make coverage gaps obvious

**Files to create:**
- `tests/test_ui/tui/fixtures/` (shared app/screen harness fixtures)
- `tests/test_ui/tui/presenters/` (presenter-focused tests)
- `tests/test_ui/helpers/` (helper-level tests)

**Files to modify:**
- `tests/test_ui/tui/test_layout_structure.py`
- `tests/test_ui/shell/test_command_contracts.py`
- `docs/development/TESTING.md`

**Post-block:** Run full UI test suite + full project tests, `pytest tests/ -v`, commit.

### 5.14: Phase 5 Changelog

Summarize Phase 5 changes into `docs/development/CHANGELOG.md`.

---

## Phase 6: TUI Startup and Performance

This phase enforces startup stability and a measurable startup performance budget.

### 6.1: TUI Startup Smoke Test (Home Screen)

Add a startup smoke test that launches the TUI and verifies it reaches the default home screen
(Query) without crashing.

**Tasks:**
- Add an app-launch smoke test that mounts `LSMApp`, runs initial startup lifecycle, and verifies:
  - no unhandled exception during startup
  - active/default screen is Query/Home
  - command input and core Query widgets are present
- Add a regression test for known crash paths during early startup initialization
- Ensure smoke test runs in the default test suite (not live-only) so crashes are caught quickly

**Files to create:**
- `tests/test_ui/tui/test_startup_smoke.py`

**Files to modify:**
- `tests/test_ui/tui/fixtures/` (startup/mount helper reuse)
- `tests/test_ui/tui/test_layout_structure.py`

**Post-block:** Run `pytest tests/test_ui/tui/test_startup_smoke.py -v` and full suite `pytest tests/ -v`, commit.

### 6.2: TUI Startup Performance Budget and Lazy Background Loading

Enforce fast perceived startup by loading into Query/Home in under one second and deferring
non-critical initialization to background workers.

**Tasks:**
- Define startup performance SLA:
  - TUI reaches Query/Home interactive state in `< 1.0s` (measured from app start to Query screen ready)
- Instrument startup timing with explicit milestones:
  - app process start
  - first screen mounted
  - Query/Home interactive ready
  - background initialization completed
- Move non-critical startup work to background loading after first render:
  - provider status refresh
  - schedule status/persistence sync
  - memory/agent auxiliary panel preloads
  - other non-blocking metadata loads
- Ensure background loading reports progress/non-fatal errors via notifications or status area without blocking UI
- Add startup performance tests/benchmarks:
  - smoke perf test for Query/Home readiness budget
  - regression benchmark for startup milestones
  - configurable threshold override for constrained environments while default remains `< 1.0s`

**Files to modify:**
- `lsm/ui/tui/app.py`
- `lsm/ui/tui/screens/query.py`
- `lsm/ui/tui/screens/agents.py`
- `lsm/ui/helpers/lifecycle.py`
- `tests/test_ui/tui/test_startup_smoke.py`
- `tests/test_ui/tui/test_performance.py` (create if absent)
- `docs/development/TESTING.md`

**Post-block:** Run startup smoke + performance test group and full tests, `pytest tests/ -v`, commit.

### 6.3: Phase 6 Changelog

Summarize Phase 6 changes into `docs/development/CHANGELOG.md`.

---

## Phase 7: Documentation and Version Updates

### 7.1: Final Documentation and Version Updates

**Version bump (0.5.0 -> 0.6.0):**
- `pyproject.toml` — `version = "0.6.0"`
- `lsm/__init__.py` — `__version__ = "0.6.0"`

**Files to modify:**
- `README.md` — Update root project README with v0.6.0 highlights
- `docs/README.md` — Update with v0.6.0 TUI improvements
- `docs/AGENTS.md` — Document interactive agent sessions, multi-agent, interaction channel
- `docs/development/CHANGELOG.md` — Complete v0.6.0 changelog entry
- `example_config.json` — Update any new config options
- Update CLAUDE.md with new file locations and architecture notes

**Post-block:** Run `pytest tests/ -v` final validation, commit.

---

## Verification Plan

1. **After Phase 1**: Visual verification on 80x24 terminal. All screens usable without scrolling
   to reach input fields. Agent screen has proper CSS styling. CSS split loads correctly with no
   style regressions. `auto` density selects compact on small terminals and comfortable on larger
   terminals, and responds correctly to terminal resize.

2. **After Phase 2**: Settings screen MVC working end-to-end. Each tab renders independently.
   Live updates propagate to config objects. Cross-tab synchronization works (provider rename,
   mode change). Save/Reset per section works. ViewModel tracks dirty state and validation errors
   correctly, and save boundaries go only through loader/serializer paths. All existing settings
   tests pass or are updated.

3. **After Phase 3**:
   - Interactive permission flow: Start agent with permission-requiring tool -> agent pauses ->
     interaction panel appears -> approve/deny -> agent resumes/stops
   - Clarification flow: Agent issues `ask_user` request -> user replies in TUI or CLI ->
     agent receives reply and continues
   - Multi-agent: Start 2+ agents simultaneously -> running agents table shows both ->
     select agent to view its log -> stop individual agent
   - Real-time logs: Start agent -> log entries appear in real-time without manual refresh
   - Backpressure: sustained log volume does not grow memory unbounded; drop counters are visible
   - Shutdown safety: stopping agent/app while waiting for input does not deadlock; waits are canceled
   - Security: All existing security tests pass. Permission gate still blocks unauthorized
     tools. Interaction channel does not weaken sandbox (timeout = deny).
   - Run `pytest tests/test_agents/test_security_*.py -v` explicitly

4. **After Phase 4**: Toast notifications appear for agent events. Help modal shows
   context-relevant commands. Query screen is slimmer with command helpers extracted under
   `lsm/ui/helpers/commands/`. Agents panel supports refresh controls, follow-log mode, unread
   counters, and narrow-terminal fallbacks.

5. **After Phase 5**: Structural regression tests are green, thread-safe event messaging is
   enforced, worker/timer lifecycle protections prevent leaks, recoverable UI error handling works,
   dirty-state prompts prevent data loss, keyboard parity is complete, performance guardrails are
   active, command contract tests are green, presenter/controller decomposition is in place for
   large screens, base-screen/shared-helper patterns are adopted, and TUI architecture/testing
   documentation is published.

6. **After Phase 6**: Startup smoke tests verify crash-free launch into Query/Home. Startup-to-Query
   load time meets `< 1.0s`, and non-critical initialization is deferred to background loading with
   observable progress/error reporting.

7. **Final**: Full test suite green, documentation/version updates complete, and manual walkthrough
   of all TUI screens on Windows command prompt at 80x24 passes.
