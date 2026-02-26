# Phase 1: Interaction Channel — Two-Phase Timeout

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1
**Status**: Complete

Adds an acknowledgment-based two-phase timeout to the agent interaction channel. An interaction
that has been displayed to the user in the TUI is acknowledged automatically; once acknowledged,
the channel waits indefinitely. This is independent of all other phases and can be developed
first.

Reference: [RESEARCH_PLAN.md §5](../docs/RESEARCH_PLAN.md#5-interaction-channel-two-phase-timeout)

---

## 1.1: InteractionConfig — acknowledged_timeout_seconds

**Description**: Add `acknowledged_timeout_seconds` to the `InteractionConfig` dataclass.
A value of `0` means infinite (no timeout once acknowledged). This is the only config change
in Phase 1.

**Tasks**:
- Add `acknowledged_timeout_seconds: int = 0` to `InteractionConfig`
- Ensure the new field is loaded from config via the existing config loading path
- Add `acknowledged_timeout_seconds` to `example_config.json` under `agents.interaction`
- Run relevant tests: `pytest tests/config/ -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/config/models/agents.py` (or wherever `InteractionConfig` is defined) — add field
- `example_config.json` — add `acknowledged_timeout_seconds: 0` to `agents.interaction`
- `tests/config/test_interaction_config.py` — test new field loads and defaults correctly

**Success criteria**: `InteractionConfig` loads `acknowledged_timeout_seconds` from config;
defaults to `0`; existing tests pass unchanged.

---

## 1.2: InteractionChannel — Two-Phase State Machine

**Description**: Modify `InteractionChannel.post_request()` to use a polling loop with two
timeout phases. Add `acknowledge_request()` and thread-safe acknowledged state.

**Tasks**:
- Add `_acknowledged: bool = False` and `_acknowledged_at: Optional[datetime] = None` as
  instance attributes, protected by the existing `_lock`
- Add `acknowledge_request(request_id: str) -> None`:
  - Under `_lock`, verify `request_id` matches the pending request's id
  - If it matches, set `_acknowledged = True`, `_acknowledged_at = datetime.utcnow()`
  - If it does not match, return without changing state
- Rewrite `post_request()` to use a polling loop:
  - Poll in small chunks (e.g., 0.5 s) using `event.wait(timeout=chunk)`
  - If event is set: response received — return it
  - If not yet acknowledged and elapsed > `timeout_seconds`: apply `timeout_action` and return
  - If acknowledged and `acknowledged_timeout_seconds == 0`: keep polling indefinitely
  - If acknowledged and `acknowledged_timeout_seconds > 0`: check elapsed-since-acknowledge
    and apply `timeout_action` if exceeded
- Reset `_acknowledged` and `_acknowledged_at` at the start of each new `post_request()` call
- All reads/writes to `_acknowledged` must be under `_lock`
- Run relevant tests: `pytest tests/agents/test_interaction.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/interaction.py` — `InteractionChannel` class
- `tests/agents/test_interaction.py` — new or extended test module:
  - Test: unacknowledged request times out after `timeout_seconds`
  - Test: acknowledged request does not time out (`acknowledged_timeout_seconds=0`)
  - Test: `acknowledge_request()` with wrong `request_id` does not acknowledge
  - Test: concurrent acknowledge and timeout polling — no race or deadlock
  - Test: `_acknowledged` resets between successive `post_request()` calls

**Success criteria**: An interaction posted and then immediately acknowledged never times out
when `acknowledged_timeout_seconds=0`. An unacknowledged interaction still times out after
`timeout_seconds`. Tests cover both phases and thread-safety.

---

## 1.3: AgentRuntimeManager and TUI Acknowledgment Signal

**Description**: Wire the acknowledgment signal from the TUI into `InteractionChannel`.
`AgentRuntimeManager` gains a forwarding method; the TUI calls it once per unique request.

**Tasks**:
- Add `acknowledge_interaction(agent_id: str, request_id: str) -> None` to `AgentRuntimeManager`:
  - Look up the agent run by `agent_id`
  - Call `run.interaction_channel.acknowledge_request(request_id)`
  - If the agent or channel is not found, return silently (defensive)
- In `_refresh_interaction_panel()` in `agents.py` (TUI screen):
  - Add `_acknowledged_interaction_ids: set[str]` as an instance attribute (init to `set()`)
  - When a new pending interaction is detected and rendered, check if `request_id` is in the set
  - If not: call `manager.acknowledge_interaction(agent_id, request_id)` and add `request_id`
    to the set
  - If yes: skip (do not acknowledge again)
- Add acknowledgment to shell path: when `/agent interact` renders an interaction prompt, call
  `manager.acknowledge_interaction(agent_id, request_id)` before blocking for user input
- Run relevant tests: `pytest tests/ui/ -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/ui/shell/commands/agents.py` — `AgentRuntimeManager.acknowledge_interaction()`
- `lsm/ui/tui/screens/agents.py` — `_refresh_interaction_panel()`, `_acknowledged_interaction_ids`
- `tests/ui/test_agent_runtime_manager.py` — test `acknowledge_interaction()` forwards correctly
- `tests/ui/tui/test_agents_screen.py` — test that acknowledgment is sent once per unique
  `request_id` regardless of how many timer ticks fire

**Success criteria**: The TUI sends exactly one acknowledgment signal per interaction request
regardless of refresh frequency. The shell path sends acknowledgment before blocking on user
input. `AgentRuntimeManager.acknowledge_interaction()` correctly forwards to the channel.

---

## 1.4: Phase 1 Code Review and Changelog

**Tasks**:
- Review `InteractionChannel` changes: verify polling loop handles all timeout branch combinations
  correctly; verify no regression in the `timeout_action` (deny/approve) logic
- Review TUI changes: verify `_acknowledged_interaction_ids` does not grow unboundedly
  (clear stale ids when interactions are resolved)
- Review tests: confirm no mocks or stubs; confirm thread-safety tests are genuine concurrent tests
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 1 changes
- Update `.agents/docs/architecture/development/AGENTS.md`: document two-phase timeout in the
  interaction section
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/AGENTS.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog entry written. AGENTS.md interaction
section updated.
