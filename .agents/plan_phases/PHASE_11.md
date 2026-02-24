# Phase 11: Release & Documentation

**Depends on:** All previous phases

| Task | Description |
|------|-------------|
| 11.1 | Version bump to 0.7.0 |
| 11.2 | Documentation audit |
| 11.3 | Config examples update |
| 11.4 | TUI WHATS_NEW |

## 11.1: Version Bump
- **Description:** Update version metadata to v0.7.0.
- **Tasks:**
  - Update `pyproject.toml`, `lsm/__init__.py`, and any runtime version references to `0.7.0`.
  - Run the full test suite (`pytest tests/`) and verify no regressions from version changes.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `pyproject.toml`
  - `lsm/__init__.py`
- **Success criteria:** All version references are `0.7.0`.

## 11.2: Documentation Audit
- **Description:** Confirm documentation reflects the release scope.
- **Tasks:**
  - Verify all new agent, tool, and provider capabilities are documented.
  - Verify architecture docs reflect restructured agent packages and provider sub-packages.
  - Verify security docs cover new tools (bash, powershell, MCP) and new providers (OAuth-gated).
  - Run the full test suite (`pytest tests/`) and verify no regressions.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `docs/**`
  - `.agents/docs/**`
- **Success criteria:** Documentation matches implemented changes.

## 11.3: Config Examples Update
- **Description:** Ensure examples include new config keys and API variables.
- **Tasks:**
  - Update `example_config.json` with: tier config, MCP server config, new provider entries, OAuth config, command allow/deny lists.
  - Update `.env.example` with new API keys (OpenRouter, news APIs, financial APIs, OAuth client IDs).
  - Write tests to validate that `example_config.json` loads successfully through the config loader without errors.
  - Run the full test suite (`pytest tests/`) and verify no regressions.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `example_config.json`
  - `.env.example`
- **Success criteria:** Config examples are complete and match implemented configuration schema.

## 11.4: TUI WHATS_NEW
- **Description:** Update TUI with v0.7.0 highlights.
- **Tasks:**
  - Update `lsm/ui/tui/screens/help.py` WHATS_NEW section with v0.7.0 highlights.
  - Write/update tests for the WHATS_NEW screen content to verify v0.7.0 features are displayed.
  - Run the full test suite (`pytest tests/`) and verify all tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/ui/tui/screens/help.py`
  - `tests/test_ui/tui/test_screens.py`
- **Success criteria:** WHATS_NEW displays v0.7.0 features.
