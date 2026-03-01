# Creating a Task Plan

This document defines how to create task plans for Local Second Mind feature development.

## Task Plan Structure

Every task plan should be broken down into **numbered task phases**. These phases are major feature implementations which are numbered:

```
## Phase N: Major Feature Name
```

### Task Blocks

Each phase is broken into task blocks with a descriptive title:

```
### N.N: Feature Description
```

### Task Block Contents

Each task block must contain:
1. A list of tasks to complete
2. A short description of the feature
3. The files to modify or create

### Post-Task Completion

After completing a task block, the following must be done:

1. Create/update tests for new features
2. API keys must be configurable via `.env` file
3. Run tests: `pytest tests/ -v`
4. Update `docs/` to reflect the changes to the code base.
5. Update `example_config.json` and `.env.example` with new config options
6. Update Architecture and Key Files sections in .agents/docs/ as needed
7. Run `git add` on all modified files and `git commit` with a message following the format in `COMMIT_MESSAGE.md` (see [COMMIT_MESSAGE.md](./COMMIT_MESSAGE.md))

### Success Criteria

Each task block must have a `**success criteria:**` which clearly describes what a successful implementation results in.

### Code Review Phase
Every Major Feature phase should have a final code review phase with tasks:
- Review the changes and ensure the phase is entirely implemented
- Review code for backwards compatibility, deprecated code, or dead code
- Review tests to ensure they are well-structured with no mocks or stubs

### Changelog

Every Major Feature phase should end with a task summarizing the changes and writing them into `docs/CHANGELOG.md`.

## Final Review Phase

Every task plan ends with a final phase for code review and final documentation review.

### Code Review Phase
The purpose of this sub phase is ensure code quality after all previous stages were completed.

**Tasks**:
- Review all phases in this plan and ensure there are no gaps or bugs remaining in the implementation
- Review all changes for backwards compatibility issues — ensure no unintended regressions
- Review for deprecated code, dead code, or legacy compatibility shims — remove them
- Review all new modules for proper error handling and logging
- Review test suite:
  - No mocks or stubs on database operations
  - No auto-pass tests
  - Test structure matches project module structure
  - All new features have unit + integration tests
- Review security:
  - FTS5 queries use parameterized queries
  - Cross-encoder inputs are sanitized
  - Agent tools validate permissions correctly
- Commit and push changes for this sub-phase.

### Integration Testing
The purpose of this sub phase is to ensure all testing (integration, unit, and SLO) are good.

**Tasks**:
- Run end-to-end integration tests:
- Run performance tests and assert against SLO targets:
  - TUI startup SLA
  - Retrieval stage p95 ≤ 500ms (at 100k chunks, 384 dims)
  - End-to-end query p95 ≤ 1.5s for local interactive use
  - Query latency at 10k, 50k, 100k chunks
  - Ingest throughput
- Run all existing tests: `pytest tests/ -v --cov=lsm --cov-report=html`
- Commit and push changes for this sub-phase.

### Architecture Documentation Update

**Tasks**:
- Update all `.agents/docs/architecture/` files to reflect changes made in this plan
- Update `ARCHITECTURE.md` top-level overview
- Commit and push changes for this sub-phase.
**Files**: All architecture docs listed above

**Success criteria**: Architecture docs accurately reflect codebase.

### Release Preparation

**Tasks**:
- Finalize `docs/CHANGELOG.md` with complete version entry
- Update version in `pyproject.toml` to new version
- Update `README.md`
- Verify `example_config.json` is complete and correct
- Verify `.env.example` is complete
- Update TUI release notes in `lsm/ui/tui/screens/help.py`:
  - Refresh `WHAT'S NEW` entries for new version feature highlights
  - Keep entries synchronized with release notes/changelog language
- Update `tests/test_ui/tui/test_screens.py` to validate the new version `WHAT'S NEW` content
- Update `docs/user-guide/AGENTS.md`:
  - Add a `Tools Access` subsection for each agent
  - Enumerate the exact tools each agent can invoke and any scope limits
- Create upgrade guide for previous version → new version users
- Run final full test suite: `pytest tests/ -v`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `pyproject.toml`
- `README.md`
- `example_config.json`
- `.env.example`
- `lsm/ui/tui/screens/help.py`
- `tests/test_ui/tui/test_screens.py`