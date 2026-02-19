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

Each task block can be broken into smaller sub-blocks:

```
#### N.N.N: Sub Feature Description
```

## Task Block Contents

Each task block must contain:
1. A list of tasks to complete
2. A short description of the feature
3. The files to modify or create

## Post-Task Completion

After completing a task block, the following must be done:

1. Create/update tests for new features
2. API keys must be configurable via `.env` file
3. Run tests: `pytest tests/ -v`
4. Update `docs/` with new features
5. Update `example_config.json` and `.env.example` with new config options
6. Write commit message (see [COMMIT_MESSAGE.md](./COMMIT_MESSAGE.md))
7. Update Architecture and Key Files sections in CLAUDE.md as needed

## Success Criteria

Each task block must have a `**success criteria:**` which clearly describes what a successful implementation results in.

## Code Review Phase

Every Major Feature phase should have a final code review phase with tasks:
- Review the changes and ensure the phase is entirely implemented
- Review code for backwards compatibility, deprecated code, or dead code
- Review tests to ensure they are well-structured with no mocks or stubs

## Changelog

Every Major Feature phase should end with a task summarizing the changes and writing them into `docs/development/CHANGELOG.md`.
