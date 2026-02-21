# Phase 5: Execution Environment

**Why fifth:** Runners and shell tools are independent of agent catalog but must be ready before agents that need code execution (Coder agent in Phase 6).

**Depends on:** Phase 3 (sandbox policy updates)

| Task | Description | Depends On |
|------|-------------|------------|
| 5.1 | Runner policy and config updates | None |
| 5.2 | Docker runner completion | 5.1 |
| 5.3 | WSL2 runner implementation | 5.1 |
| 5.4 | Bash and PowerShell tools | 5.1 |
| 5.5 | Tests and documentation | 5.2–5.4 |

## 5.1: Runner Policy and Config Updates
- **Description:** Align runner selection with sandbox policy and add configuration hooks for new runners.
- **Tasks:**
  - Audit current `ToolSandbox` runner selection rules.
  - Add configuration fields for WSL2 runner and command allow/deny lists.
  - Update runner selection to respect `execution_mode` and `force_docker` policy.
  - Write tests for runner selection logic under various policy configurations, WSL2 config field validation, and `execution_mode`/`force_docker` enforcement (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/sandbox.py`
  - `lsm/config/**`
- **Success criteria:** Runner selection is policy-driven and configurable.

## 5.2: Docker Runner Completion
- **Description:** Finish Docker runner behavior and evaluate hot-load execution paths.
- **Tasks:**
  - Implement missing Docker runner features (volume mapping, environment scrubbing, timeout limits).
  - Evaluate hot-load strategy (persistent container with volume mounts) vs per-run container spawning. Document trade-offs and implement the chosen approach.
  - Volume mapping must respect sandbox `allowed_read_paths` and `allowed_write_paths`.
  - Environment scrubbing via existing `env_scrubber.py`.
  - Document runner constraints and failure modes.
  - Write tests for volume mapping with sandbox path enforcement, environment scrubbing, timeout limits, and lifecycle behavior (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/docker_runner.py`
  - `lsm/agents/tools/env_scrubber.py`
  - `docs/`
- **Success criteria:** Docker runner executes tool commands with enforced limits and predictable lifecycle behavior.

## 5.3: WSL2 Runner Implementation
- **Description:** Add a Windows-hosted WSL2 runner for exec/network tools.
- **Tasks:**
  - Implement `WSL2Runner` with path translation and environment scrubbing.
  - Integrate runner availability checks in `ToolSandbox`.
  - Add logging and error handling for WSL2 invocation failures.
  - Write tests for WSL2 path translation, environment scrubbing, availability checks, and invocation error handling (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/wsl2_runner.py`
  - `lsm/agents/tools/sandbox.py`
- **Success criteria:** WSL2 runner executes commands in sandbox constraints when enabled.

## 5.4: Bash and PowerShell Tools
- **Description:** Provide shell command tools with allow/deny constraints and path validation.
- **Tasks:**
  - Add `bash` tool for command execution with allow/deny config.
  - Add `powershell` tool for command execution with allow/deny config.
  - Implement path validation and sandbox enforcement for command arguments. Use `ToolSandbox.check_read_path()` / `check_write_path()` for any file path arguments detected in the command string.
  - Allow/deny configuration: `agents.sandbox.command_allowlist` and `agents.sandbox.command_denylist` in config.
  - Write tests for allow/deny list enforcement, path validation, and sandbox path checking (TDD: write tests before implementation).
- **Security testing strategy:** Shell tools are a high-risk attack surface. Security tests must cover:
  - Command injection via argument escaping (semicolons, pipes, backticks, `$()`, `&&`, `||`)
  - Path traversal attacks (relative paths, symlinks, `..` sequences escaping sandbox)
  - Environment variable exfiltration attempts
  - Allowlist/denylist bypass attempts (case variations, path aliasing, command aliasing)
  - Resource exhaustion (fork bombs, infinite loops, disk-filling commands)
  - Chained command sequences that individually pass but collectively escape sandbox
  - Tests must be added to the existing STRIDE security test suite (T1–T8 categories), specifically T1 (file access), T2 (command escalation), and T5 (resource exhaustion).
  - Run the relevant test suite (`pytest tests/test_agents/`) including all STRIDE security tests and verify all pass.
- **Files:**
  - `lsm/agents/tools/bash.py`
  - `lsm/agents/tools/powershell.py`
  - `lsm/agents/tools/sandbox.py`
  - `tests/test_agents/test_security_*.py`
- **Success criteria:** Shell tools honor allow/deny lists and validate file paths before execution. Security tests pass across all STRIDE categories.

## 5.5: Tests and Documentation
- **Description:** Validate runner and shell behavior with policy tests.
- **Tasks:**
  - Add unit tests for runner selection and command allow/deny behavior.
  - Add integration tests for Docker and WSL2 runners (when available).
  - Document config options and usage examples.
  - Run the full test suite (`pytest tests/`) and verify all new and existing tests pass, including all tests added in tasks 5.1–5.4.
- **Files:**
  - `tests/test_agents_tools/`
  - `docs/`
- **Success criteria:** Runner and shell tools are tested and documented with clear configuration examples.
