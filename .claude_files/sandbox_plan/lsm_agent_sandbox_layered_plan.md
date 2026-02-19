# Layered Sandbox Implementation Plan (LSM Agents)

This document packages the agreed-upon **layered sandbox architecture** for the Local Second Mind (LSM) agent system. The goal is to provide **progressive isolation**, starting with low-friction local controls and culminating in **Docker-based hard sandboxing** for hazardous tools, with **explicit user confirmation as the ultimate fallback**.

This plan is designed to integrate cleanly with the existing Agent Framework (Phase 7), including:
- `AgentConfig` / `SandboxConfig`
- `ToolSandbox`
- `AgentHarness`
- Tool registry and runtime logging

---

## Design Principles

1. **Deny-by-default**: Tools operate only within explicitly granted capabilities.
2. **Progressive hardening**: Most tools run locally; only risky tools escalate.
3. **Policy-first**: Execution decisions are explainable and logged.
4. **Cross-platform**: Docker is the canonical hard sandbox backend.
5. **User sovereignty**: Explicit confirmation is always available as a fallback.

---

## Sandbox Layers Overview

| Layer | Purpose | Mechanism |
|------|--------|-----------|
| 0 | Tool capability contract | Tool metadata + schemas |
| 1 | Local execution guardrails | Path + process limits |
| 2 | Workspace containment | Per-run workspace jail |
| 3 | Network gating | Capability-based access |
| 4 | Hard sandbox | Docker runner |
| 5 | Human-in-the-loop | Explicit user confirmation |

---

## Phase A — Sandbox Foundations

### A1. Tool Risk Metadata

Extend `BaseTool` with explicit risk and execution intent:

- `risk_level`: `read_only | writes_workspace | writes_fs | network | exec`
- `preferred_runner`: `local | docker`
- `needs_network: bool`
- `requires_permission: bool`

Expose this metadata via `ToolRegistry` for:
- Runner selection
- Permission decisions
- Logging and plan previews

**Acceptance**: Every tool self-describes its risk profile.

---

### A2. SandboxConfig Extensions

Add runner policy and resource controls:

- `execution_mode`: `local_only | prefer_local | prefer_docker | docker_required`
- `docker`:
  - `enabled`
  - `image`
  - `network_default`
  - `cpu_limit`
  - `mem_limit_mb`
  - `pids_limit`
  - `read_only_root`
- `limits`:
  - `timeout_s_default`
  - `max_stdout_kb`
  - `max_stderr_kb`
  - `max_file_write_mb`
- `permissions`:
  - `require_user_permission_by_risk`
  - (retain per-tool overrides)

**Acceptance**: Config expresses when Docker or confirmation is required.

---

### A3. Runner Abstraction

Introduce a clean execution interface:

- `BaseRunner.run(tool, args, ctx) -> ToolExecutionResult`

`ToolExecutionResult` includes:
- stdout / stderr (truncated)
- artifacts (workspace files)
- runner used
- runtime metrics
- policy events

**Acceptance**: Harness executes tools without knowing how they run.

---

## Phase B — Local Sandbox Hardening

### B1. Path Canonicalization & Escape Prevention

Centralize path enforcement:
- canonicalize paths
- deny `..` traversal
- block symlink escapes

Apply to **all file tools**.

**Acceptance**: No read/write escapes allowed; covered by tests.

---

### B2. Workspace Jail

- Create per-run workspace:
  - `<agents_folder>/<agent>_<timestamp>/workspace/`
- Mount knowledge base as read-only
- Require tools to write only within workspace
- Track written artifacts

**Acceptance**: All writes are attributable and bounded.

---

### B3. Local Process Controls

For any local subprocess execution:
- `shell=False`
- executable allowlist
- timeouts + kill process group
- sanitized environment
- output truncation

**Acceptance**: Local execution is predictable and constrained.

---

## Phase C — Permission Gates

### C1. Permission Decision Pipeline

Permission precedence:
1. Per-tool explicit override
2. Per-risk policy
3. Tool default
4. Safe default

Integrate with:
- TUI prompts
- CLI confirmations (`--yes`)

Log all permission decisions.

**Acceptance**: Permission logic is deterministic and auditable.

---

### C2. Execution Plan Preview

For hazardous tools:
- Display tool + args
- Show runner (local/docker)
- Show filesystem + network intent
- Allow approve-once or approve-for-session

**Acceptance**: User sees consequences before execution.

---

## Phase D — Docker Sandbox (Hard Isolation)

### D1. DockerRunner Implementation

- Execute tools inside ephemeral containers
- Mounts:
  - workspace (RW)
  - allowed read paths (RO)
- Defaults:
  - `--network=none`
  - CPU/memory/pids limits
  - non-root user
  - read-only root FS (where feasible)

**Acceptance**: At least one tool runs end-to-end in Docker.

---

### D2. Runner Selection Policy

Recommended routing:
- `read_only` → local
- `writes_workspace` → local
- `writes_fs | network | exec` → docker

If Docker unavailable:
- require user confirmation **or**
- refuse with actionable error

**Acceptance**: Runner choice is deterministic and logged.

---

### D3. Runtime Image

- Minimal Python-based image
- Contains only tool runtime deps
- Built via `lsm sandbox build-image`

**Acceptance**: Image builds reproducibly and is documented.

---

## Phase E — Harness Integration & Logging

### E1. Harness Integration

- Use runner selector
- Append execution results to context
- Log:
  - requested action
  - runner chosen
  - permission decisions
  - policy events

**Acceptance**: Logs explain *why* execution happened as it did.

---

### E2. Documentation

- Update Agents Guide
- Add sandbox section
- Provide example configs:
  - local-only
  - docker-preferred
  - docker-required

---

## Phase F — Testing

### F1. Unit Tests

- Path escape prevention
- Permission precedence
- Runner selection

### F2. Integration Tests (Docker optional)

- Mark with `@pytest.mark.docker`
- Validate:
  - workspace writes
  - network blocking

---

## Recommended Implementation Order

1. Tool risk metadata + runner abstraction
2. Path + workspace sandbox
3. Permission gates + plan preview
4. Runner selection policy
5. Docker runner + image
6. Harness integration + unit tests
7. Docker integration tests + docs

---

## Definition of Done

- Low-risk tools run locally under strict constraints
- High-risk tools run in Docker by default
- Docker absence triggers confirmation or refusal
- Every execution is explainable via logs

---

This plan establishes a secure, extensible sandbox architecture aligned with LSM’s local-first philosophy while remaining practical for daily development.

