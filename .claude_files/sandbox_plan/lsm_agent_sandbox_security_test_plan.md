# Security Test Plan — LSM Agent Sandbox

This document derives **security-focused tests directly from the LSM Agent Sandbox Threat Model**. It defines concrete unit, integration, and regression tests to ensure the sandbox behaves correctly under adversarial or accidental agent behavior.

The goal is not formal verification, but **practical enforcement of trust boundaries** and early detection of regressions that weaken sandbox guarantees.

---

## Scope

These tests validate protections against:

- Unauthorized filesystem access
- Arbitrary command execution
- Privilege escalation
- Network abuse
- Resource exhaustion (DoS)
- Data integrity corruption
- Prompt / instruction injection
- Secret leakage

Tests are written to align with existing modules:

- `ToolSandbox`
- `SandboxConfig`
- `AgentHarness`
- Tool schemas and registry
- Runner abstraction (`LocalRunner`, `DockerRunner`)

---

## Test Suite Organization

Recommended structure:

```
 tests/
 └─ test_agents/
    ├─ test_security_paths.py
    ├─ test_security_permissions.py
    ├─ test_security_runner_policy.py
    ├─ test_security_network.py
    ├─ test_security_resources.py
    ├─ test_security_secrets.py
    ├─ test_security_injection.py
    └─ test_security_docker_integration.py  (@pytest.mark.docker)
```

---

## T1 — Arbitrary File Access (Read / Write Escapes)

### Unit Tests: Path Normalization & Containment

**Target**: `ToolSandbox`, path canonicalization helpers

**Test Cases**:

1. Block relative path traversal
   - Attempt `read_file("../secrets.txt")`
   - Expect: deny with policy error

2. Block write path traversal
   - Attempt `write_file("../../outside.txt")`
   - Expect: deny

3. Block absolute paths outside allowlist
   - `/etc/passwd` (Linux)
   - `C:\\Windows\\System32` (Windows)

4. Block symlink escape
   - Symlink inside allowed dir pointing outside
   - Attempt read/write via symlink

5. Windows junction / UNC path escape
   - `\\\\server\\share\\file`
   - `C:\\allowed\\..\\Windows`

**Assertions**:
- Canonical resolved path logged
- Policy event recorded ("path rejected")

---

## T2 — Arbitrary Command Execution

### Unit Tests: Tool Schema Enforcement

**Target**: Tool input schemas

**Test Cases**:

1. Reject free-form command fields
   - Pass `{"cmd": "rm -rf /"}` to non-shell tool

2. Reject unknown tool invocation
   - Action: `"shell"`
   - Expect: tool not found

### Integration Tests: Local Runner Safety

**Target**: `LocalProcessRunner`

**Test Cases**:

1. Shell metacharacters treated as literal args
   - `"hi; rm -rf /"`

2. Executable allowlist enforced
   - Allowed: `rg`
   - Attempt: `python`

---

## T3 — Privilege Escalation

### Unit Tests: Docker Runner Configuration

**Target**: Docker command construction

**Test Cases**:

- `--network=none` by default
- `--read-only` root FS
- Non-root user execution
- CPU / memory / PID limits applied
- Minimal mount set (workspace RW, KB RO)

### Integration Tests (Docker)

**Test Cases**:

- Attempt to write to `/etc` inside container
- Expect: failure, no host modification

---

## T4 — Network Abuse

### Unit Tests: Network Capability Gating

**Target**: `ToolSandbox`

**Test Cases**:

1. `allow_url_access=false`
   - `load_url` denied

2. Tool without `needs_network=true`
   - Cannot access network even if schema valid

### Integration Tests (Docker)

**Test Cases**:

1. Network disabled container
   - Attempt outbound HTTP
   - Expect: failure

2. Network-enabled tool
   - Allowed access to test endpoint

---

## T5 — Resource Exhaustion (DoS)

### Unit Tests: Harness Guards

**Target**: `AgentHarness`

**Test Cases**:

1. Iteration cap enforced
   - Infinite tool loop
   - Stop at `max_iterations`

2. Token budget exhaustion
   - Simulated oversized responses

3. Output truncation
   - Tool returns large stdout

4. Tool timeout
   - Tool sleeps past timeout

### Integration Tests (Docker)

- Verify memory and CPU limits are set

---

## T6 — Data Integrity Corruption

### Unit Tests: Workspace Discipline

**Target**: Write tools

**Test Cases**:

1. Writes restricted to workspace
2. Overwrite protection (if implemented)

### Integration Tests: Artifact Tracking

**Test Cases**:

- Created and modified files recorded in `artifacts`
- All artifacts under workspace root

---

## T7 — Prompt / Instruction Injection

### Unit Tests: LLM Response Parsing

**Target**: `AgentHarness`

**Test Cases**:

1. Non-JSON response
   - No tool executed

2. Malformed JSON
   - Reject safely

3. Injection text in arguments
   - Schema validation blocks execution

### Regression Tests

- Natural language requests to run tools without `action` field must not execute

---

## T8 — Secret Leakage

### Unit Tests: Environment Scrubbing

**Target**: LocalRunner, DockerRunner

**Test Cases**:

1. Host env secrets not visible
   - `OPENAI_API_KEY` absent in tool output

2. Log redaction (if implemented)
   - Secrets masked in logs

> Tests may be marked `xfail` until redaction is implemented.

---

## Runner Selection Policy Tests

**Target**: `RunnerSelector`

**Test Cases**:

1. `read_only` → local
2. `writes_workspace` → local
3. `exec | network` → docker
4. `docker_required` without Docker → refuse
5. Per-tool override > per-risk > default

---

## CI Recommendations

- Always run unit security tests
- Run Docker integration tests only when Docker is available
- Mark Docker tests with `@pytest.mark.docker`

---

## Summary

This test plan operationalizes the threat model into **executable security guarantees**. Together with layered sandboxing, these tests ensure:

- Agent autonomy does not imply system authority
- Risky actions require stronger isolation or explicit consent
- Sandbox regressions are caught early and visibly

This plan should evolve alongside new tools and runner capabilities.

