# Threat Model — LSM Agent Sandbox

This document defines the **threat model** for the layered sandbox architecture used by the Local Second Mind (LSM) agent system. Its purpose is to make explicit:

- What threats the sandbox **is designed to mitigate**
- What threats are **explicitly out of scope**
- The **trust boundaries** involved in agent execution
- The **residual risks** that remain even with Docker-based isolation

This is a *developer-facing* document intended to guide implementation decisions, reviews, and future hardening—not to claim absolute security.

---

## 1. System Overview (Scope of Analysis)

### Assets

The sandbox is designed to protect the following assets:

- User’s filesystem outside allowed paths
- Knowledge base (read-only integrity)
- Agent runtime state and logs
- API keys, credentials, and environment secrets
- Host operating system stability
- User trust and intent (prevent surprise side effects)

### Actors

- **User**: trusted, author of config and initiator of agent runs
- **Agent (LLM-driven)**: *untrusted but intention-aligned*
- **Tools**: trusted code, potentially operating on untrusted inputs
- **External content**: untrusted (URLs, documents, remote responses)

---

## 2. Trust Boundaries

| Boundary | Description |
|--------|-------------|
| Agent ↔ Tool | LLM requests actions; tools enforce policy |
| Tool ↔ Runner | Execution backend (local vs Docker) |
| Runner ↔ Host OS | Process / container isolation boundary |
| Workspace ↔ Host FS | Filesystem containment boundary |
| Network ↔ Local System | External data ingress |

Each boundary assumes **potentially malicious or malformed input** crossing inward.

---

## 3. Threat Categories

Threats are grouped using a simplified STRIDE-inspired model, adapted for local agent systems.

### T1. Arbitrary File Access

**Threat**: Agent attempts to read or write files outside allowed paths.

**Examples**:
- Reading `~/.ssh/id_rsa`
- Writing to system config files
- Overwriting project source code

**Mitigations**:
- Canonical path resolution
- Deny-by-default read/write path lists
- Workspace-only write discipline
- Read-only mounts in Docker

**Residual Risk**:
- Bugs in path canonicalization
- OS-specific path edge cases (Windows junctions, symlinks)

---

### T2. Arbitrary Command Execution

**Threat**: Agent executes unexpected or dangerous commands.

**Examples**:
- `rm -rf /`
- Fork bombs
- Installing persistent malware

**Mitigations**:
- No generic shell tool exposed
- Tool-level schemas (no free-form command strings)
- Local runner: `shell=False`, allowlists
- Docker runner for high-risk tools
- CPU, memory, and time limits

**Residual Risk**:
- Vulnerabilities inside container runtime
- Misclassified tool risk level

---

### T3. Privilege Escalation

**Threat**: Code executed by the agent gains higher privileges than intended.

**Examples**:
- Exploiting container escape vulnerabilities
- Exploiting kernel bugs via syscalls

**Mitigations**:
- Non-root execution in Docker
- Read-only container filesystem
- Dropped capabilities
- Minimal runtime image

**Out of Scope**:
- Zero-day kernel exploits
- Malicious host-level compromise

---

### T4. Network Abuse

**Threat**: Agent exfiltrates data or performs unintended network actions.

**Examples**:
- Sending local notes to third-party servers
- Excessive API usage
- Accessing malicious endpoints

**Mitigations**:
- Network disabled by default
- Explicit `needs_network` tool capability
- Domain allowlists (future)
- Docker `--network=none`

**Residual Risk**:
- Approved network tools leaking data by design

---

### T5. Resource Exhaustion (DoS)

**Threat**: Agent causes system instability through excessive resource use.

**Examples**:
- Infinite loops
- Large file writes
- Excessive token consumption

**Mitigations**:
- Iteration caps
- Token budgets
- CPU/memory/pids limits
- Output truncation
- File write size limits

**Residual Risk**:
- Legitimate workloads near limits
- Host-level contention beyond container caps

---

### T6. Data Integrity Corruption

**Threat**: Agent corrupts or overwrites important data.

**Examples**:
- Modifying notes incorrectly
- Partial writes
- Accidental deletion

**Mitigations**:
- Workspace isolation
- Explicit write paths
- Human-in-the-loop confirmation for risky writes
- Artifact logging

**Residual Risk**:
- User-approved destructive actions

---

### T7. Prompt / Instruction Injection

**Threat**: External content manipulates agent behavior.

**Examples**:
- Documents instructing agent to run tools
- Remote content altering goals

**Mitigations**:
- Strict JSON action parsing
- Tool calls gated by sandbox
- No direct execution of content
- System prompt hierarchy

**Residual Risk**:
- Subtle reasoning manipulation (non-executable)

---

### T8. Secret Leakage

**Threat**: Agent accesses or leaks credentials.

**Examples**:
- Environment variable exposure
- API keys written to logs

**Mitigations**:
- Environment scrubbing for runners
- No secrets mounted into containers by default
- Log redaction hooks (planned)

**Residual Risk**:
- Secrets intentionally passed to tools

---

## 4. Explicit Non-Goals (Out of Scope)

The sandbox **does not attempt** to defend against:

- A malicious user with OS-level access
- Kernel or hypervisor zero-days
- Side-channel attacks
- Hardware-level attacks
- Compromise of Docker itself
- Malicious code intentionally installed by the user

LSM assumes a **non-hostile user** and focuses on **agent misbehavior and accidents**.

---

## 5. Residual Risk Summary

| Risk | Status |
|-----|-------|
| Accidental destructive actions | Mitigated via permissions |
| Unexpected tool behavior | Mitigated via schemas |
| Container escape | Low probability, high impact |
| Misconfiguration | User responsibility |
| Reasoning-level manipulation | Unavoidable |

---

## 6. Security Posture Statement

LSM’s sandbox provides **practical containment**, not formal security guarantees.

It is designed to:
- Prevent accidental damage
- Make agent actions explainable
- Require explicit user intent for hazardous operations
- Provide a clean upgrade path to stronger isolation

It is **not** intended for executing untrusted third-party code or operating in adversarial environments.

---

## 7. Future Hardening (Optional)

Potential future improvements (non-required):

- Seccomp profiles for Linux containers
- Read-only knowledge base snapshots
- Domain-level network allowlists
- WASM-based tool runtimes
- Immutable workspace snapshots with rollback

---

**Bottom line**: This sandbox meaningfully reduces risk from autonomous agents while preserving LSM’s local-first, developer-controlled philosophy.

