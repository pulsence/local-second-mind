# Agent Sandbox Security

This document describes the security posture for `lsm/agents/` and the adversarial test strategy used to keep it hardened.

## Threat Model

The agent runtime executes LLM-directed tool calls. Primary threat actors:

1. Prompt-injection content in user documents or model responses that attempts to coerce unsafe tool calls.
2. Malicious or hallucinated tool arguments intended to break sandbox boundaries.
3. Secret leakage through environment variables, logs, or persisted run state.

Defense-in-depth is required because the model can produce untrusted structured actions that look syntactically valid.

## Attack Surface Inventory

Untrusted inputs reach the sandbox through:

1. LLM tool-call payloads (`action`, `action_arguments`) parsed in `lsm/agents/harness.py`.
2. Tool input arguments passed to `ToolSandbox.execute()` in `lsm/agents/tools/sandbox.py`.
3. User-provided filesystem paths for read/write tools.
4. URL/network targets for network-risk tools.
5. Tool outputs and model text persisted in run state and logs.
6. Process environment inherited by future tool runners.

## STRIDE Coverage Matrix

| STRIDE | Primary Defenses | Test Files |
|---|---|---|
| Spoofing | Tool registry lookup, unknown-tool rejection, strict action handling | `tests/test_agents/test_security_permissions.py`, `tests/test_agents/test_security_injection.py` |
| Tampering | Path canonicalization, traversal rejection, symlink escape blocking, write-path enforcement | `tests/test_agents/test_security_paths.py`, `tests/test_agents/test_security_integrity.py` |
| Repudiation | Structured logs + state persistence, artifact tracking in `AgentState` | `tests/test_agents/test_security_integrity.py`, `tests/test_agents/test_harness.py` |
| Information Disclosure | Env scrubbing, secret redaction in logs/state/context | `tests/test_agents/test_security_secrets.py` |
| Denial of Service | Iteration caps, token budgets, tool-output truncation | `tests/test_agents/test_security_resources.py` |
| Elevation of Privilege | PermissionGate precedence, risk-based policies, global sandbox subset enforcement | `tests/test_agents/test_security_permissions.py`, `tests/test_agents/test_sandbox.py` |

T1-T8 security suites:

- T1 Arbitrary file access: `tests/test_agents/test_security_paths.py`
- T2+T3 command/privilege escalation: `tests/test_agents/test_security_permissions.py`
- T4 network abuse: `tests/test_agents/test_security_network.py`
- T5 resource exhaustion: `tests/test_agents/test_security_resources.py`
- T6 data integrity: `tests/test_agents/test_security_integrity.py`
- T7 prompt injection: `tests/test_agents/test_security_injection.py`
- T8 secret leakage: `tests/test_agents/test_security_secrets.py`

## Testing Methodology

Security tests are adversarial by design: they attempt sandbox escapes, policy bypasses, and leakage vectors. Tests must verify rejection behavior, not only happy-path success.

Run the security suite:

```bash
pytest tests/test_agents/test_security_*.py -v
```

Run all agent tests:

```bash
pytest tests/test_agents -v
```

## Extending Security Coverage

When adding a tool or agent:

1. Assign tool `risk_level`, `needs_network`, and permission posture.
2. Add sandbox tests for path/network/permission behavior specific to the new tool.
3. Add prompt-injection tests proving malformed or adversarial model output cannot trigger unsafe execution.
4. Add secret-leakage tests if tool output may include credentials/tokens.
5. Add resource tests for large outputs, long loops, or heavy payloads.
6. Update this document’s STRIDE matrix and `docs/development/TESTING.md`.

## Permission Gate Reference

Permission precedence in `PermissionGate.check(...)`:

1. Per-tool explicit override: `sandbox.require_user_permission[tool_name]`
2. Per-risk policy: `sandbox.require_permission_by_risk[risk_level]`
3. Tool default: `tool.requires_permission`
4. Allow

Examples:

- `require_user_permission["query_remote"] = false` allows `query_remote` even if `require_permission_by_risk["network"] = true`.
- `require_permission_by_risk["writes_workspace"] = true` blocks write-risk tools unless explicitly overridden.
- Read-only tools remain allowed by default unless explicitly gated.
