"""
Docker sandbox entrypoint for tool execution.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any


def _fail(message: str, *, code: int = 1) -> int:
    sys.stderr.write(message.strip() + "\n")
    return code


def _load_tool(module_name: str, class_name: str) -> Any:
    if not module_name.startswith("lsm.agents.tools."):
        raise ValueError("Tool module must be under lsm.agents.tools")
    module = importlib.import_module(module_name)
    tool_class = getattr(module, class_name)
    try:
        return tool_class()
    except TypeError as exc:
        raise RuntimeError(
            f"Tool class '{class_name}' requires constructor args; "
            "docker entrypoint currently supports no-arg tool classes only"
        ) from exc


def _collect_artifacts(tool: Any, args: dict[str, Any]) -> list[str]:
    if getattr(tool, "risk_level", "") != "writes_workspace":
        return []
    path_value = args.get("path")
    if path_value is None:
        return []
    path_text = str(path_value).strip()
    if not path_text:
        return []
    return [path_text]


def main() -> int:
    raw = sys.stdin.read()
    if not str(raw).strip():
        return _fail("Missing tool payload on stdin")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fail(f"Invalid JSON payload: {exc}")

    if not isinstance(payload, dict):
        return _fail("Tool payload must be a JSON object")

    module_name = str(payload.get("tool_module", "")).strip()
    class_name = str(payload.get("tool_class", "")).strip()
    args = payload.get("args", {})

    if not module_name or not class_name:
        return _fail("Tool payload must include tool_module and tool_class")
    if not isinstance(args, dict):
        return _fail("Tool payload args must be a JSON object")

    try:
        tool = _load_tool(module_name, class_name)
        stdout = str(tool.execute(args))
        response = {
            "stdout": stdout,
            "stderr": "",
            "artifacts": _collect_artifacts(tool, args),
        }
        sys.stdout.write(json.dumps(response))
        return 0
    except Exception as exc:  # pragma: no cover - integration behavior.
        return _fail(f"Docker entrypoint execution failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
