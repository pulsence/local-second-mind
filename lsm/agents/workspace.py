"""
Helpers for per-agent workspace layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lsm.utils.paths import resolve_path


def ensure_agent_workspace(
    agent_name: str,
    agents_folder: Path | str,
    *,
    sandbox: Optional[object] = None,
) -> Path:
    """
    Ensure the per-agent workspace layout exists and optionally bind sandbox defaults.

    Layout:
      <agents_folder>/<agent_name>/
        logs/
        artifacts/
        memory/
    """
    normalized_name = str(agent_name or "").strip() or "agent"
    root = resolve_path(Path(agents_folder) / normalized_name, strict=False)
    for subdir in ("logs", "artifacts", "memory"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    if sandbox is not None:
        setter = getattr(sandbox, "set_workspace_root", None)
        if callable(setter):
            setter(root)
        else:
            try:
                setattr(sandbox, "workspace_root", root)
            except Exception:
                pass

    return root
