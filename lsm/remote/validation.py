"""
Validation helpers for remote provider output contracts.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from lsm.remote.base import RemoteResult


STABLE_ID_KEYS = ("source_id",)


def validate_output(results: Sequence[RemoteResult]) -> List[str]:
    """
    Validate RemoteResult output structure.

    Args:
        results: Sequence of RemoteResult objects.

    Returns:
        List of violation messages (empty when valid).
    """
    violations: List[str] = []
    if results is None:
        violations.append("results is None")
        return violations

    for idx, result in enumerate(results):
        if not isinstance(result, RemoteResult):
            violations.append(f"result[{idx}] is not a RemoteResult")
            continue
        if not str(result.title or "").strip():
            violations.append(f"result[{idx}].title is required")
        if not str(result.url or "").strip():
            violations.append(f"result[{idx}].url is required")
        if result.snippet is None or not str(result.snippet).strip():
            violations.append(f"result[{idx}].snippet is required")
        try:
            score = float(result.score)
        except (TypeError, ValueError):
            violations.append(f"result[{idx}].score must be numeric")
        else:
            if score < 0.0 or score > 1.0:
                violations.append(f"result[{idx}].score must be between 0 and 1")

        metadata = result.metadata
        if metadata is None or not isinstance(metadata, dict):
            violations.append(f"result[{idx}].metadata must be a dict")
        else:
            has_stable_id = any(str(metadata.get(key, "")).strip() for key in STABLE_ID_KEYS)
            if not has_stable_id:
                violations.append(f"result[{idx}].metadata missing stable source_id")

    return violations


def collect_field_names(fields: Iterable[dict]) -> List[str]:
    """
    Normalize output/input field declarations into a list of field names.

    Args:
        fields: Iterable of field definition dicts.

    Returns:
        List of field names.
    """
    names: List[str] = []
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name", "")).strip()
        if name:
            names.append(name)
    return names
