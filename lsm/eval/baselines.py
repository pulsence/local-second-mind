"""
Baseline management for retrieval evaluation.

Baselines are named evaluation snapshots saved to the filesystem
under the global folder, enabling before/after comparisons.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from lsm.logging import get_logger

logger = get_logger(__name__)

_BASELINES_DIR_NAME = "eval_baselines"


def _baselines_dir(global_folder: Path) -> Path:
    d = global_folder / _BASELINES_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_baseline(result_dict: dict, name: str, global_folder: Path) -> Path:
    """
    Save an evaluation result as a named baseline.

    Args:
        result_dict: Serializable evaluation result dict.
        name: Baseline name (alphanumeric + dashes/underscores).
        global_folder: LSM global folder path.

    Returns:
        Path to saved baseline file.
    """
    path = _baselines_dir(global_folder) / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2)
    logger.info(f"Saved baseline '{name}' to {path}")
    return path


def load_baseline(name: str, global_folder: Path) -> Optional[dict]:
    """
    Load a named baseline.

    Args:
        name: Baseline name.
        global_folder: LSM global folder path.

    Returns:
        Evaluation result dict, or None if not found.
    """
    path = _baselines_dir(global_folder) / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_baselines(global_folder: Path) -> List[str]:
    """
    List all saved baseline names.

    Args:
        global_folder: LSM global folder path.

    Returns:
        Sorted list of baseline names.
    """
    d = _baselines_dir(global_folder)
    return sorted(p.stem for p in d.glob("*.json"))


def delete_baseline(name: str, global_folder: Path) -> bool:
    """
    Delete a named baseline.

    Args:
        name: Baseline name.
        global_folder: LSM global folder path.

    Returns:
        True if deleted, False if not found.
    """
    path = _baselines_dir(global_folder) / f"{name}.json"
    if path.exists():
        path.unlink()
        logger.info(f"Deleted baseline '{name}'")
        return True
    return False
