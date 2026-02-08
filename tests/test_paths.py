from __future__ import annotations

from pathlib import Path

from lsm.paths import (
    ensure_global_folders,
    get_chats_folder,
    get_global_folder,
    get_mode_chats_folder,
    get_notes_folder,
    resolve_relative_path,
)


def test_get_global_folder_override(tmp_path: Path) -> None:
    target = tmp_path / "lsm_home"
    resolved = get_global_folder(target)
    assert resolved == target.resolve()


def test_notes_and_chats_subfolders(tmp_path: Path) -> None:
    root = tmp_path / "home"
    assert get_notes_folder(root) == root.resolve() / "Notes"
    assert get_chats_folder(root) == root.resolve() / "Chats"
    assert get_mode_chats_folder("grounded", root) == root.resolve() / "Chats" / "grounded"


def test_resolve_relative_path_uses_global_folder(tmp_path: Path) -> None:
    root = tmp_path / "home"
    resolved = resolve_relative_path("notes/custom", global_folder=root)
    assert resolved == (root / "notes" / "custom").resolve()


def test_ensure_global_folders_creates_structure(tmp_path: Path) -> None:
    root = tmp_path / "global"
    ensure_global_folders(root)
    assert root.exists()
    assert (root / "Chats").exists()
    assert (root / "Notes").exists()
