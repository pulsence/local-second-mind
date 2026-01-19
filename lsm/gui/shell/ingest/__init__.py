"""LSM Ingest Shell module - shared ingest commands and CLI utilities.

Interactive usage is now TUI-only; REPL helpers remain for reuse.
"""

from __future__ import annotations

__all__ = [
    # Command dispatch
    "handle_command",
    # Display
    "print_banner",
    "print_help",
    # Commands
    "handle_info_command",
    "handle_stats_command",
    "handle_explore_command",
    "handle_show_command",
    "handle_search_command",
    "handle_build_command",
    "handle_tag_command",
    "handle_tags_command",
    "handle_wipe_command",
    "handle_vectordb_providers_command",
    "handle_vectordb_status_command",
    # CLI
    "run_build_cli",
    "run_tag_cli",
    "run_wipe_cli",
]

from lsm.gui.shell.ingest.repl import handle_command
from lsm.gui.shell.ingest.display import print_banner, print_help
from lsm.gui.shell.ingest.commands import (
    handle_info_command,
    handle_stats_command,
    handle_explore_command,
    handle_show_command,
    handle_search_command,
    handle_build_command,
    handle_tag_command,
    handle_tags_command,
    handle_wipe_command,
    handle_vectordb_providers_command,
    handle_vectordb_status_command,
)
from lsm.gui.shell.ingest.cli import run_build_cli, run_tag_cli, run_wipe_cli
