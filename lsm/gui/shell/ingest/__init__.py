"""LSM Ingest Shell module - ingest mode REPL and CLI.

This module contains:
- repl: Interactive REPL for ingest management
- commands: Command handlers for the ingest REPL
- display: Display utilities (banner, help, tree formatting)
- cli: Command-line interface for ingest operations
"""

from __future__ import annotations

__all__ = [
    # REPL
    "run_ingest_repl",
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
    "main",
]

from lsm.gui.shell.ingest.repl import run_ingest_repl, handle_command
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
from lsm.gui.shell.ingest.cli import main
