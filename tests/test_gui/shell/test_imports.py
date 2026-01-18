"""
Tests verifying that all imports work correctly after module restructuring.
"""

import pytest


class TestIngestModuleImports:
    """Test that ingest module imports work from new and old locations."""

    def test_import_from_new_location(self):
        """Test imports from lsm.gui.shell.ingest."""
        from lsm.gui.shell.ingest import (
            run_ingest_repl,
            handle_command,
            print_banner,
            print_help,
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
        assert callable(run_ingest_repl)
        assert callable(handle_command)
        assert callable(print_banner)
        assert callable(print_help)

    def test_import_from_deprecated_location(self):
        """Test backward compatibility imports from lsm.ingest.repl."""
        from lsm.ingest.repl import (
            run_ingest_repl,
            handle_command,
            print_banner,
            print_help,
        )
        assert callable(run_ingest_repl)
        assert callable(handle_command)


class TestQueryModuleImports:
    """Test that query module imports work from new and old locations."""

    def test_import_from_new_location(self):
        """Test imports from lsm.gui.shell.query."""
        from lsm.gui.shell.query import (
            run_repl,
            run_query_turn,
            print_banner,
            print_help,
            print_source_chunk,
            handle_command,
            COMMAND_HINTS,
        )
        assert callable(run_repl)
        assert callable(run_query_turn)
        assert callable(print_banner)
        assert callable(print_help)
        assert isinstance(COMMAND_HINTS, set)

    def test_import_from_deprecated_location(self):
        """Test backward compatibility imports from lsm.query.repl."""
        from lsm.query.repl import (
            run_repl,
            run_query_turn,
            print_banner,
            print_help,
            handle_command,
            COMMAND_HINTS,
        )
        assert callable(run_repl)
        assert callable(handle_command)


class TestShellModuleImports:
    """Test that shell module imports work correctly."""

    def test_logging_import_from_new_location(self):
        """Test logging imports from lsm.gui.shell.logging."""
        from lsm.gui.shell.logging import (
            setup_logging,
            get_logger,
            configure_logging_from_args,
        )
        assert callable(setup_logging)
        assert callable(get_logger)
        assert callable(configure_logging_from_args)

    def test_logging_import_from_deprecated_location(self):
        """Test backward compatibility logging imports from lsm.cli.logging."""
        from lsm.cli.logging import (
            setup_logging,
            get_logger,
            configure_logging_from_args,
        )
        assert callable(setup_logging)
        assert callable(get_logger)

    def test_unified_shell_import(self):
        """Test unified shell import."""
        from lsm.gui.shell.unified import run_unified_shell
        assert callable(run_unified_shell)

    def test_commands_import_from_new_location(self):
        """Test commands imports from lsm.gui.shell.commands."""
        from lsm.gui.shell.commands import run_ingest, run_query
        assert callable(run_ingest)
        assert callable(run_query)

    def test_commands_import_from_deprecated_location(self):
        """Test backward compatibility commands imports from lsm.commands."""
        from lsm.commands import run_ingest, run_query
        assert callable(run_ingest)
        assert callable(run_query)


class TestMainEntryPointImports:
    """Test that __main__.py uses the new import paths correctly."""

    def test_main_module_imports(self):
        """Test that lsm.__main__ can be imported."""
        import lsm.__main__ as main_module
        assert hasattr(main_module, 'main')
        assert hasattr(main_module, 'build_parser')
        assert callable(main_module.main)
