"""
Tests verifying that all imports work correctly after module restructuring.
"""

import pytest


class TestIngestModuleImports:
    """Test that ingest module imports work from new locations."""

    def test_import_from_new_location(self):
        """Test imports from lsm.ui.shell.ingest."""
        from lsm.ui.shell.ingest import (
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
        assert callable(handle_command)
        assert callable(print_banner)
        assert callable(print_help)


class TestQueryModuleImports:
    """Test that query module imports work from core lsm.query locations."""

    def test_import_from_core_query(self):
        """Test imports from lsm.query (core module)."""
        from lsm.query.execution import run_repl, run_query_turn
        from lsm.query.display import (
            print_banner,
            print_help,
            print_source_chunk,
        )
        from lsm.query.commands import handle_command, COMMAND_HINTS
        assert callable(run_repl)
        assert callable(run_query_turn)
        assert callable(print_banner)
        assert callable(print_help)
        assert isinstance(COMMAND_HINTS, set)


class TestShellModuleImports:
    """Test that shell module imports work correctly."""

    def test_logging_import_from_new_location(self):
        """Test logging imports from lsm.logging."""
        from lsm.logging import (
            setup_logging,
            get_logger,
            configure_logging_from_args,
        )
        assert callable(setup_logging)
        assert callable(get_logger)
        assert callable(configure_logging_from_args)


    def test_commands_import_from_new_location(self):
        """Test commands imports from lsm.ui.shell.commands."""
        from lsm.ui.shell.commands import run_ingest, run_query
        assert callable(run_ingest)
        assert callable(run_query)


class TestRemoteModuleImports:
    """Test that remote module imports work from lsm.remote."""

    def test_import_base_classes(self):
        """Test imports of base classes from lsm.remote."""
        from lsm.remote import (
            BaseRemoteProvider,
            RemoteResult,
        )
        assert BaseRemoteProvider is not None
        assert RemoteResult is not None

    def test_import_factory_functions(self):
        """Test imports of factory functions from lsm.remote."""
        from lsm.remote import (
            create_remote_provider,
            register_remote_provider,
            get_registered_providers,
        )
        assert callable(create_remote_provider)
        assert callable(register_remote_provider)
        assert callable(get_registered_providers)

    def test_import_providers(self):
        """Test imports of provider classes from lsm.remote."""
        from lsm.remote import (
            BraveSearchProvider,
            WikipediaProvider,
            ArXivProvider,
            SemanticScholarProvider,
            COREProvider,
            PhilPapersProvider,
            IxTheoProvider,
            OpenAlexProvider,
            CrossrefProvider,
        )
        assert BraveSearchProvider is not None
        assert WikipediaProvider is not None
        assert ArXivProvider is not None


class TestMainEntryPointImports:
    """Test that __main__.py uses the new import paths correctly."""

    def test_main_module_imports(self):
        """Test that lsm.__main__ can be imported."""
        import lsm.__main__ as main_module
        assert hasattr(main_module, 'main')
        assert hasattr(main_module, 'build_parser')
        assert callable(main_module.main)
