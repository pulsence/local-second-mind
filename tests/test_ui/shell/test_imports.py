"""
Tests verifying that all imports work correctly after module restructuring.
"""

import pytest


class TestIngestCLIImports:
    """Test that ingest CLI imports work from lsm.ui.shell.cli."""

    def test_import_from_new_location(self):
        """Test imports from lsm.ui.shell.cli."""
        from lsm.ui.shell.cli import run_ingest, run_build_cli, run_tag_cli, run_wipe_cli
        assert callable(run_ingest)
        assert callable(run_build_cli)
        assert callable(run_tag_cli)
        assert callable(run_wipe_cli)


class TestQueryModuleImports:
    """Test that query module imports work from core lsm.query locations."""

    def test_import_from_core_query(self):
        """Test imports from lsm.query (core module)."""
        from lsm.query.api import query, query_sync, QueryResult
        from lsm.query.context import (
            build_combined_context,
            build_local_context,
            build_remote_context,
            ContextResult,
        )
        from lsm.query.session import SessionState, Candidate
        from lsm.query.planning import prepare_local_candidates, LocalQueryPlan
        assert callable(query)
        assert callable(query_sync)
        assert QueryResult is not None
        assert callable(build_combined_context)
        assert callable(build_local_context)
        assert callable(build_remote_context)
        assert ContextResult is not None
        assert SessionState is not None
        assert Candidate is not None
        assert callable(prepare_local_candidates)
        assert LocalQueryPlan is not None


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
        """Test commands imports from lsm.ui.shell.cli."""
        from lsm.ui.shell.cli import run_ingest
        assert callable(run_ingest)


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
