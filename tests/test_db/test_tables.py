"""Unit tests for lsm.db.tables.TableNames."""

import pytest

from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames


class TestTableNamesDefaults:
    """Default prefix produces lsm_* table names."""

    def test_default_prefix(self):
        tn = TableNames()
        assert tn.prefix == "lsm_"

    def test_chunks(self):
        assert DEFAULT_TABLE_NAMES.chunks == "lsm_chunks"

    def test_manifest(self):
        assert DEFAULT_TABLE_NAMES.manifest == "lsm_manifest"

    def test_schema_versions(self):
        assert DEFAULT_TABLE_NAMES.schema_versions == "lsm_schema_versions"

    def test_reranker_cache(self):
        assert DEFAULT_TABLE_NAMES.reranker_cache == "lsm_reranker_cache"

    def test_agent_memories(self):
        assert DEFAULT_TABLE_NAMES.agent_memories == "lsm_agent_memories"

    def test_agent_memory_candidates(self):
        assert DEFAULT_TABLE_NAMES.agent_memory_candidates == "lsm_agent_memory_candidates"

    def test_agent_schedules(self):
        assert DEFAULT_TABLE_NAMES.agent_schedules == "lsm_agent_schedules"

    def test_cluster_centroids(self):
        assert DEFAULT_TABLE_NAMES.cluster_centroids == "lsm_cluster_centroids"

    def test_graph_nodes(self):
        assert DEFAULT_TABLE_NAMES.graph_nodes == "lsm_graph_nodes"

    def test_graph_edges(self):
        assert DEFAULT_TABLE_NAMES.graph_edges == "lsm_graph_edges"

    def test_embedding_models(self):
        assert DEFAULT_TABLE_NAMES.embedding_models == "lsm_embedding_models"

    def test_job_status(self):
        assert DEFAULT_TABLE_NAMES.job_status == "lsm_job_status"

    def test_stats_cache(self):
        assert DEFAULT_TABLE_NAMES.stats_cache == "lsm_stats_cache"

    def test_remote_cache(self):
        assert DEFAULT_TABLE_NAMES.remote_cache == "lsm_remote_cache"

    def test_migration_progress(self):
        assert DEFAULT_TABLE_NAMES.migration_progress == "lsm_migration_progress"

    def test_vec_chunks_no_prefix(self):
        assert DEFAULT_TABLE_NAMES.vec_chunks == "vec_chunks"

    def test_chunks_fts_no_prefix(self):
        assert DEFAULT_TABLE_NAMES.chunks_fts == "chunks_fts"


class TestTableNamesCustomPrefix:
    """Custom prefix produces app_* table names."""

    def test_custom_prefix(self):
        tn = TableNames(prefix="app_")
        assert tn.chunks == "app_chunks"
        assert tn.manifest == "app_manifest"
        assert tn.schema_versions == "app_schema_versions"
        assert tn.reranker_cache == "app_reranker_cache"
        assert tn.agent_memories == "app_agent_memories"
        assert tn.graph_nodes == "app_graph_nodes"
        assert tn.job_status == "app_job_status"
        assert tn.stats_cache == "app_stats_cache"
        assert tn.remote_cache == "app_remote_cache"
        assert tn.migration_progress == "app_migration_progress"


class TestTableNamesEmptyPrefix:
    """Empty prefix produces unprefixed table names."""

    def test_empty_prefix(self):
        tn = TableNames(prefix="")
        assert tn.chunks == "chunks"
        assert tn.manifest == "manifest"
        assert tn.schema_versions == "schema_versions"
        assert tn.graph_edges == "graph_edges"


class TestTableNamesValidation:
    """Invalid prefix with special characters raises ValueError."""

    def test_invalid_prefix_hyphen(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            TableNames(prefix="my-prefix")

    def test_invalid_prefix_space(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            TableNames(prefix="my prefix")

    def test_invalid_prefix_semicolon(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            TableNames(prefix="drop;--")

    def test_valid_prefix_alphanumeric(self):
        tn = TableNames(prefix="my_app_v2_")
        assert tn.chunks == "my_app_v2_chunks"

    def test_valid_prefix_digits(self):
        tn = TableNames(prefix="v2_")
        assert tn.chunks == "v2_chunks"


class TestApplicationTables:
    """application_tables() returns correct tuple."""

    def test_default_length(self):
        tables = DEFAULT_TABLE_NAMES.application_tables()
        assert len(tables) == 14

    def test_default_values(self):
        tables = DEFAULT_TABLE_NAMES.application_tables()
        assert "lsm_chunks" in tables
        assert "lsm_manifest" in tables
        assert "lsm_schema_versions" in tables
        assert "lsm_reranker_cache" in tables
        assert "lsm_agent_memories" in tables
        assert "lsm_agent_memory_candidates" in tables
        assert "lsm_agent_schedules" in tables
        assert "lsm_cluster_centroids" in tables
        assert "lsm_graph_nodes" in tables
        assert "lsm_graph_edges" in tables
        assert "lsm_embedding_models" in tables
        assert "lsm_job_status" in tables
        assert "lsm_stats_cache" in tables
        assert "lsm_remote_cache" in tables

    def test_excludes_migration_progress(self):
        tables = DEFAULT_TABLE_NAMES.application_tables()
        assert "lsm_migration_progress" not in tables

    def test_excludes_virtual_tables(self):
        tables = DEFAULT_TABLE_NAMES.application_tables()
        assert "vec_chunks" not in tables
        assert "chunks_fts" not in tables

    def test_custom_prefix_length(self):
        tn = TableNames(prefix="app_")
        tables = tn.application_tables()
        assert len(tables) == 14
        assert all(t.startswith("app_") for t in tables)

    def test_empty_prefix_values(self):
        tn = TableNames(prefix="")
        tables = tn.application_tables()
        assert "chunks" in tables
        assert "manifest" in tables


class TestFrozenDataclass:
    """TableNames is frozen (immutable after creation)."""

    def test_frozen(self):
        tn = TableNames()
        with pytest.raises(AttributeError):
            tn.prefix = "other_"
