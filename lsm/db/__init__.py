"""Database-level helpers for schema and migration workflows."""

from .completion import detect_completion_mode, get_stale_files
from .connection import (
    create_sqlite_connection,
    resolve_db_path,
    resolve_postgres_connection_factory,
    resolve_sqlite_connection,
    resolve_vectordb_provider_name,
)
from .migration import (
    MigrationSource,
    MigrationTarget,
    MigrationValidationError,
    migrate,
    validate_migration,
)
from .schema import APPLICATION_TABLES, ensure_application_schema
from .schema_version import (
    SchemaVersionMismatchError,
    check_schema_compatibility,
    get_active_schema_version,
    record_schema_version,
)
from .transaction import transaction

__all__ = [
    "create_sqlite_connection",
    "resolve_db_path",
    "resolve_postgres_connection_factory",
    "resolve_sqlite_connection",
    "resolve_vectordb_provider_name",
    "detect_completion_mode",
    "get_stale_files",
    "APPLICATION_TABLES",
    "ensure_application_schema",
    "MigrationSource",
    "MigrationTarget",
    "MigrationValidationError",
    "migrate",
    "validate_migration",
    "SchemaVersionMismatchError",
    "check_schema_compatibility",
    "get_active_schema_version",
    "record_schema_version",
    "transaction",
]
