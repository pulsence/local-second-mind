"""Database-level helpers for schema and migration workflows."""

from .completion import detect_completion_mode, get_stale_files
from .schema_version import (
    SchemaVersionMismatchError,
    check_schema_compatibility,
    get_active_schema_version,
    record_schema_version,
)

__all__ = [
    "detect_completion_mode",
    "get_stale_files",
    "SchemaVersionMismatchError",
    "check_schema_compatibility",
    "get_active_schema_version",
    "record_schema_version",
]
