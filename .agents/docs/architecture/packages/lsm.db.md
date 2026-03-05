# lsm.db

Description: Database abstraction layer — cross-backend SQL compatibility, connection resolution, schema ownership, migration, health checks, clustering, enrichment, and transactions.
Folder Path: `lsm/db/`

## Modules

- [compat.py](../lsm/db/compat.py): Cross-backend SQL compatibility layer (dialect detection, placeholder conversion, `execute()`, `fetchone()`, `fetchall()`, `commit()`, `table_exists()`, `db_error()`)
- [connection.py](../lsm/db/connection.py): Connection management — `create_sqlite_connection()`, `resolve_db_path()`, `resolve_connection()` context manager
- [schema.py](../lsm/db/schema.py): Application schema ownership — `ensure_application_schema()` for all non-vector tables
- [schema_version.py](../lsm/db/schema_version.py): Schema version tracking for unified database ingest state
- [tables.py](../lsm/db/tables.py): Centralized table name registry (`TableNames`, `DEFAULT_TABLE_NAMES`)
- [migration.py](../lsm/db/migration.py): Cross-backend state transfer migration framework
- [health.py](../lsm/db/health.py): Startup diagnostics — version mismatches, missing/corrupt databases, legacy provider detection, partial migration detection
- [clustering.py](../lsm/db/clustering.py): Cluster-aware retrieval infrastructure — k-means/HDBSCAN assignment, centroid storage, cluster-filtered queries
- [enrichment.py](../lsm/db/enrichment.py): Post-migration chunk enrichment pipeline (backfills missing metadata in three tiers)
- [transaction.py](../lsm/db/transaction.py): Savepoint-aware transaction context manager (`transaction()`) — `BEGIN/COMMIT` for top-level, `SAVEPOINT/RELEASE` for nested
- [completion.py](../lsm/db/completion.py): Incremental completion mode detection for selective re-ingest
- [job_status.py](../lsm/db/job_status.py): Startup advisory checks for offline jobs (stale/unrun job detection)
