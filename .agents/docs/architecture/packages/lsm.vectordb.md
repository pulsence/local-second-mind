# lsm.vectordb

Description: Vector database abstraction layer with SQLite+sqlite-vec as default and PostgreSQL+pgvector as an alternative backend.
Folder Path: `lsm/vectordb/`

## Modules

- [base.py](../lsm/vectordb/base.py): BaseVectorDBProvider ABC, VectorDBGetResult, VectorDBQueryResult
- [factory.py](../lsm/vectordb/factory.py): create_vectordb_provider() factory with lazy class loading
- [sqlite_vec.py](../lsm/vectordb/sqlite_vec.py): SQLite+sqlite-vec provider with unified `lsm.db` schema
- [postgresql.py../lsm/vectordb/postgresql.py): PostgreSQL + pgvector provider implementation
- [chromadb.py](../lsm/vectordb/chromadb.py): Legacy ChromaDB provider retained for migration tooling only

## Migrations

- [migrations/chromadb_to_postgres.py../lsm/vectordb/migrations/chromadb_to_postgres.py): Migration tool from ChromaDB to PostgreSQL
