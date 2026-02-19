# lsm.vectordb

Description: Vector database abstraction layer with ChromaDB and PostgreSQL+pgvector providers.
Folder Path: `lsm/vectordb/`

## Modules

- [base.py](../lsm/vectordb/base.py): BaseVectorDBProvider ABC, VectorDBGetResult, VectorDBQueryResult
- [factory.py](../lsm/vectordb/factory.py): create_vectordb_provider() factory with lazy class loading
- [chromadb.py](../lsm/vectordb/chromadb.py): ChromaDB provider implementation
- [postgresql.py../lsm/vectordb/postgresql.py): PostgreSQL + pgvector provider implementation

## Migrations

- [migrations/chromadb_to_postgres.py../lsm/vectordb/migrations/chromadb_to_postgres.py): Migration tool from ChromaDB to PostgreSQL
