# lsm

Description: Local Second Mind - Local-first RAG system for personal knowledge management.
Folder Path: `lsm/`

## Packages

- [lsm.agents](lsm.agents.md): Agent framework, tools, sandbox, memory, scheduler
- [lsm.config](lsm.config.md): Configuration loading and models
- [lsm.ingest](lsm.ingest.md): Document parsing, chunking, embedding pipeline
- [lsm.providers](lsm.providers.md): LLM provider implementations
- [lsm.query](lsm.query.md): Query retrieval, reranking, synthesis
- [lsm.remote](lsm.remote.md): Remote source providers
- [lsm.ui](lsm.ui.md): User interfaces (TUI, Shell, Web, Desktop)
- [lsm.vectordb](lsm.vectordb.md): Vector database abstraction

## Core Modules

- [logging.py](../lsm/logging.py): Logging configuration (get_logger, setup_logging)
- [paths.py](../lsm/paths.py): Path utilities
- [__main__.py](../lsm/__main__.py): CLI entry point
