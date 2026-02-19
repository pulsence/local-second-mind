# Architecture Overview

This document provides a package-level map of Local Second Mind. See [INDEX.md](./INDEX.md) for the quick start guide.

## Package Details

See individual package documentation in the [architectures/packages](./architectures/packages/) folder:

- [lsm.agents](./architectures/packages/lsm.agents.md) - Agent runtime, tools, memory backends, scheduling, and meta orchestration.
- [lsm.config](./architectures/packages/lsm.config.md) - Typed configuration models, loaders, and validation/serialization helpers.
- [lsm.ingest](./architectures/packages/lsm.ingest.md) - Ingest pipeline: parsing, chunking, embedding, tagging, and manifests.
- [lsm.providers](./architectures/packages/lsm.providers.md) - LLM provider interfaces, factories, and shared provider utilities.
- [lsm.query](./architectures/packages/lsm.query.md) - Query planning, retrieval, reranking, synthesis, notes, and citations.
- [lsm.remote](./architectures/packages/lsm.remote.md) - Remote source provider protocol, chains, and storage.
- [lsm.ui](./architectures/packages/lsm.ui.md) - TUI, shell, web, and desktop interfaces plus shared UI helpers.
- [lsm.vectordb](./architectures/packages/lsm.vectordb.md) - Vector DB providers, migrations, and stats APIs.


## API Reference

Reference docs for CLI commands and configuration surfaces:

- [Config API](./architecture/api-reference/CONFIG.md) - Configuration schema, defaults, and validation notes.
- [Providers API](./architecture/api-reference/PROVIDERS.md) - LLM provider options and settings reference.
- [Remote API](./architecture/api-reference/REMOTE.md) - Remote provider configuration and output schema.
- [REPL / TUI Commands](./architecture/api-reference/REPL.md) - Command reference for shell/TUI usage.
- [Adding Providers](./architecture/api-reference/ADDING_PROVIDERS.md) - Steps for registering custom providers.

## Development Overviews

Developer-focused architecture notes and operational docs:

- [Agents](./architecture/development/AGENTS.md) - Agent system architecture, tools, and runtime workflows.
- [Ingest](./architecture/development/INGEST.md) - Ingest pipeline architecture and data flow.
- [Modes](./architecture/development/MODES.md) - Mode system architecture and policies.
- [Overview](./architecture/development/OVERVIEW.md) - High-level architecture summary.
- [Providers](./architecture/development/PROVIDERS.md) - Provider architecture and integration points.
- [Query](./architecture/development/QUERY.md) - Query pipeline architecture and core components.
- [Security](./architecture/development/SECURITY.md) - Security model, threat inventory, and testing approach.
- [Testing](./architecture/development/TESTING.md) - Test strategy, tiers, and organization.
- [TUI Architecture](./architecture/development/TUI_ARCHITECTURE.md) - TUI architecture, state, and UX conventions.
