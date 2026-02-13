# Local Second Mind Documentation

## Caveat Emptor

This project is maintained for personal use first.

- Issue and pull request response times are not guaranteed.
- Pre-`1.0.0` releases may include breaking changes.
- Check `development/CHANGELOG.md` before upgrading.

## User Guide

- [Getting Started](user-guide/GETTING_STARTED.md)
- [Configuration](user-guide/CONFIGURATION.md)
- [CLI Usage](user-guide/CLI_USAGE.md)
- [Query Modes](user-guide/QUERY_MODES.md)
- [Notes](user-guide/NOTES.md)
- [Remote Sources](user-guide/REMOTE_SOURCES.md)
- [Integrations](user-guide/INTEGRATIONS.md)
- [Vector Databases](user-guide/VECTOR_DATABASES.md)
- [Local Models](user-guide/LOCAL_MODELS.md)

## Agents

- [Agents Guide](AGENTS.md)

## Architecture

- [Overview](architecture/OVERVIEW.md)
- [Ingest](architecture/INGEST.md)
- [Query](architecture/QUERY.md)
- [Providers](architecture/PROVIDERS.md)
- [Modes](architecture/MODES.md)

## API Reference

- [Config API](api-reference/CONFIG.md)
- [Providers API](api-reference/PROVIDERS.md)
- [Remote API](api-reference/REMOTE.md)
- [REPL / TUI Commands](api-reference/REPL.md)
- [Adding Providers](api-reference/ADDING_PROVIDERS.md)

## Development

- [Contributing](../CONTRIBUTING.md)
- [Setup](development/SETUP.md)
- [Testing](development/TESTING.md)
- [Development Plan](development/PLAN.md)
- [Changelog](development/CHANGELOG.md)

## What's New in 0.5.0

- Expanded testing to tiered smoke/integration/live flows with richer synthetic data and live provider coverage.
- Hardened the agent sandbox with adversarial STRIDE security tests, permission gating, environment scrubbing, and log redaction.
- Added new built-in agents: `writing`, `synthesis`, and `curator`.
- Added agent memory storage/API/tooling with SQLite/PostgreSQL backends and curator-driven memory distillation workflows.
- Added scheduler configuration, engine, and CLI/TUI management for recurring agent runs with safe unattended defaults.
- Added meta-agent orchestration with task graphs, sub-agent system tools, shared workspace execution, and synthesis artifacts.

## Version

Current release: `0.5.0`
