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

- [Planned Timeline](../TODO)
- [Contributing](../CONTRIBUTING.md)
- [Setup](development/SETUP.md)
- [Testing](development/TESTING.md)
- [Development Plan](development/PLAN.md)
- [Changelog](development/CHANGELOG.md)

## What's New in 0.6.0

- Compact TUI layout with density modes (`auto`, `compact`, `comfortable`) for small terminals
- Split modular CSS and refactored Settings screen with MVC architecture
- Interactive agent approvals/replies with multi-agent runtime and real-time log streaming
- TUI startup under 1 second with lazy background ML initialization
- Session-completed agent history, unified log formats, and refined agent screen UX
- Hardened TUI test infrastructure with fast/slow marker split and global state fixtures

## Version

Current release: `0.6.0`
