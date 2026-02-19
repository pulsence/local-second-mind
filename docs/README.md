# Local Second Mind Documentation

## Caveat Emptor

This project is maintained for personal use first.

- Issue and pull request response times are not guaranteed.
- Pre-`1.0.0` releases may include breaking changes.
- Check `docs/CHANGELOG.md` before upgrading.

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

## Developer Documentation

- Developer and agent docs now live in `.agents/docs/`.
- Use `.agents/docs/INDEX.md` as the entry point.

## What's New in 0.6.0

- Compact TUI layout with density modes (`auto`, `compact`, `comfortable`) for small terminals
- Split modular CSS and refactored Settings screen with MVC architecture
- Interactive agent approvals/replies with multi-agent runtime and real-time log streaming
- TUI startup under 1 second with lazy background ML initialization
- Session-completed agent history, unified log formats, and refined agent screen UX
- Hardened TUI test infrastructure with fast/slow marker split and global state fixtures

## Version

Current release: `0.6.0`
