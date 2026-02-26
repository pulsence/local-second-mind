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
- [Agents](user-guide/AGENTS.md)

## Developer Documentation

- Developer and agent docs now live in `.agents/docs/`.
- Use `.agents/docs/INDEX.md` as the entry point.

## What's New in 0.7.0

- Agent system overhaul: general, librarian, assistant, and manuscript editor agents
- Meta-agents with parallel task graph planning and `ThreadPoolExecutor` execution engine
- File graphing system for code, text, PDF, and HTML with graph-aware read/edit tooling
- 20+ new remote providers: academic (PubMed, SSRN, PhilArchive), cultural heritage, news, and RSS/Atom
- OAuth2 infrastructure with Gmail, Microsoft Graph, CalDAV, and communication assistant agents
- OpenRouter provider, MCP host support (`global.mcp_servers`), and tiered LLM config (`llms.tiers`)
- Native tool-calling for OpenAI, Anthropic, and Gemini providers with prompt-schema fallback
- Docker runner improvements, WSL2 runner, and sandbox-enforced bash/powershell execution tools

## Version

Current release: `0.7.0`
