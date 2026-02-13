# Local Second Mind (LSM)

Local-first RAG for personal knowledge management.

LSM ingests local documents, builds embeddings, retrieves relevant context, and produces cited answers with configurable LLM providers.

## Version

`0.5.0`

## Caveat Emptor

This project is maintained for personal use first.

- **Pull requests are not accepted.** See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.
- Bugs and feature requests are welcome as [GitHub issues](https://github.com/pulsence/local-second-mind/issues).
- Until `v1.0.0`, breaking changes can happen between releases, especially in configuration schema and interfaces.
- Pin versions and review `docs/development/CHANGELOG.md` before upgrading.

## What Is New in 0.5.0

- Tiered real-world testing (`smoke`, `integration`, `live`, `live_llm`, `live_remote`, `live_vectordb`, `docker`) with expanded synthetic and live coverage
- Hardened agent sandbox with adversarial STRIDE security tests, permission gates, environment scrubbing, and log redaction
- New built-in agents: `writing`, `synthesis`, and expanded `curator` workflows
- Agent memory system with SQLite/PostgreSQL backends, API/context injection, CLI/TUI actions, and curation-based memory distillation
- Scheduler engine with interval/cron scheduling, overlap policies, persisted state, and unattended safety defaults
- Meta-agent orchestration with task graphs, sub-agent system tools (`spawn_agent`, `await_agent`, `collect_artifacts`), shared workspaces, and final synthesis artifacts

## Install

```bash
pip install -e .
```

## Quick Start

1. Copy config:

```bash
cp example_config.json config.json
```

2. Add API keys to `.env` (see `.env.example`).

3. Build embeddings:

```bash
lsm ingest build
```

4. Start the TUI:

```bash
lsm
```

## Minimal Config Example

```json
{
  "global": {
    "global_folder": "C:/Users/You/Local Second Mind",
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "batch_size": 32
  },
  "ingest": {
    "roots": ["C:/Users/You/Documents"]
  },
  "llms": {
    "providers": [{ "provider_name": "openai" }],
    "services": {
      "default": { "provider": "openai", "model": "gpt-5.2" }
    }
  }
}
```

## CLI

- `lsm` - launch TUI
- `lsm ingest build [--dry-run] [--force] [--skip-errors]`
- `lsm ingest tag [--max N]`
- `lsm ingest wipe --confirm`

Global flags:

- `--config path/to/config.json`
- `--verbose`
- `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL`
- `--log-file path/to/lsm.log`

## Documentation

- `docs/README.md`
- `docs/user-guide/GETTING_STARTED.md`
- `docs/user-guide/CONFIGURATION.md`
- `docs/user-guide/QUERY_MODES.md`
- `docs/user-guide/NOTES.md`
- `docs/user-guide/REMOTE_SOURCES.md`
- `docs/AGENTS.md`
- `docs/development/CHANGELOG.md`

## License

MIT (`LICENSE`)
