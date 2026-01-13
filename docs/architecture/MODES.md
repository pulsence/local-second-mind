# Mode System Architecture

Modes define how LSM blends sources and synthesizes answers. This document
explains the internal mode system and its integration points.

## Mode Data Model

A mode is represented by `ModeConfig`:

- `synthesis_style`: `grounded` or `insight`
- `source_policy`: `SourcePolicyConfig`
- `notes`: `NotesConfig`

`SourcePolicyConfig` groups three sub-policies:

- `LocalSourcePolicy`
- `RemoteSourcePolicy`
- `ModelKnowledgePolicy`

## Built-In Modes

When `modes` is not provided in config, LSM populates a built-in registry with:

- `grounded`
- `insight`
- `hybrid`

These are defined in `LSMConfig._get_builtin_modes()`.

## Mode Selection Logic

- Default mode is `query.mode` in `QueryConfig`.
- The query REPL can switch modes with `/mode <name>`.
- `LSMConfig.validate()` ensures the selected mode exists.

## Source Policy Integration

- `local` policy controls `k`, `k_rerank`, and `min_relevance` used in query.
- `remote` policy determines whether remote providers are queried.
- `model_knowledge` policy toggles a warning banner after the answer.

## Notes Integration

`notes` is applied per mode and determines:

- whether `/note` is allowed
- where the note is written
- how the filename is generated

## Extension Points

Custom modes can be defined in `config.json` under `modes`. This allows
project-specific tuning without modifying code.

## Current Limitations

- `rank_strategy` in `RemoteSourcePolicy` is reserved for future merging logic.
- `ModelKnowledgePolicy.require_label` is advisory only.
