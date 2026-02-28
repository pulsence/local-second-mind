# Mode System Architecture

Modes are composition presets that define retrieval behavior, source policy, and
synthesis instructions for query execution.

## Mode Data Model

`ModeConfig` is defined in `lsm/config/models/modes.py` and includes:

- `retrieval_profile`: retrieval strategy name (default `hybrid_rrf`)
- `synthesis_style`: `grounded` or `insight`
- `synthesis_instructions`: full instruction prompt used for synthesis
- `local_policy`: `LocalSourcePolicy(enabled, k, min_relevance)`
- `remote_policy`: `RemoteSourcePolicy(enabled, rank_strategy, max_results, remote_providers)`
- `model_knowledge_policy`: `ModelKnowledgePolicy(enabled, require_label)`
- optional `chats`: per-mode transcript overrides (`auto_save`, `dir`)

`k_rerank` was removed from mode-local policy in v0.8 mode composition. `k`
represents the final local context budget for synthesis.

## Built-In Modes

Built-ins are defined as `GROUNDED_MODE`, `INSIGHT_MODE`, `HYBRID_MODE` and
registered in `BUILT_IN_MODES`.

Defaults:

| Mode | retrieval_profile | local.k | local.min_relevance | remote.enabled | model_knowledge.enabled | synthesis_style |
| --- | --- | --- | --- | --- | --- | --- |
| grounded | hybrid_rrf | 12 | 0.25 | false | false | grounded |
| insight | hybrid_rrf | 8 | 0.0 | true (`max_results=5`) | true | insight |
| hybrid | hybrid_rrf | 12 | 0.15 | true (`max_results=5`) | true | grounded |

`LSMConfig` uses `get_builtin_modes()` when custom `modes` are not provided.

## Prompt Ownership

Mode synthesis prompts are owned by query modules, not providers:

- `SYNTHESIZE_GROUNDED_INSTRUCTIONS`: `lsm/query/prompts.py`
- `SYNTHESIZE_INSIGHT_INSTRUCTIONS`: `lsm/query/prompts.py`

Each mode stores instructions directly via `synthesis_instructions`, allowing
per-mode prompt overrides in user config.

## Mode Resolution

- Default active mode comes from `query.mode`.
- Session override is available in TUI (`/mode <name>`).
- `LSMConfig.validate()` ensures mode existence and applies fallback.

## Backward-Compatibility Behavior

The loader still accepts legacy `source_policy` input and maps it to
`local_policy`, `remote_policy`, and `model_knowledge_policy`.

New serialized config output uses only the v0.8 mode fields.

## Limitations

- `remote_policy.rank_strategy` is currently advisory for some paths.
- `model_knowledge_policy.require_label` is policy metadata; strict enforcement
  depends on synthesis instructions.
