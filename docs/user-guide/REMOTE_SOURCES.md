# Remote Sources Guide

Remote sources allow LSM to fetch external information (web search, APIs) during
query sessions. Remote sources are optional and are controlled by mode policies
and provider configuration.

## How Remote Sources Work

1. A mode enables remote sources via `source_policy.remote.enabled`.
2. LSM loads configured remote providers from `remote_providers`.
3. Each provider searches for results and returns `RemoteResult` items.
4. Results are displayed after the answer.

Current behavior: remote results are not yet merged into the LLM context.
They are presented as a separate section.

## Built-In Providers

LSM ships with a Brave Search provider:

- Type: `web_search` or `brave_search`
- Environment variable: `BRAVE_API_KEY`

## Configuring Brave Search

```json
"remote_providers": {
  "brave": {
    "type": "web_search",
    "enabled": true,
    "weight": 1.0,
    "api_key": "${BRAVE_API_KEY}",
    "max_results": 5,
    "endpoint": "https://api.search.brave.com/res/v1/web/search"
  }
}
```

Notes:

- `api_key` can be omitted if `BRAVE_API_KEY` is set.
- `endpoint` is optional and defaults to the Brave API endpoint.
- `max_results` can be overridden per provider.

## Enabling Remote Sources in a Mode

```json
"modes": {
  "hybrid": {
    "synthesis_style": "grounded",
    "source_policy": {
      "local": { "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
      "remote": { "enabled": true, "rank_strategy": "weighted", "max_results": 5 },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
  }
}
```

## Provider Weighting and Ranking

`weight` and `rank_strategy` are reserved for future blending strategies. The
current implementation simply aggregates results in provider order.

## Adding Custom Remote Providers

To add a custom provider:

1. Implement `BaseRemoteProvider`.
2. Register it with `register_remote_provider`.
3. Reference it by `type` in `remote_providers`.

See `docs/api-reference/REMOTE.md` for the provider interface.

## API Key Management

- Prefer environment variables (`BRAVE_API_KEY`).
- Avoid committing API keys to source control.
- Use `.env` and `.env.example` for local development.

## Troubleshooting

- If remote results are empty, verify API keys and network access.
- If you see `Unknown remote provider type`, check the `type` value.
- If the Brave API returns an error, validate subscription status and rate
  limits.
