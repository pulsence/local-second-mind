# Remote Provider API Reference

This document describes the remote provider interface for web search and other
external sources.

## RemoteResult

Location: `lsm/query/remote/base.py`

Fields:

- `title: str`
- `url: str`
- `snippet: str`
- `score: float = 1.0`
- `metadata: dict[str, Any] = {}`

## BaseRemoteProvider

Location: `lsm/query/remote/base.py`

### search(query, max_results=5) -> list[RemoteResult]

Search remote sources and return results.

### get_name() -> str

Return a human-readable provider name.

### validate_config() -> None

Optional validation hook. Raise `ValueError` for invalid config.

## Factory APIs

Location: `lsm/query/remote/factory.py`

### register_remote_provider(provider_type, provider_class)

Registers a provider class for a type string.

### create_remote_provider(provider_type, config)

Instantiates a provider for a type string.

### get_registered_providers()

Returns the registry mapping types to classes.

## Built-In Providers

### Brave Search

- Type: `web_search` and `brave_search`
- Config keys: `api_key`, `endpoint`, `timeout`
- Environment variable: `BRAVE_API_KEY`

## Integration in Query Pipeline

Remote providers are used when:

- a mode enables `source_policy.remote.enabled`
- `LSMConfig.get_active_remote_providers()` returns providers with `enabled = true`

Results are displayed after the answer.
