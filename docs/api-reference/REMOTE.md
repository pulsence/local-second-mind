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

### Wikipedia

- Type: `wikipedia`
- Config keys: `language`, `endpoint`, `timeout`, `min_interval_seconds`, `section_limit`,
  `snippet_max_chars`, `include_disambiguation`, `user_agent`
- Environment variable: `LSM_WIKIPEDIA_USER_AGENT` (recommended)

### arXiv

- Type: `arxiv`
- Config keys: `endpoint`, `timeout`, `min_interval_seconds`, `snippet_max_chars`,
  `sort_by`, `sort_order`, `categories`, `user_agent`
- Environment variable: `LSM_ARXIV_USER_AGENT` (recommended)

### Semantic Scholar

- Type: `semantic_scholar`
- Config keys: `api_key`, `endpoint`, `timeout`, `min_interval_seconds`,
  `snippet_max_chars`, `fields_of_study`, `year_range`, `open_access_only`
- Environment variable: `SEMANTIC_SCHOLAR_API_KEY` (optional, increases rate limit)

Provides academic paper search across computer science, neuroscience, and other
disciplines. Includes rich citation metadata, influential citation metrics, and
paper recommendations.

### CORE

- Type: `core`
- Config keys: `api_key`, `endpoint`, `timeout`, `min_interval_seconds`,
  `snippet_max_chars`, `repository_ids`, `year_from`, `year_to`,
  `full_text_only`, `language`
- Environment variable: `CORE_API_KEY` (required)

Aggregates open access research from repositories worldwide. Provides access to
millions of full-text papers from institutional repositories and journals.

### PhilPapers

- Type: `philpapers`
- Config keys: `api_id`, `api_key`, `timeout`, `min_interval_seconds`,
  `snippet_max_chars`, `subject_categories`, `include_books`, `open_access_only`
- Environment variables: `PHILPAPERS_API_ID`, `PHILPAPERS_API_KEY` (optional)

PhilPapers is a comprehensive index and bibliography of philosophy maintained
by the community of philosophers. Uses the OAI-PMH interface for open access
content.

**Subject Categories:**
- Epistemology, Ethics, Metaphysics
- Philosophy of Mind, Language, Science
- Logic, History of Philosophy, Political Philosophy
- Aesthetics, Philosophy of Religion
- Ancient, Medieval, Modern, Continental Philosophy
- And many more specialized areas

**Query Syntax:**
- `author:Name` - Search by author name
- `subject:category` - Search by subject category (e.g., `subject:ethics`)
- `title:phrase` - Search in titles

### Index Theologicus (IxTheo)

- Type: `ixtheo`
- Config keys: `endpoint`, `timeout`, `min_interval_seconds`, `snippet_max_chars`,
  `language`, `traditions`, `search_type`, `include_reviews`, `year_from`, `year_to`
- Environment variable: None required (free access)

Index Theologicus is an international scientific open access bibliography for
theology and religious studies, maintained by the University of Tübingen. Covers
Christianity and dialogue with other religions across confessions, languages,
and media types.

**Supported Languages:** English (en), German (de), French (fr), Italian (it),
Spanish (es), Portuguese (pt), Greek (el), Russian (ru), Chinese (zh), Latin (la),
Hebrew (he), Arabic (ar)

**Religious Traditions:**
- Christian: Catholic, Protestant, Orthodox
- Judaism, Islam, Buddhism, Hinduism
- Comparative Religion, Religious Studies

**Query Syntax:**
- `author:Name` - Search by author
- `title:phrase` - Search in titles
- `subject:topic` - Search by subject
- Bible references in quotes (e.g., `"Mt 5:1-12"`, `"Gen 1:1"`)

**Search Types:** all, title, author, subject, series, toc, isbn, publisher

## Integration in Query Pipeline

Remote providers are used when:

- a mode enables `source_policy.remote.enabled`
- `LSMConfig.get_active_remote_providers()` returns providers with `enabled = true`
  (filtered by `source_policy.remote.remote_providers` when present)

Results are included in the LLM context and displayed after the answer.
