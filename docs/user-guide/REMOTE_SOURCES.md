# Remote Sources Guide

Remote sources allow LSM to fetch external information (web search, APIs) during
query sessions. Remote sources are optional and are controlled by mode policies
and provider configuration.

## How Remote Sources Work

1. A mode enables remote sources via `source_policy.remote.enabled`.
2. LSM loads configured remote providers from `remote_providers`.
3. If `source_policy.remote.remote_providers` is set, only those providers are used.
4. Each provider searches for results and returns `RemoteResult` items.
5. Results are displayed after the answer.

Remote results are included in the LLM context and also presented as a
separate section.

## Built-In Providers

LSM ships with eleven built-in remote providers covering STEM, humanities, and cross-disciplinary research:

### General & Web Search
- **Brave Search** - General web search
  - Type: `web_search` or `brave_search`
  - Environment variable: `BRAVE_API_KEY`
- **Wikipedia** - Encyclopedia articles
  - Type: `wikipedia`
  - No API key required (user agent recommended)

### STEM Research
- **arXiv** - Physics, Math, CS, Quantitative Biology, etc.
  - Type: `arxiv`
  - No API key required
- **Semantic Scholar** - Computer Science, Neuroscience, Biomedical
  - Type: `semantic_scholar`
  - Environment variable: `SEMANTIC_SCHOLAR_API_KEY` (optional, increases rate limit)
- **CORE** - Open access research aggregator
  - Type: `core`
  - Environment variable: `CORE_API_KEY` (required)

### Humanities & Philosophy
- **PhilPapers** - Philosophy papers and books
  - Type: `philpapers`
  - Environment variable: `PHILPAPERS_API_ID` and `PHILPAPERS_API_KEY`
- **IxTheo** - Theology and religious studies (Index Theologicus)
  - Type: `ixtheo`
  - No API key required

### Cross-Disciplinary Academic
- **OpenAlex** - Comprehensive academic database (200M+ works)
  - Type: `openalex`
  - No API key required (email recommended for polite pool)
- **Crossref** - DOI metadata and bibliographic search
  - Type: `crossref`
  - No API key required (email recommended)
- **OAI-PMH** - Generic harvester for institutional repositories
  - Type: `oai_pmh`
  - Supports: arXiv, PubMed Central, Zenodo, DOAJ, HAL, RePEc, Europeana, PhilPapers

## Configuring Brave Search

```json
"remote_providers": [
  {
    "name": "brave",
    "type": "web_search",
    "weight": 1.0,
    "api_key": "${BRAVE_API_KEY}",
    "max_results": 5,
    "endpoint": "https://api.search.brave.com/res/v1/web/search"
  }
]
```

Notes:

- `api_key` can be omitted if `BRAVE_API_KEY` is set.
- `endpoint` is optional and defaults to the Brave API endpoint.
- `max_results` can be overridden per provider.

## Configuring Wikipedia

```json
"remote_providers": [
  {
    "name": "wikipedia",
    "type": "wikipedia",
    "weight": 0.7,
    "language": "en",
    "user_agent": "LocalSecondMind/1.0 (contact: you@example.com)",
    "max_results": 5,
    "min_interval_seconds": 1.0,
    "section_limit": 2,
    "snippet_max_chars": 600,
    "include_disambiguation": false
  }
]
```

Notes:

- `user_agent` should identify your app and contact info per Wikipedia policy.
- `language` controls the Wikipedia subdomain (`en`, `de`, etc.).
- `min_interval_seconds` throttles requests to respect API rate limits.

## Configuring arXiv

```json
"remote_providers": [
  {
    "name": "arxiv",
    "type": "arxiv",
    "weight": 0.9,
    "user_agent": "LocalSecondMind/1.0 (contact: you@example.com)",
    "max_results": 5,
    "min_interval_seconds": 3.0,
    "sort_by": "relevance",
    "sort_order": "descending",
    "categories": ["cs.AI", "cs.LG"]
  }
]
```

Notes:

- `categories` filters results to arXiv categories when provided.
- `sort_by` supports `relevance`, `lastUpdatedDate`, or `submittedDate`.

### arXiv Field Labels

Use these labels in your query text to target specific metadata fields:

- `title:` for title matches
- `author:` for author names
- `abstract:` or `topic:` for abstract text
- `cat:` or `category:` for arXiv category filters
- `all:` for a full-record search

Examples:

- `title:"pulsar timing"`
- `author:lorimer`
- `abstract:"fast radio bursts"`
- `cat:astro-ph.HE`
- `title:"pulsar timing" author:lorimer`

## Configuring Semantic Scholar

```json
"remote_providers": [
  {
    "name": "semantic_scholar",
    "type": "semantic_scholar",
    "weight": 0.9,
    "api_key": "${SEMANTIC_SCHOLAR_API_KEY}",
    "max_results": 5,
    "min_interval_seconds": 1.0,
    "fields_of_study": ["Computer Science", "Neuroscience"],
    "year_range": "2020-2024",
    "open_access_only": false
  }
]
```

Notes:

- `api_key` is optional but recommended for higher rate limits.
- `fields_of_study` filters to specific academic disciplines.
- `year_range` format is `YYYY-YYYY` or `YYYY-` for open-ended.
- `open_access_only` restricts results to papers with free PDF access.

### Semantic Scholar Features

- **Citations and References**: Retrieve papers that cite or are cited by a paper.
- **Influential Citations**: Filter by influential citation count.
- **Paper Recommendations**: Get recommended papers based on a seed paper.
- **Author Papers**: Find all papers by a specific author.

## Configuring CORE

```json
"remote_providers": [
  {
    "name": "core",
    "type": "core",
    "weight": 0.8,
    "api_key": "${CORE_API_KEY}",
    "max_results": 5,
    "min_interval_seconds": 1.0,
    "repository_ids": [],
    "year_from": 2020,
    "year_to": 2024,
    "full_text_only": false,
    "language": "en"
  }
]
```

Notes:

- `api_key` is required. Register at https://core.ac.uk/services/api
- `repository_ids` filters to specific institutional repositories.
- `full_text_only` restricts to papers with full text available.
- `language` filters by language code (e.g., "en", "de").

### CORE Features

- **Open Access**: Access millions of open access research papers.
- **Full Text**: Download full-text PDFs when available.
- **Repository Filtering**: Search within specific institutional repositories.
- **Multi-Language**: Filter by language for non-English research.

## Enabling Remote Sources in a Mode

```json
"modes": [
  {
    "name": "hybrid",
    "synthesis_style": "grounded",
    "source_policy": {
      "local": { "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
      "remote": {
        "enabled": true,
        "rank_strategy": "weighted",
        "max_results": 5,
        "remote_providers": ["brave", "arxiv"]
      },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
  }
]
```

## Provider Weighting and Ranking

Each provider has a `weight` (0.0-1.0) that affects result ranking. When
`rank_strategy` is set to `"weighted"`, results are sorted by `score * weight`.

### Global Weights

Set default weights in the provider configuration:

```json
"remote_providers": [
  {"name": "arxiv", "type": "arxiv", "weight": 0.9},
  {"name": "wikipedia", "type": "wikipedia", "weight": 0.7}
]
```

### Per-Mode Weight Overrides

Override weights for specific modes using inline provider references:

```json
"modes": [
  {
    "name": "philosophy_research",
    "source_policy": {
      "remote": {
        "enabled": true,
        "rank_strategy": "weighted",
        "remote_providers": [
          {"source": "philpapers", "weight": 0.95},
          {"source": "openalex", "weight": 0.7},
          {"source": "wikipedia", "weight": 0.5}
        ]
      }
    }
  }
]
```

You can mix string names (use global weight) and objects (override weight):

```json
"remote_providers": ["brave", {"source": "arxiv", "weight": 0.95}]
```

### Rank Strategies

- `"weighted"` - Sort results by `score * weight` (recommended)
- `"sequential"` - Return results in provider order
- `"interleaved"` - Alternate between providers

## Field-Specific Provider Recommendations

Choose providers based on your research domain:

### Sciences (STEM)

| Field | Recommended Providers |
|-------|----------------------|
| Physics | arXiv, Semantic Scholar, OpenAlex |
| Mathematics | arXiv, Semantic Scholar, OpenAlex |
| Computer Science | arXiv, Semantic Scholar, OpenAlex, CORE |
| Biology | Semantic Scholar, OpenAlex, CORE, PubMed (via OAI-PMH) |
| Medicine | Semantic Scholar, OpenAlex, CORE, PubMed (via OAI-PMH) |
| General STEM | OpenAlex, Crossref, CORE, Semantic Scholar |

### Humanities

| Field | Recommended Providers |
|-------|----------------------|
| Philosophy | PhilPapers, OpenAlex |
| Theology | IxTheo, PhilPapers, OpenAlex |
| Religious Studies | IxTheo, PhilPapers, OpenAlex |
| History | OpenAlex, Crossref, institutional repos (via OAI-PMH) |
| Classics | OpenAlex, Crossref |

### Social Sciences

| Field | Recommended Providers |
|-------|----------------------|
| Economics | OpenAlex, Crossref, RePEc (via OAI-PMH) |
| Psychology | Semantic Scholar, OpenAlex |
| Sociology | OpenAlex, Crossref, CORE |
| General | OpenAlex, Crossref, CORE |

### Cross-Disciplinary

| Use Case | Recommended Providers |
|----------|----------------------|
| Citation/DOI lookup | Crossref |
| Open access papers | CORE, OpenAlex, Zenodo (via OAI-PMH) |
| Comprehensive search | OpenAlex, Crossref, Semantic Scholar |
| Preprints | arXiv, Zenodo (via OAI-PMH) |

## Terminal Commands

LSM provides several commands for managing remote providers during a session:

### List Providers

```
/remote-providers
```

Shows all registered provider types and configured providers with weights and
API key availability.

### Test a Provider

```
/remote-search <provider> <query>
```

Search a specific provider and display results. Useful for testing configuration.

Example:
```
/remote-search wikipedia machine learning
/remote-search arxiv quantum computing
/remote-search philpapers epistemology
```

### Search All Providers

```
/remote-search-all <query>
```

Search all configured providers simultaneously. Results are deduplicated and sorted
by weighted score.

### Adjust Provider Weight

Set provider weights in `config.json` under `remote_providers`:

```json
"remote_providers": [
  {"name": "arxiv", "type": "arxiv", "weight": 0.95},
  {"name": "wikipedia", "type": "wikipedia", "weight": 0.5}
]
```

## Adding Custom Remote Providers

To add a custom provider:

1. Implement `BaseRemoteProvider`.
2. Register it with `register_remote_provider`.
3. Reference it by `type` in `remote_providers` entries.

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
