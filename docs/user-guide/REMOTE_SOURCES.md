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

## Structured Provider Protocol (Phase 8.1)

All built-in remote providers now expose a dict-based protocol in addition to
plain text query search:

- `get_input_fields()` returns provider-declared accepted input fields.
- `get_output_fields()` returns normalized output schema metadata.
- `get_description()` returns a concise provider capability description.
- `search_structured(input_dict, max_results)` accepts dict inputs and returns
  normalized result dicts with standard keys:
  - `url`, `title`, `description`, `doi`, `authors`, `year`, `score`,
    `metadata`.

This protocol is intended for tool-calling and agent-style orchestration where
providers need self-describing schemas instead of free-form prompts.

## Provider Chains

You can define `remote_provider_chains` in config to run providers as a
pipeline. Each link can map prior output fields into next-link input fields
using `"output:input"` mappings (for example, `"doi:doi"`).

Preconfigured chains can be enabled via the top-level `remote.chains` list.
The built-in `scholarly_discovery` chain runs OpenAlex → Crossref →
Unpaywall → CORE and downloads full text into `<global_folder>/Downloads/scholarly`.

```json
"remote": {
  "chains": ["scholarly_discovery"]
}
```

## Built-In Providers

LSM ships with remote providers spanning web search, academic databases,
cultural archives, and news APIs.

### General & Web Search
- **Brave Search** - General web search
- Type: `web_search` or `brave_search`
- Environment variable: `BRAVE_API_KEY`
- **Wikipedia** - Encyclopedia articles
- Type: `wikipedia`
- Environment variable: `LSM_WIKIPEDIA_USER_AGENT` (recommended)

### Academic & Biomedical
- **arXiv** - Physics, math, CS, quantitative biology
- Type: `arxiv`
- **Semantic Scholar** - Computer science, neuroscience, biomedical
- Type: `semantic_scholar`
- Environment variable: `SEMANTIC_SCHOLAR_API_KEY` (optional, higher rate limits)
- **OpenAlex** - Cross-disciplinary academic works
- Type: `openalex`
- Environment variables: `OPENALEX_EMAIL` (polite pool), `OPENALEX_API_KEY` (recommended)
- **Crossref** - DOI metadata and bibliographic search
- Type: `crossref`
- Environment variables: `CROSSREF_EMAIL` (polite pool), `CROSSREF_API_KEY` (optional)
- **Unpaywall** - Open access DOI resolver
- Type: `unpaywall`
- Environment variable: `UNPAYWALL_EMAIL` (required by policy)
- **CORE** - Open access aggregation + full text
- Type: `core`
- Environment variable: `CORE_API_KEY` (required)
- **PubMed** - Biomedical and PubMed Central metadata
- Type: `pubmed`
- Environment variable: `PUBMED_API_KEY` (optional)
- **SSRN** - Preprints (OAI-PMH)
- Type: `ssrn`
- **PhilArchive** - Philosophy preprints (OAI-PMH)
- Type: `philarchive`
- **Project MUSE** - Humanities metadata (OAI-PMH)
- Type: `project_muse`
- **PhilPapers** - Philosophy papers and books
- Type: `philpapers`
- Environment variables: `PHILPAPERS_API_ID`, `PHILPAPERS_API_KEY`
- **IxTheo** - Theology and religious studies
- Type: `ixtheo`
- **OAI-PMH** - Generic harvester for institutional repositories
- Type: `oai_pmh`
- Environment variable: `LSM_OAI_PMH_USER_AGENT` (recommended)

### Cultural Heritage & Archives
- **Archive.org** - Books, media, and documents
- Type: `archive_org`
- **DPLA** - Digital Public Library of America
- Type: `dpla`
- Environment variable: `DPLA_API_KEY` (required)
- **Library of Congress** - LOC JSON API
- Type: `loc`
- **Smithsonian** - Smithsonian Open Access API
- Type: `smithsonian`
- Environment variable: `SMITHSONIAN_API_KEY` (required)
- **Met Museum** - Metropolitan Museum of Art collection
- Type: `met`
- **Rijksmuseum** - Rijksmuseum data services
- Type: `rijksmuseum`
- Environment variable: `RIJKSMUSEUM_API_KEY` (required)
- **IIIF** - Image/Presentation/Content Search APIs
- Type: `iiif`
- **Wikidata** - SPARQL endpoint
- Type: `wikidata`
- **Perseus CTS** - Classical text passages by CTS URN
- Type: `perseus_cts`

### News Sources
- **NYTimes** - Top Stories + Article Search
- Type: `nytimes`
- Environment variable: `NYTIMES_API_KEY` (required)
- **The Guardian** - Content API
- Type: `guardian`
- Environment variable: `GUARDIAN_API_KEY` (required)
- **GDELT** - Global news/event coverage
- Type: `gdelt`
- **NewsAPI** - Topic aggregation
- Type: `newsapi`
- Environment variable: `NEWSAPI_API_KEY` (required)
- **RSS/Atom** - Any feed URL
- Type: `rss`

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
      "local": { "min_relevance": 0.25, "k": 12 },
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

See `.agents/docs/architecture/api-reference/REMOTE.md` for the provider interface.

## API Key Management

- Prefer environment variables (`BRAVE_API_KEY`).
- Avoid committing API keys to source control.
- Use `.env` and `.env.example` for local development.

## Troubleshooting

- If remote results are empty, verify API keys and network access.
- If you see `Unknown remote provider type`, check the `type` value.
- If the Brave API returns an error, validate subscription status and rate
  limits.
