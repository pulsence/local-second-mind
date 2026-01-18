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

### OpenAlex

- Type: `openalex`
- Config keys: `email`, `endpoint`, `timeout`, `min_interval_seconds`,
  `snippet_max_chars`, `year_from`, `year_to`, `open_access_only`, `type`, `concepts`
- Environment variable: `OPENALEX_EMAIL` (recommended for polite pool)

OpenAlex is a fully open catalog of the global research system, indexing over
240 million scholarly works across all academic disciplines. No API key required -
completely free and open access. Successor to Microsoft Academic Graph.

**Features:**
- Cross-disciplinary academic paper search
- Comprehensive metadata including citations, topics, and concepts
- Author and institution disambiguation
- Open access status and PDF links
- Citation network analysis

**Query Syntax:**
- `author:Name` - Search by author name
- `title:phrase` - Search in titles
- `doi:10.xxxx/xxx` - Look up by DOI

**Work Types:** article, book, book-chapter, dataset, dissertation, proceedings,
proceedings-article, report, review, and more

**Rate Limits:** 100,000 requests/day. Adding email increases rate from 1 req/sec
to 10 req/sec.

### Crossref

- Type: `crossref`
- Config keys: `email`, `api_key`, `endpoint`, `timeout`, `min_interval_seconds`,
  `snippet_max_chars`, `year_from`, `year_to`, `type`, `has_full_text`,
  `has_references`, `has_orcid`
- Environment variables: `CROSSREF_EMAIL` (recommended), `CROSSREF_API_KEY` (optional)

Crossref is the official DOI registration agency providing comprehensive
bibliographic metadata for over 150 million scholarly works including journal
articles, books, conference proceedings, and datasets.

**Features:**
- DOI-based metadata lookup
- Comprehensive bibliographic data (authors, venues, publishers)
- ORCID integration
- Citation counts and reference lists
- License and full-text link information
- Journal, funder, and publisher filtering

**Query Syntax:**
- `author:Name` - Search by author name
- `title:phrase` - Search in titles
- `doi:10.xxxx/xxx` - Look up by DOI
- `orcid:0000-0000-0000-0000` - Search by ORCID

**Work Types:** journal-article, book, book-chapter, proceedings-article,
dissertation, dataset, report, and many more

**Rate Limits:** No signup required. Polite pool (with email) provides better
rate limits. Metadata Plus subscription available for production use.

### OAI-PMH (Open Archives Initiative)

- Type: `oai_pmh`
- Config keys: `repository`, `repositories`, `custom_repositories`, `metadata_prefix`,
  `set_spec`, `timeout`, `min_interval_seconds`, `snippet_max_chars`, `user_agent`
- Environment variable: `LSM_OAI_PMH_USER_AGENT` (optional)

OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting) is a
protocol for harvesting metadata from institutional repositories, digital
libraries, and archives worldwide. No API key required for most repositories.

**Features:**
- Generic harvester supporting any OAI-PMH compliant repository
- Multiple metadata formats: Dublin Core (oai_dc), MARC21, DataCite
- Resumption token handling for large result sets
- Set-based filtering for collections
- Pre-configured well-known repositories
- Custom repository configuration support

**Pre-configured Repositories:**
- `arxiv` - Physics, Math, CS, etc. (export.arxiv.org)
- `pubmed` - Biomedical literature (PubMed Central)
- `zenodo` - General research outputs
- `doaj` - Directory of Open Access Journals
- `hal` - French national archive (HAL)
- `repec` - Economics papers
- `europeana` - European cultural heritage
- `philpapers` - Philosophy research

**Configuration Examples:**

Single known repository:
```json
{
  "name": "oai_pmh",
  "type": "oai_pmh",
  "enabled": true,
  "repository": "zenodo"
}
```

Multiple repositories:
```json
{
  "name": "oai_pmh_multi",
  "type": "oai_pmh",
  "enabled": true,
  "repositories": ["arxiv", "zenodo", "pubmed"]
}
```

Custom repository:
```json
{
  "name": "institutional_repo",
  "type": "oai_pmh",
  "enabled": true,
  "custom_repositories": {
    "my_university": {
      "name": "My University Repository",
      "base_url": "https://repository.example.edu/oai",
      "metadata_prefix": "oai_dc",
      "url_template": "https://repository.example.edu/item/{id}"
    }
  }
}
```

**Adding Custom Repositories:**

To add a custom OAI-PMH repository, you need:
1. The base URL of the OAI-PMH endpoint
2. (Optional) The preferred metadata format (default: `oai_dc`)
3. (Optional) A set spec to filter results
4. (Optional) A URL template for linking to records

You can discover OAI-PMH endpoints by:
- Looking for "OAI-PMH" in the repository's documentation
- Checking for common endpoint paths: `/oai`, `/oai2`, `/oai-pmh`, `/oai.pl`
- Using the Identify verb to verify: `curl "https://repo.example.org/oai?verb=Identify"`

**Supported Metadata Formats:**
- `oai_dc` - Dublin Core (required by all OAI-PMH repositories)
- `marc21` / `marcxml` - MARC bibliographic records
- `datacite` - DataCite format for research data

**Rate Limits:** Varies by repository. arXiv requires 3 seconds between requests;
most others allow 1 second or less. Rate limiting is automatically applied
based on repository configuration.

## Integration in Query Pipeline

Remote providers are used when:

- a mode enables `source_policy.remote.enabled`
- `LSMConfig.get_active_remote_providers()` returns providers with `enabled = true`
  (filtered by `source_policy.remote.remote_providers` when present)

Results are included in the LLM context and displayed after the answer.
