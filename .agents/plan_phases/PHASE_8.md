# Phase 8: Remote Source Providers

**Why eighth:** Protocol infrastructure (OAI consolidation, RSS) from Phase 7 is now available. Providers are grouped by domain within the phase.

**Depends on:** Phase 7.2 (OAI consolidation), Phase 7.4 (RSS reader)

**Package structure:** Remote providers must be organized into domain-specific sub-packages under `lsm/remote/providers/`:
- `lsm/remote/providers/academic/` — scholarly, biomedical, philosophy, humanities
- `lsm/remote/providers/cultural/` — archives, museums, cultural heritage
- `lsm/remote/providers/news/` — news sources and RSS
- `lsm/remote/providers/web/` — web search (Brave, etc.)
- Existing providers (`brave.py`, `wikipedia.py`, `arxiv.py`, etc.) must be migrated into their appropriate sub-packages.
- `lsm/remote/providers/__init__.py` re-exports for backwards compatibility.

| Task | Description | Depends On |
|------|-------------|------------|
| 8.0 | Provider sub-package restructure | None |
| 8.1 | Structured output validation framework | None |
| 8.2 | Scholarly discovery pipeline | 7.2, 8.1 |
| 8.3 | Academic & biomedical providers | 7.2, 8.1 |
| 8.4 | Cultural heritage & archive providers | 8.1 |
| 8.5 | News providers | 7.4, 8.1 |
| 8.6 | Specialized protocol providers | 8.1 |
| 8.7 | Tests and documentation | 8.0–8.6 |

## 8.0: Provider Sub-Package Restructure
- **Description:** Reorganize existing and new remote providers into domain-specific sub-packages.
- **Tasks:**
  - Create sub-package directories: `academic/`, `cultural/`, `news/`, `web/`.
  - Move existing providers into appropriate sub-packages (e.g., `brave.py` → `web/brave.py`, `arxiv.py` → `academic/arxiv.py`, `openalex.py` → `academic/openalex.py`, `wikipedia.py` → `web/wikipedia.py`).
  - Update `lsm/remote/providers/__init__.py` re-exports and factory registration so existing imports and config references continue to work.
  - Update all test imports.
- **Files:**
  - `lsm/remote/providers/*/`
  - `lsm/remote/__init__.py`
  - `lsm/remote/factory.py`
  - `tests/test_providers/remote/`
- **Success criteria:** All existing provider tests pass from new locations. Factory registration unchanged. Config references unchanged.

## 8.1: Structured Output Validation Framework
- **Description:** Ensure all remote providers produce stable, structured output suitable for use in `RemoteProviderChain`. This is a prerequisite for all provider implementations.
- **Tasks:**
  - Define a structured output contract: every provider's `search()` and `search_structured()` must return `RemoteResult` objects with all required fields populated (title, url, snippet, score, metadata with stable IDs).
  - Add a `validate_output(results: List[RemoteResult]) -> List[str]` utility that checks output conformance and returns a list of violations.
  - Add `get_output_fields()` contract enforcement: every provider must declare its output fields, and `RemoteProviderChain` must validate that a link's output fields match the next link's expected input fields.
  - Add integration test base class `RemoteProviderOutputTest` that all provider tests can inherit from to automatically validate output structure.
  - Audit and update all existing providers to conform to the validated output contract.
- **Files:**
  - `lsm/remote/base.py`
  - `lsm/remote/chain.py`
  - `lsm/remote/validation.py`
  - `tests/test_providers/remote/test_base.py`
- **Success criteria:** All providers pass output validation. Chain field mapping is validated at construction time. New providers automatically inherit output validation tests.

## 8.2: Scholarly Discovery Pipeline
- **Description:** Chain OpenAlex → Crossref → Unpaywall → CORE to discover, enrich, resolve open-access links, and retrieve full-text scholarly documents.
- **Tasks:**
  - Extend existing `openalex.py` to return stable IDs and DOI metadata in structured output.
  - Extend existing `crossref.py` to enrich metadata and resolve DOI/ISSN.
  - Implement `unpaywall.py` provider for open-access link resolution by DOI.
  - Wire a `RemoteProviderChain` that downloads full text when available and hands off to ingest pipeline.
  - Update existing `core.py` as the full-text retrieval fallback.
  - Implement preconfigured chain objects (e.g., `ScholarlyDiscoveryChain`) that bundle the standard OpenAlex → Crossref → Unpaywall → CORE pipeline into reusable, named chain instances.
  - Add config support for enabling/disabling preconfigured chains: users can list enabled chains under a `remote.chains` section in their config file, and disabled chains are skipped during provider resolution.
- **Files:**
  - `lsm/remote/providers/academic/openalex.py`
  - `lsm/remote/providers/academic/crossref.py`
  - `lsm/remote/providers/academic/unpaywall.py`
  - `lsm/remote/providers/academic/core.py`
  - `lsm/remote/chain.py`
- **Success criteria:** A DOI query yields normalized metadata + full-text retrieval when available. Chain executes end-to-end. Preconfigured chains can be enabled/disabled via config. All providers pass structured output validation (8.1).

## 8.3: Academic & Biomedical Providers
- **Description:** Add discipline-specific academic sources.
- **Providers:**
  - `pubmed.py` — PubMed/PubMed Central via E-utilities. Open-access full text when available.
  - `ssrn.py` — SSRN preprints and metadata.
  - `philarchive.py` — PhilArchive philosophy preprints.
  - `project_muse.py` — Project MUSE humanities metadata.
- **Files:**
  - `lsm/remote/providers/academic/pubmed.py`
  - `lsm/remote/providers/academic/ssrn.py`
  - `lsm/remote/providers/academic/philarchive.py`
  - `lsm/remote/providers/academic/project_muse.py`
- **Success criteria:** Each provider returns normalized `RemoteResult` with open-access links when possible. Passes output validation.

## 8.4: Cultural Heritage & Archive Providers
- **Description:** Add providers for archives, museums, and cultural heritage datasets.
- **Providers:**
  - `archive_org.py` — Archive.org metadata + file retrieval.
  - `dpla.py` — Digital Public Library of America.
  - `loc.py` — Library of Congress JSON/YAML API.
  - `smithsonian.py` — Smithsonian Open Access API.
  - `met.py` — Metropolitan Museum of Art Collection API.
  - `rijksmuseum.py` — Rijksmuseum data services.
  - `iiif.py` — IIIF Image/Presentation/Content Search APIs.
  - `wikidata.py` — Wikidata SPARQL endpoint.
- **Files:**
  - `lsm/remote/providers/cultural/archive_org.py`
  - `lsm/remote/providers/cultural/dpla.py`
  - `lsm/remote/providers/cultural/loc.py`
  - `lsm/remote/providers/cultural/smithsonian.py`
  - `lsm/remote/providers/cultural/met.py`
  - `lsm/remote/providers/cultural/rijksmuseum.py`
  - `lsm/remote/providers/cultural/iiif.py`
  - `lsm/remote/providers/cultural/wikidata.py`
- **Success criteria:** Each provider returns normalized results with stable IDs and source URLs. Passes output validation.

## 8.5: News Providers
- **Description:** Implement news sources with API-backed retrieval. Sources without APIs use the RSS reader from 7.4.
- **Providers:**
  - `nytimes.py` — NYTimes Top Stories + Article Search APIs.
  - `guardian.py` — The Guardian Content API.
  - `gdelt.py` — GDELT for global news coverage and event data.
  - `newsapi.py` — NewsAPI for topic aggregation.
  - RSS feeds for sources without dedicated APIs (via `rss.py` provider).
- **Files:**
  - `lsm/remote/providers/news/nytimes.py`
  - `lsm/remote/providers/news/guardian.py`
  - `lsm/remote/providers/news/gdelt.py`
  - `lsm/remote/providers/news/newsapi.py`
- **Success criteria:** News queries return current articles with source, timestamp, and canonical URL. Passes output validation.

## 8.6: Specialized Protocol Providers
- **Providers:**
  - `perseus_cts.py` — Perseus CTS API for classical text retrieval by CTS URNs.
- **Files:**
  - `lsm/remote/providers/cultural/perseus_cts.py`
- **Success criteria:** CTS URN queries return text passages with citation metadata. Passes output validation.

## 8.7: Tests and Documentation
- **Description:** Validate provider behavior, authentication, and schema normalization.
- **Tasks:**
  - Add tests for provider config validation and output normalization.
  - Add integration tests for at least one provider per category.
  - Document API keys and configuration in `docs/` and `.env.example`.
- **Files:**
  - `tests/test_remote/`
  - `docs/`
  - `.env.example`
  - `example_config.json`
- **Success criteria:** Provider integrations are tested and documented with real config examples.
