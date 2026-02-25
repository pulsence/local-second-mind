# lsm.remote

Description: Remote source providers for external APIs — academic, cultural, news, web, and communication sources — with structured output validation, provider chains, and OAuth2 support.
Folder Path: `lsm/remote/`

## Modules

- [base.py](../lsm/remote/base.py): BaseRemoteProvider abstract class and RemoteResult dataclass with stable-ID enforcement
- [validation.py](../lsm/remote/validation.py): Structured output validation; `validate_output()` checks RemoteResult conformance and stable ID presence
- [factory.py](../lsm/remote/factory.py): Provider factory and registration
- [chain.py](../lsm/remote/chain.py): RemoteProviderChain for multi-provider orchestration with field-mapping validation
- [chains.py](../lsm/remote/chains.py): Preconfigured chain definitions (e.g., ScholarlyDiscoveryChain); `build_chain()` enables/disables chains via `remote.chains` config
- [oauth.py](../lsm/remote/oauth.py): OAuth2 authorization code flow, encrypted token storage, and automatic token refresh
- [storage.py](../lsm/remote/storage.py): Result caching helpers and RSS/Atom feed cache (FeedCache)
- [utils.py](../lsm/remote/utils.py): Shared utility helpers

## Provider Sub-Packages

### lsm.remote.providers.academic
Academic and biomedical sources.
- [base_oai.py](../lsm/remote/providers/base_oai.py): Shared OAI-PMH client, parsers (Dublin Core, MARC, DataCite), and BaseOAIProvider
- [arxiv.py](../lsm/remote/providers/academic/arxiv.py): arXiv preprint search
- [core.py](../lsm/remote/providers/academic/core.py): CORE open-access full-text retrieval
- [crossref.py](../lsm/remote/providers/academic/crossref.py): Crossref DOI metadata and bibliographic enrichment
- [openalex.py](../lsm/remote/providers/academic/openalex.py): OpenAlex research metadata with DOI and stable-ID output
- [semantic_scholar.py](../lsm/remote/providers/academic/semantic_scholar.py): Semantic Scholar AI research database
- [unpaywall.py](../lsm/remote/providers/academic/unpaywall.py): Open-access link resolution by DOI
- [pubmed.py](../lsm/remote/providers/academic/pubmed.py): PubMed/PubMed Central via E-utilities; open-access full text when available
- [ssrn.py](../lsm/remote/providers/academic/ssrn.py): SSRN preprints via OAI-PMH
- [philpapers.py](../lsm/remote/providers/academic/philpapers.py): PhilPapers philosophy bibliography
- [philarchive.py](../lsm/remote/providers/academic/philarchive.py): PhilArchive philosophy preprints via OAI-PMH
- [project_muse.py](../lsm/remote/providers/academic/project_muse.py): Project MUSE humanities metadata via OAI-PMH
- [ixtheo.py](../lsm/remote/providers/academic/ixtheo.py): Index Theologicus theology database
- [oai_pmh.py](../lsm/remote/providers/academic/oai_pmh.py): Generic OAI-PMH harvester for arbitrary repositories

### lsm.remote.providers.cultural
Cultural heritage, archives, and museum collections.
- [archive_org.py](../lsm/remote/providers/cultural/archive_org.py): Archive.org Advanced Search API with metadata and file retrieval
- [dpla.py](../lsm/remote/providers/cultural/dpla.py): Digital Public Library of America
- [loc.py](../lsm/remote/providers/cultural/loc.py): Library of Congress JSON/YAML API
- [smithsonian.py](../lsm/remote/providers/cultural/smithsonian.py): Smithsonian Open Access API
- [met.py](../lsm/remote/providers/cultural/met.py): Metropolitan Museum of Art Collection API
- [rijksmuseum.py](../lsm/remote/providers/cultural/rijksmuseum.py): Rijksmuseum data services
- [iiif.py](../lsm/remote/providers/cultural/iiif.py): IIIF Image/Presentation/Content Search APIs
- [wikidata.py](../lsm/remote/providers/cultural/wikidata.py): Wikidata SPARQL endpoint
- [perseus_cts.py](../lsm/remote/providers/cultural/perseus_cts.py): Perseus CTS API for classical text retrieval by CTS URNs

### lsm.remote.providers.news
News sources and RSS feed aggregation.
- [nytimes.py](../lsm/remote/providers/news/nytimes.py): NYTimes Top Stories and Article Search APIs
- [guardian.py](../lsm/remote/providers/news/guardian.py): The Guardian Content API
- [gdelt.py](../lsm/remote/providers/news/gdelt.py): GDELT global news coverage and event data
- [newsapi.py](../lsm/remote/providers/news/newsapi.py): NewsAPI topic aggregation
- [rss.py](../lsm/remote/providers/news/rss.py): RSS/Atom feed provider with caching and seen-item tracking

### lsm.remote.providers.web
Web search providers.
- [brave.py](../lsm/remote/providers/web/brave.py): Brave Search API
- [wikipedia.py](../lsm/remote/providers/web/wikipedia.py): Wikipedia API

### lsm.remote.providers.communication
Communication providers (email and calendar) using OAuth2.
- [models.py](../lsm/remote/providers/communication/models.py): Shared data models (EmailMessage, EmailDraft, CalendarEvent)
- [gmail.py](../lsm/remote/providers/communication/gmail.py): Gmail via Google API (OAuth2) — read, search, draft
- [microsoft_graph_mail.py](../lsm/remote/providers/communication/microsoft_graph_mail.py): Outlook via Microsoft Graph (OAuth2)
- [imap.py](../lsm/remote/providers/communication/imap.py): IMAP/SMTP fallback for self-hosted email
- [google_calendar.py](../lsm/remote/providers/communication/google_calendar.py): Google Calendar API (OAuth2)
- [microsoft_graph_calendar.py](../lsm/remote/providers/communication/microsoft_graph_calendar.py): Microsoft Graph Calendar (OAuth2)
- [caldav.py](../lsm/remote/providers/communication/caldav.py): CalDAV fallback for self-hosted calendars
