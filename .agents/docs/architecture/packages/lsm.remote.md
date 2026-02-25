# lsm.remote

Description: Remote source providers for external APIs (web search, academic papers, encyclopedias).
Folder Path: `lsm/remote/`

## Modules

- [base.py](../lsm/remote/base.py): BaseRemoteProvider abstract class and RemoteResult
- [factory.py](../lsm/remote/factory.py): Provider factory and registration
- [chain.py](../lsm/remote/chain.py): Remote source chain query
- [storage.py](../lsm/remote/storage.py): Remote source result storage

## Providers

- [brave.py](../lsm/remote/providers/brave.py): Brave Search API provider
- [wikipedia.py](../lsm/remote/providers/wikipedia.py): Wikipedia provider
- [arxiv.py](../lsm/remote/providers/arxiv.py): ArXiv provider
- [base_oai.py](../lsm/remote/providers/base_oai.py): Shared OAI-PMH helpers and BaseOAIProvider
- [rss.py](../lsm/remote/providers/rss.py): RSS/Atom feed provider with caching
- [semantic_scholar.py](../lsm/remote/providers/semantic_scholar.py): Semantic Scholar provider
- [crossref.py](../lsm/remote/providers/crossref.py): CrossRef provider
- [openalex.py](../lsm/remote/providers/openalex.py): OpenAlex provider
- [core.py](../lsm/remote/providers/core.py): CORE provider
- [philpapers.py](../lsm/remote/providers/philpapers.py): PhilPapers provider
- [ixtheo.py](../lsm/remote/providers/ixtheo.py): IxTheo provider
- [oai_pmh.py](../lsm/remote/providers/oai_pmh.py): OAI-PMH provider
