# lsm.ingest

Description: Document parsing, chunking, embedding pipeline, language detection, and translation.
Folder Path: `lsm/ingest/`

## Modules

- [pipeline.py](../lsm/ingest/pipeline.py): Main ingest orchestration
- [structure_chunking.py](../lsm/ingest/structure_chunking.py): Structure-aware chunking (headings, paragraphs, sentences)
- [chunking.py](../lsm/ingest/chunking.py): Legacy fixed-size chunking
- [models.py](../lsm/ingest/models.py): PageSegment, ParseResult, WriteJob dataclasses
- [parsers.py](../lsm/ingest/parsers.py): Document parsers (PDF, DOCX, MD, HTML, TXT)
- [fs.py](../lsm/ingest/fs.py): File discovery (iter_files) and folder tag collection
- [language.py](../lsm/ingest/language.py): Language detection (langdetect, ISO 639-1)
- [translation.py](../lsm/ingest/translation.py): LLM-based chunk translation
- [tagging.py](../lsm/ingest/tagging.py): AI tagging functionality
- [manifest.py](../lsm/ingest/manifest.py): Manifest load/save and versioning
- [stats.py](../lsm/ingest/stats.py): Collection statistics
- [stats_cache.py](../lsm/ingest/stats_cache.py): StatsCache for caching collection stats
- [explore.py](../lsm/ingest/explore.py): File tree building utilities
- [progress.py](../lsm/ingest/progress.py): Progress tracking utilities
- [utils.py](../lsm/ingest/utils.py): Utility functions
- [api.py](../lsm/ingest/api.py): High-level ingest API
