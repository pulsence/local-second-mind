# Phase 18: Post-Migration Boundary-Drift Rechunk

**Status**: In Progress

After migration from ChromaDB (or other legacy backends), the enrichment pipeline marks
chunks with `start_char = -1` when their text cannot be located in the source file. This
indicates boundary drift — the chunks were created with the old fixed-size chunking strategy
and their boundaries no longer align with the current file content. This phase adds the
ability to re-chunk those files using the current structure-aware chunking strategy, either
interactively at the end of migration or via CLI flags.

---

## 18.1: Drifted Source Path Detection

**Description**: Extend `detect_stale_chunks` and `EnrichmentReport` to surface the
distinct source paths that have boundary-drifted chunks (`start_char = -1`), and populate
them through the enrichment pipeline.

**Tasks**:
- Add a query in `detect_stale_chunks` for distinct source paths where `start_char = -1`
- Add `drifted_source_paths` list to the `tier3` section of the stale report dict
- Update `needs_reingest` to also trigger when drifted paths exist
- Add `drifted_source_paths: tuple[str, ...] = ()` field to `EnrichmentReport`
- In `run_enrichment_pipeline` Tier 3 section, read drifted paths from the stale report,
  append `"boundary drifted: <path>"` entries to `tier3_needed`, and pass the path list
  to the new report field
- Write tests:
  - `detect_stale_chunks` returns `drifted_source_paths` in tier3
  - `EnrichmentReport` includes `drifted_source_paths` from pipeline run
  - `needs_reingest` is true when drifted paths exist

**Files**:
- `lsm/db/enrichment.py`
- `tests/test_db/test_enrichment.py`

**Success criteria**: `detect_stale_chunks` returns drifted source paths in tier3.
`EnrichmentReport.drifted_source_paths` is populated after enrichment runs on a database
with boundary-drifted chunks. All enrichment tests pass.

---

## 18.2: Ingest Pipeline `force_source_paths` Parameter

**Description**: Add a `force_source_paths` parameter to the ingest pipeline and API layer,
enabling exact-path filtering for selective re-ingestion. This parallels the existing
`stale_file_paths` skip-if-not-in-set pattern already in the pipeline.

**Tasks**:
- Add `force_source_paths: Optional[set[str]] = None` parameter to `ingest()` in
  `pipeline.py` (after `force_file_pattern`)
- Add filter logic after existing `stale_file_paths` check: skip files not in the set
- Add to the `force_this_file` condition so matched files bypass manifest freshness checks
- Add `force_source_paths: Optional[set[str]] = None` to `run_ingest()` in `api.py`
  and pass through to `ingest()`

**Files**:
- `lsm/ingest/pipeline.py`
- `lsm/ingest/api.py`

**Success criteria**: Calling `run_ingest(config, force_source_paths={"/path/to/file.md"})`
only processes the specified file and skips all others. Existing ingest behavior is
unchanged when `force_source_paths` is `None`.

---

## 18.3: CLI Rechunk Flags and Interactive Offer

**Description**: Add `--rechunk` and `--skip-rechunk` flags to the `lsm migrate` command,
and implement an interactive rechunk offer that appears after enrichment when
boundary-drifted files are detected.

**Tasks**:
- Add `--rechunk` and `--skip-rechunk` arguments to `migrate_parser` in `__main__.py`
- Add `rechunk` and `skip_rechunk` parameters to `run_migrate_cli()` and `run_migrate()`
- Implement `_handle_rechunk_offer()` in `cli.py`:
  - Print count of drifted files and explain what rechunking does
  - `--skip-rechunk`: print skip message, return
  - `--rechunk`: proceed automatically
  - Neither: interactive `[y/N]` prompt
  - Filter to paths that exist on disk
  - Call `run_ingest(config, force_source_paths=valid_paths)`
  - Print result summary
- Call `_handle_rechunk_offer()` after `_print_enrichment_summary()` in both
  `run_migrate_cli()` and `_run_standalone_enrichment()`
- Update `_print_enrichment_summary()` to show drifted count separately from missing
  summaries

**Files**:
- `lsm/__main__.py`
- `lsm/ui/shell/cli.py`

**Success criteria**: `lsm migrate --help` shows `--rechunk` and `--skip-rechunk` flags.
After migration with drifted files, user is prompted to rechunk (or auto-rechunks with
`--rechunk`). The rechunk uses the current chunking strategy, versions old chunks, and
re-embeds.

---

## 18.4: Debug Phase

User-reported issues and bugs encountered during 18.1–18.3 implementation are resolved
here. The user will provide example output in `<GLOBAL_FOLDER>/Debug/` as needed.

---

## 18.5: Code Review and Changelog

**Tasks**:
- Review all changes from 18.1–18.3 for correctness and backwards compatibility
- Review for dead code or unused imports
- Review tests: no mocks/stubs, no auto-pass tests
- Ensure `--rechunk` does not run unless drifted files exist
- Ensure `force_source_paths` does not interfere with normal ingest when `None`
- Summarize changes in `docs/CHANGELOG.md` under the Unreleased section
- Commit and push

**Files**:
- All files modified in 18.1–18.3
- `docs/CHANGELOG.md`

**Success criteria**: All tests pass. No regressions. Changelog updated.

---

*End of Phase 18.*
