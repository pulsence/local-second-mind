# REPL Command Reference

LSM provides a unified shell and two REPLs (ingest and query). This document
lists all supported commands, syntax, and side effects.

## Unified Shell

Run with:

```bash
python -m lsm shell --config config.json
```

Global commands:

- `/ingest` or `/i`: switch to ingest context
- `/query` or `/q`: switch to query context
- `/help`: show global or context-specific help
- `/exit` or `/quit`: exit the shell

## Ingest REPL Commands

Context: `lsm/ingest/repl.py`

- `/info`:
  - show collection name, ID, and chunk count
- `/stats`:
  - compute detailed stats, file distributions, and error report summary
- `/explore [query]`:
  - list indexed files; supports substrings, extensions, or glob patterns
  - `--full-path` shows full prefixes instead of compact tree
- `/show <path>`:
  - show chunks for a specific file path
- `/search <query>`:
  - search metadata by path substring
- `/build [--force]`:
  - run ingest pipeline (incremental by default)
  - `--force` reprocesses all files after confirmation
- `/tag [--max N]`:
  - run AI tagging on untagged chunks
- `/tags`:
  - list all unique AI and user tags
- `/wipe`:
  - delete all chunks after confirmation
- `/help`:
  - show ingest help
- `/exit`:
  - exit the ingest REPL

Side effects:

- `/build` writes to the Chroma collection and updates the manifest.
- `/wipe` deletes all data in the collection.
- `/tag` writes tags into chunk metadata.

## Query REPL Commands

Context: `lsm/query/repl.py`

- `/help`:
  - show help
- `/exit`:
  - exit the query REPL
- `/show S#`:
  - show the cited chunk text
- `/expand S#`:
  - show the full chunk text without truncation
- `/open S#`:
  - open the source file using the OS default application
- `/models`:
  - list models available to the OpenAI API key
- `/model`:
  - show current model
- `/model <name>`:
  - set model for this session
- `/providers`:
  - list available LLM providers
- `/mode`:
  - show current mode
- `/mode <name>`:
  - switch mode for this session
- `/note`:
  - open the last query in an editor and save as a note
- `/load <path>`:
  - pin all chunks from a file for the next query
- `/load clear`:
  - clear all pinned chunks
- `/debug`:
  - show last query diagnostics
- `/set path_contains <substring> [more...]`:
  - set path filters for this session
- `/set ext_allow .md .pdf`:
  - set allowlist extensions
- `/set ext_deny .txt`:
  - set denylist extensions
- `/clear path_contains|ext_allow|ext_deny`:
  - clear a filter

Side effects:

- `/note` writes a Markdown file to the notes directory.
- `/model` overrides the model for the current session only.
- `/mode` overrides the mode for the current session only.
- `/load` pins chunk IDs for the next query.

## Exit Codes and Errors

- Most commands print user-facing error messages on failure.
- API errors during query fallback to local excerpts when possible.
