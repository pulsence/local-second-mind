# Notes System Guide

LSM can save query sessions as Markdown notes. Notes include the query, answer,
and source excerpts to make results easy to revisit later.

## How Notes Work

- Notes are generated from the last query in the query REPL.
- `/note` opens the note in your default editor so you can edit before saving.
- Notes are stored as Markdown files in a configured directory.

## Notes Configuration

Notes are configured per mode via a `modes` entry with matching `name`:

```json
"notes": {
  "enabled": true,
  "dir": "notes",
  "template": "default",
  "filename_format": "timestamp"
}
```

### Fields

- `enabled`: enable or disable note saving for this mode
- `dir`: directory to store notes (relative to config file)
- `template`: template name (currently only `default` is implemented)
- `filename_format`: `timestamp` or `query_slug`

The `incremental` format is not implemented yet; unknown values fall back to
`timestamp`.

## Filename Formats

- `timestamp`: `YYYYMMDD-HHMMSS.md`
- `query_slug`: `query-slug-YYYYMMDD.md`

## Editor Integration

When you run `/note`, LSM uses the following editor logic:

1. If `EDITOR` is set, it is used.
2. If `VISUAL` is set, it is used.
3. Otherwise, a platform default is used:
   - Windows: `notepad`
   - macOS: `open -e`
   - Linux: `nano`

You can set your editor in the environment:

```bash
# Windows PowerShell
$env:EDITOR="code --wait"

# macOS/Linux
export EDITOR="code --wait"
```

## Note Contents

Notes include these sections:

- Query metadata (date, mode)
- Query text
- Local sources with relevance and excerpts
- Remote sources (if any)
- Final answer

This format is generated in `lsm/query/notes.py`.

## Using Notes in the Query REPL

1. Run a query in the REPL.
2. Type `/note` to open the note for editing.
3. Save and close the editor to write the note.

If there was no previous query, `/note` will warn and do nothing.

## Organizing Notes

Recommended strategies:

- Use per-mode directories (e.g., `research_notes`, `analysis_notes`).
- Use `query_slug` when you want human-readable filenames.
- Keep notes next to your project or knowledge base for easy search.

## Troubleshooting

- If `/note` fails to open, verify your `EDITOR` is installed and in PATH.
- If notes are not saved, check that `dir` is writable.
- If notes appear empty, verify the query completed and returned sources.
