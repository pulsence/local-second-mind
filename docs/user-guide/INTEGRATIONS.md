## Note-taking Integrations

Local Second Mind can format query notes for Obsidian and Logseq. Enable
integration options in the global top-level `notes` config.

### Obsidian

Recommended settings:
- `integration`: `"obsidian"`
- `wikilinks`: `true`
- `backlinks`: `true`
- `include_tags`: `true`

Example:
```json
{
  "notes": {
    "enabled": true,
    "dir": "notes",
    "integration": "obsidian",
    "wikilinks": true,
    "backlinks": true,
    "include_tags": true
  }
}
```

### Logseq

Recommended settings:
- `integration`: `"logseq"`
- `wikilinks`: `true`
- `backlinks`: `true`
- `include_tags`: `true`

Example:
```json
{
  "notes": {
    "enabled": true,
    "dir": "notes",
    "integration": "logseq",
    "wikilinks": true,
    "backlinks": true,
    "include_tags": true
  }
}
```

### Citation Export

Export citations from the last query or from a saved note:
- `/export-citations` (defaults to BibTeX, last query)
- `/export-citations zotero`
- `/export-citations bibtex path/to/note.md`

### Cost Tracking

Session-only tracking with CSV export:
- `/costs`
- `/costs export costs.csv`
- `/budget set 5.00`
