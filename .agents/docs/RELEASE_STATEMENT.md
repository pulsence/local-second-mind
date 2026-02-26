# Release Statement Template

Use this template when drafting GitHub release notes for Local Second Mind.
Copy the block below, replace all `{{ PLACEHOLDER }}` tokens, then delete this header section.

---

## Local Second Mind v{{ VERSION }} ({{ DATE YYYY-MM-DD }})

{{ One-paragraph executive summary. Describe the major themes of this release in 2–4 sentences.
   Name the high-level feature areas (e.g. "agent ecosystem", "provider improvements") without
   listing every item — the sections below cover details. }}

### What's New

{{ One bullet per major user-visible feature area. Start each bullet with "Added".
   Group related items into a single bullet rather than fragmenting. }}

- Added {{ feature area 1 }}: {{ brief capability description }}.
- Added {{ feature area 2 }}: {{ brief capability description }}.
- Added {{ feature area 3 }}: {{ brief capability description }}.

### Upgrade Notes

{{ Omit this section entirely if there are no breaking changes or new required config fields. }}

- New/updated config fields:
  - `{{ section.field }}` — {{ what it controls }}
  - `{{ section.field }}` — {{ what it controls }}
- {{ Breaking change description }}: {{ old behavior }} changed to {{ new behavior }}.
  - {{ Migration step if needed. }}
- {{ Validation/schema change }}: {{ what was previously tolerated that now errors. }}

### Summary Changelog

- **Added**
  - {{ Fine-grained item. Use same phrasing style as "What's New" but more specific. }}
  - {{ Fine-grained item. }}
- **Changed**
  - {{ Behavior or interface change. }}
  - {{ Behavior or interface change. }}
- **Fixed**
  - {{ Bug or hardening fix. }}
  - {{ Bug or hardening fix. }}

**Detailed internal changelog:** `docs/CHANGELOG.md`

---

## Authoring Guidelines

### Header

```
## Local Second Mind vX.Y.Z (YYYY-MM-DD)
```

Use the full semantic version and the release date in ISO format.

### Executive Summary

- 2–4 sentences maximum.
- Name major feature *areas* (e.g. "file graphing infrastructure", "OAuth2 communication platform")
  rather than individual items.
- Mention the rough count of significant additions only if it adds clarity (e.g. "20+ new remote
  source providers").

### What's New

- One bullet per major feature area. Consolidate tightly related items.
- Each bullet starts with **"Added"** (present perfect, not gerund).
- Order: most impactful / most user-visible items first.
- Omit internal-only refactors here; those belong in Summary Changelog → Changed.

### Upgrade Notes

Include this section **only** when one or more of the following apply:

| Trigger | Example |
|---------|---------|
| New required or impactful config fields | `llms.tiers`, `global.mcp_servers` |
| Breaking import or API path changes | agent subpackage restructuring |
| Changed tool output shape | `source_map` returning outlines instead of snippets |
| Validation now rejects previously-tolerated configs | strict remote provider schemas |

List each item as a sub-bullet with a code-formatted field name or symbol and a short description.

### Summary Changelog

Three subsections in order: **Added**, **Changed**, **Fixed**.

- Items are finer-grained than "What's New" bullets — one item per concrete component or behavior.
- Omit subsections with no entries.
- Use the same verb-first, present-tense style as the example (`Added`, `Changed`, `Fixed` prefix
  is the subsection heading, not repeated on each line).

### Footer

Always end with:

```
**Detailed internal changelog:** `docs/CHANGELOG.md`
```
