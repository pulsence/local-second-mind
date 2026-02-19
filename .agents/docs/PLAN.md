# Local Second Mind v0.7.0 Prepartion Plan: Transition Work

This plan restructures project documentation to separate end-user docs from developer/agent docs.

## Phase 1: Directory Structure Setup (COMPLETED)

### 1.1: Rename and Create Base Structure
- Rename `.claude_files/` to `.agents/`
- Create `.agents/future_plans/` folder
- Move `.claude_files/INGEST_FUTURE.md` to `.agents/future_plans/`
- Delete `.claude_files/sandbox_plan/` folder (content consolidated in `.agents/docs/architecture/development/SECURITY.md`)

### 1.2: Create .agents/docs/ Folder Structure
```
.agents/
├── docs/
│   ├── INDEX.md           # High-level index for agents
│   ├── ARCHITECTURE.md    # Combined architecture + key files
│   └── architecture/      # Package-level documentation
│       ├── lsm.md
│       ├── lsm.agents.md
│       ├── lsm.config.md
│       ├── lsm.ingest.md
│       ├── lsm.providers.md
│       ├── lsm.query.md
│       ├── lsm.remote.md
│       ├── lsm.ui.md
│       └── lsm.vectordb.md
└── future_plans/
    └── INGEST_FUTURE.md   # Future embedding/retrieval improvements
```

**Success criteria:** Directory structure matches the intended layout.

---

## Phase 2: Create INDEX.md (COMPLETED)

### 2.1: Create INDEX.md from CLAUDE.md
- Use CLAUDE.md as the base
- Remove `## Architecture` and `## Key Files` sections (moved to ARCHITECTURE.md)
- Add links to ARCHITECTURE.md and package docs in `architecture/`
- Keep compact and focused on what agents need to quickly understand

**Success criteria:** INDEX.md is a concise entry point with links to detailed docs.

---

## Phase 3: Create ARCHITECTURE.md (COMPLETED)

### 3.1: Extract and Combine
- Extract `## Architecture` section from CLAUDE.md
- Extract `## Key Files` section from CLAUDE.md
- Combine into single ARCHITECTURE.md
- Add links to package documentation files

**Success criteria:** ARCHITECTURE.md serves as index for primary packages and design patterns.

---

## Phase 4: Create Package Documentation (COMPLETED)B

### 4.1: Create Package Overview Files
Create one markdown file per top-level package with format:

Files to create:
- `lsm.agents.md` - agents, memory, tools subpackages
- `lsm.config.md` - loader, models subpackage
- `lsm.ingest.md` - pipeline, chunking, parsing modules
- `lsm.providers.md` - LLM providers
- `lsm.query.md` - retrieval, synthesis, integrations
- `lsm.remote.md` - remote providers
- `lsm.ui.md` - shell, tui, web, desktop subpackages
- `lsm.vectordb.md` - vector DB providers and migrations

Each file follows this template:
```markdown
## lsm.<package>
Description: <1-2 sentences>
Folder Path: <path>

### Sub Packages
- [lsm.<sub>](link): <description>

### Modules
- [file.py](link): <description>
```

**Success criteria:** All packages documented with subpackages and modules.

---

## Phase 5: Restructure docs/ for End Users (COMPLETED)

### 5.1: Identify Developer Docs to Move
Move from `docs/` to `.agents/docs/`:
- `docs/architecture/` (entire folder - developer-focused)
- `docs/api-reference/` (entire folder - developer-focused)
- `docs/development/` (entire folder - developer-focused)
- `docs/AGENTS.md` (move to `.agents/docs/`)

### 5.2: Keep in docs/ (End User Docs)
- `docs/README.md` - update to reflect new structure
- `docs/user-guide/` - keep for end users
- `docs/NOTES.md`, `docs/QUERY_MODES.md`, `docs/REMOTE_SOURCES.md` - keep if user-facing

### 5.3: Update docs/README.md
- Remove links to architecture/api-reference/development
- Keep only user-guide links
- Update "What's New" section

**Success criteria:** `docs/` contains only end-user documentation.

---

## Phase 6: Update References

### 6.1: Update CLAUDE.md
- Add note pointing to `.agents/docs/INDEX.md` for full documentation
- Keep brief overview for quick reference

### 6.2: Update AGENTS.md
- Update to point to `.agents/docs/INDEX.md`

### 6.3: Verify no broken links
- Run grep for references to old paths
- Update any cross-references

**Success criteria:** All references point to correct locations.

---

## Notes

### Security Documentation
The `.agents/docs/architecture/development/SECURITY.md` file has been verified to properly consolidate all security notes from the deleted `.claude_files/sandbox_plan/` folder:
- Threat model (T1-T8 categories)
- STRIDE coverage matrix with test file references
- Attack surface inventory
- Permission gate reference
- Testing methodology

The sandbox_plan files were historical planning documents that have been fully implemented and their content consolidated.

---

## Task Summary

| Phase | Tasks | Est. Files |
|-------|-------|------------|
| 1 | Directory setup | ~4 ops |
| 2 | INDEX.md | 1 file |
| 3 | ARCHITECTURE.md | 1 file |
| 4 | Package docs | ~9 files |
| 5 | Restructure docs/ | ~15 files moved |
| 6 | Update references | ~3 files |
