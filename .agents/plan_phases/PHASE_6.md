# Phase 6: Core Agent Catalog

**Why sixth:** Now that framework is restructured (Phase 3), tooling is enhanced (Phase 4), and runners are ready (Phase 5), new agents can be built on solid foundations.

**Depends on:** Phase 3.1 (restructure), Phase 4 (tooling)

| Task | Description | Depends On |
|------|-------------|------------|
| 6.1 | Agent catalog and registration | 3.1 |
| 6.2 | General Agent | 3.3, 4.1 |
| 6.3 | Librarian Agent | 2.4, 4.3 |
| 6.4 | Assistant Agent | None beyond base framework |
| 6.5 | Manuscript Editor Agent | 2.3, 4.2, 4.4 |

## 6.1: Agent Catalog and Registration
- **Description:** Extend the existing agent registry to include new agent types and themes.
- **Tasks:**
  - Add new agent names to `lsm/agents/factory.py` and module exports.
  - Update UI/shell agent lists to include new types.
  - Ensure agent metadata (description, default tools, risk posture) is exposed.
- **Files:**
  - `lsm/agents/factory.py`
  - `lsm/ui/shell/commands/agents.py`
  - `lsm/ui/tui/screens/agents.py`
- **Success criteria:** New agents are discoverable in UI and shell commands.

## 6.2: General Agent
- **Description:** Provide a general-purpose task agent that uses available tools responsibly.
- **Tasks:**
  - Define prompt and tool allowlist defaults.
  - Implement run outputs (summary + artifact list).
  - Add guardrails for permission and iteration limits.
- **Files:**
  - `lsm/agents/productivity/general.py`
  - `lsm/agents/harness.py`
- **Success criteria:** General agent can execute tool-based tasks and emits standard artifacts.

## 6.3: Librarian Agent
- **Description:** Explore embeddings DB and create idea graphs/metadata summaries.
- **Tasks:**
  - Define embeddings query workflow and output format.
  - Integrate graph output with `file_graph` and memory tools.
  - Emit `idea_graph.md` and supporting artifacts.
- **Files:**
  - `lsm/agents/productivity/librarian.py`
  - `lsm/agents/tools/query_embeddings.py`
- **Success criteria:** Librarian agent produces a structured idea graph and citations.

## 6.4: Assistant Agent
- **Description:** Summarize cross-agent activity and suggest memory updates. This is the "HR for agents" assistant — NOT the communication assistants (those are in Phase 9).
- **Tasks:**
  - Review `run_summary.json` artifacts from all agent runs.
  - Produce consolidated summaries of actions taken, results produced, and security concerns.
  - Identify cross-agent patterns that should be stored as memories or trigger memory updates/removals.
  - Emit action recommendations for user review.
  - Integrate with memory tools for promotion/rejection workflow.
- **Files:**
  - `lsm/agents/assistants/assistant.py`
  - `lsm/agents/memory/**`
- **Success criteria:** Assistant agent produces consolidated summaries and actionable memory candidates.

## 6.5: Manuscript Editor Agent
- **Description:** Specialized agent for iteratively editing text documents.
- **Tasks:**
  - Define manuscript editing workflow: read document outline → identify sections for revision → iterative editing rounds → emit revision log + final artifact.
  - Use text graphing for section-level edits.
  - Emit revision logs and a final manuscript artifact.
- **Files:**
  - `lsm/agents/productivity/manuscript_editor.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** Manuscript editor produces revised documents with traceable edits.
