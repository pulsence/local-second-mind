# Phase 9: Communication Platform

**Why ninth:** Email/calendar/news assistants need their data providers built alongside them. OAuth infrastructure is a prerequisite for email/calendar providers. Placed after remote sources (Phase 8) to leverage established provider patterns.

**Depends on:** Phase 6.4 (Assistant agent pattern), Phase 8 (provider patterns established)

| Task | Description | Depends On |
|------|-------------|------------|
| 9.1 | OAuth2 shared infrastructure | None |
| 9.2 | Email providers | 9.1 |
| 9.3 | Calendar providers | 9.1 |
| 9.4 | Email Assistant agent | 9.2, 6.1 |
| 9.5 | Calendar Assistant agent | 9.3, 6.1 |
| 9.6 | News Assistant agent | 8.5, 6.1 |

## 9.1: OAuth2 Shared Infrastructure
- **Description:** Build a shared OAuth2 client module for providers that require user authorization (Gmail, Google Calendar, Microsoft Graph).
- **Tasks:**
  - Implement OAuth2 authorization code flow with redirect handling (local HTTP callback server).
  - Token storage: encrypted tokens in `<global_folder>/oauth_tokens/` with per-provider isolation.
  - Automatic token refresh with configurable refresh buffer.
  - Consent/scope management: request minimal scopes, store granted scopes.
  - Add configuration: `remote_providers.<name>.oauth: { client_id, client_secret, scopes, redirect_uri }`.
  - Write tests for OAuth2 authorization code flow, token storage/encryption, automatic token refresh, scope management, and config field validation (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_providers/`) and verify all new and existing tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/remote/oauth.py`
  - `lsm/config/models/modes.py`
- **Success criteria:** OAuth flow completes for Google and Microsoft. Tokens persist and auto-refresh. Credentials are never logged.

## 9.2: Email Providers
- **Description:** Add major email providers for assistant agents.
- **Providers:**
  - `gmail.py` — Gmail via Google API (OAuth2). Read, search, draft.
  - `microsoft_graph_mail.py` — Outlook via Microsoft Graph (OAuth2). Read, search, draft.
  - `imap.py` — IMAP/SMTP fallback for self-hosted email.
- **Tasks:**
  - Implement each provider with read, search, and draft capabilities.
  - All write operations (send, move, delete) must go through approval gating via `ask_user`.
  - Write tests for email read/search operations, draft creation, `ask_user` approval gating for write operations, and OAuth integration for Gmail and Microsoft Graph (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_providers/`) and verify all new and existing tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/remote/providers/gmail.py`
  - `lsm/remote/providers/microsoft_graph_mail.py`
  - `lsm/remote/providers/imap.py`
- **Success criteria:** Email providers can read and search mail. Draft creation requires explicit user approval before sending.

## 9.3: Calendar Providers
- **Description:** Add major calendar providers for assistant agents.
- **Providers:**
  - `google_calendar.py` — Google Calendar API (OAuth2).
  - `microsoft_graph_calendar.py` — Microsoft Graph Calendar (OAuth2).
  - `caldav.py` — CalDAV fallback for self-hosted calendars.
- **Tasks:**
  - Implement each provider with read and mutating event capabilities.
  - All mutating operations (create, update, delete events) require explicit user approval.
  - Write tests for calendar event reading, mutating operation approval gating, and OAuth integration for Google and Microsoft Graph (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_providers/`) and verify all new and existing tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/remote/providers/google_calendar.py`
  - `lsm/remote/providers/microsoft_graph_calendar.py`
  - `lsm/remote/providers/caldav.py`
- **Success criteria:** Calendar providers can read events and propose changes that require approval.

## 9.4: Email Assistant Agent
- **Description:** Agent that reads, summarizes, and drafts emails with user approval gating.
- **Tasks:**
  - Read emails by time window: last 1 hour, last 24 hours, custom range.
  - Filter by criteria: search string, to/from specific persons, unread only, specific folder.
  - Produce email summary organized by importance/topic.
  - Generate task list from action-requiring emails.
  - Draft reply/compose with explicit user approval before any send.
  - Write tests for time-window email retrieval, filter criteria, summary generation, task list extraction, and approval-gated draft/send workflow (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/agents/assistants/email_assistant.py`
- **Success criteria:** Agent summarizes inbox, generates task lists, and drafts emails that require user approval to send.

## 9.5: Calendar Assistant Agent
- **Description:** Agent that reads calendar, provides scheduling intelligence, and manages events with approval.
- **Tasks:**
  - Read calendar and summarize upcoming events by day/week.
  - Given a proposed event, suggest available time slots based on existing calendar.
  - Add/remove/edit events with explicit user approval for every mutation.
  - Write tests for calendar summarization, available time slot suggestion, and approval-gated event mutation workflows (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/agents/assistants/calendar_assistant.py`
- **Success criteria:** Agent provides scheduling suggestions and requires approval for all calendar changes.

## 9.6: News Assistant Agent
- **Description:** Agent that produces curated news summaries from configured news sources.
- **Tasks:**
  - Produce news summary in newsletter style over a configurable time frame.
  - Filter by specific topics or criteria.
  - Source from news providers (8.5) and RSS feeds.
  - Write tests for newsletter-style summary generation, topic filtering, time frame configuration, and multi-source aggregation (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
  - Commit the implementation following the format in `.agents/docs/COMMIT_MESSAGE.md`.
- **Files:**
  - `lsm/agents/assistants/news_assistant.py`
- **Success criteria:** Agent produces newsletter-style summaries from configured news sources with topic filtering.
