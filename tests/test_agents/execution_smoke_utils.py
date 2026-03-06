from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lsm.agents.memory import SQLiteMemoryStore
from lsm.config.loader import build_config_from_raw
from lsm.config.models.agents import MemoryConfig
from lsm.query.pipeline import RetrievalPipeline
from lsm.remote.base import RemoteResult
from lsm.remote.providers.communication.models import CalendarEvent, EmailDraft, EmailMessage
from lsm.vectordb.base import VectorDBGetResult, VectorDBQueryResult


def build_smoke_config(tmp_path: Path):
    raw = {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "path": str(tmp_path / "data"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "tiers": {
                "normal": {"provider": "openai", "model": "gpt-5.2"},
                "complex": {"provider": "openai", "model": "gpt-5.2"},
            },
            "services": {
                "default": {"provider": "openai", "model": "gpt-5.2"},
                "query": {"provider": "openai", "model": "gpt-5.2"},
            },
        },
        "db": {
            "path": str(tmp_path / "data"),
            "vector": {"provider": "sqlite", "collection": "local_kb"},
        },
        "query": {"mode": "grounded"},
        "remote_providers": [
            {"name": "arxiv", "type": "arxiv", "max_results": 3},
        ],
        "remote_provider_chains": [
            {
                "name": "Research Digest",
                "links": [{"source": "arxiv"}],
            }
        ],
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 4000,
            "max_iterations": 4,
            "context_window_strategy": "compact",
            "interaction": {"auto_continue": True},
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": True,
                "execution_mode": "local_only",
                "require_user_permission": {
                    "load_url": False,
                    "spawn_agent": False,
                    "await_agent": False,
                    "collect_artifacts": False,
                    "bash": False,
                    "powershell": False,
                },
                "require_permission_by_risk": {
                    "writes_workspace": False,
                    "network": False,
                    "exec": False,
                },
                "tool_llm_assignments": {},
            },
            "memory": {
                "enabled": True,
                "storage_backend": "sqlite",
            },
            "agent_configs": {},
        },
    }
    return build_config_from_raw(raw, tmp_path / "config.json")


def make_memory_store(tmp_path: Path) -> SQLiteMemoryStore:
    cfg = MemoryConfig(storage_backend="sqlite")
    conn = sqlite3.connect(str(tmp_path / "agent-memory.sqlite3"), check_same_thread=False)
    return SQLiteMemoryStore(conn, cfg, owns_connection=True)


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        _ = kwargs
        if isinstance(texts, str):
            texts = [texts]
        return [[0.6, 0.4] for _ in texts]


class FakeVectorCollection:
    def __init__(self) -> None:
        self.records = [
            {
                "id": "c1",
                "document": "Sin is a voluntary act contrary to right reason.",
                "metadata": {
                    "source_path": "notes/aquinas.md",
                    "source_name": "aquinas.md",
                    "title": "On Sin",
                    "author": "Thomas Aquinas",
                    "year": "1273",
                    "ext": ".md",
                    "is_current": True,
                },
                "embedding": [1.0, 0.0],
                "distance": 0.05,
            },
            {
                "id": "c2",
                "document": "Hamartia in Greek often refers to missing the mark.",
                "metadata": {
                    "source_path": "notes/language.md",
                    "source_name": "language.md",
                    "title": "Etymology",
                    "author": "Lexicon",
                    "year": "1999",
                    "ext": ".md",
                    "is_current": True,
                },
                "embedding": [0.95, 0.05],
                "distance": 0.08,
            },
            {
                "id": "c3",
                "document": "A placeholder note about unrelated topics.",
                "metadata": {
                    "source_path": "notes/misc.md",
                    "source_name": "misc.md",
                    "title": "Misc",
                    "author": "Unknown",
                    "year": "2001",
                    "ext": ".md",
                    "is_current": True,
                },
                "embedding": [-1.0, 0.0],
                "distance": 0.6,
            },
        ]

    def query(self, embedding, top_k, filters=None):
        _ = embedding
        matched = self._filter_records(filters)
        matched = sorted(matched, key=lambda item: item["distance"])[:top_k]
        return VectorDBQueryResult(
            ids=[item["id"] for item in matched],
            documents=[item["document"] for item in matched],
            metadatas=[item["metadata"] for item in matched],
            distances=[item["distance"] for item in matched],
        )

    def get(self, ids=None, filters=None, limit=None, offset=0, include=None):
        include = include or ["metadatas"]
        if ids is not None:
            selected = [item for item in self.records if item["id"] in ids]
        else:
            selected = self._filter_records(filters)
        if offset:
            selected = selected[offset:]
        if limit is not None:
            selected = selected[:limit]
        return VectorDBGetResult(
            ids=[item["id"] for item in selected],
            documents=[item["document"] for item in selected] if "documents" in include else None,
            metadatas=[item["metadata"] for item in selected] if "metadatas" in include else None,
            embeddings=[item["embedding"] for item in selected] if "embeddings" in include else None,
        )

    def _filter_records(self, filters=None):
        if not isinstance(filters, dict):
            return list(self.records)

        def _matches(item):
            metadata = item["metadata"]
            for key, value in filters.items():
                if key == "is_current":
                    if bool(metadata.get("is_current")) != bool(value):
                        return False
                    continue
                if isinstance(value, dict) and "$eq" in value:
                    expected = value["$eq"]
                else:
                    expected = value
                if metadata.get(key) != expected:
                    return False
            return True

        return [item for item in self.records if _matches(item)]


class FakePipelineLLMProvider:
    name = "openai"
    model = "gpt-5.2"

    def __init__(self) -> None:
        self.last_response_id = "pipeline-resp-1"

    def send_message(self, **kwargs):
        _ = kwargs
        return "Grounded synthesis from the pipeline. [S1]"

    def estimate_cost(self, input_tokens, output_tokens):
        _ = input_tokens, output_tokens
        return 0.0


class FakeQueryLLMProvider:
    name = "openai"
    model = "gpt-5.2"

    def __init__(self) -> None:
        self.last_response_id = "tool-resp-1"

    def send_message(self, **kwargs):
        prompt = str(kwargs.get("input", ""))
        return f"LLM tool response for: {prompt}"


class FakeRemoteProvider:
    def search_structured(self, input_dict, max_results=5):
        query = str(input_dict.get("query", ""))
        return [
            {
                "title": f"Remote result for {query}",
                "url": "https://example.com/remote",
                "description": "Structured remote result",
                "score": 0.9,
                "metadata": {"source_id": "remote-1"},
            }
        ][:max_results]

    def search(self, query, max_results=5):
        return [
            RemoteResult(
                title=f"Remote result for {query}",
                url="https://example.com/news",
                snippet="Remote snippet",
                score=0.8,
                metadata={"source_id": "remote-news-1"},
            )
        ][:max_results]


class FakeRemoteChain:
    def __init__(self, config, chain_config):
        self.config = config
        self.chain_config = chain_config

    def execute(self, input_dict, max_results=5):
        query = str(input_dict.get("query", ""))
        return [
            {
                "title": f"Chain result for {query}",
                "url": "https://example.com/chain",
                "description": "Chained remote result",
            }
        ][:max_results]


class FakeCalendarProvider:
    name = "fake-calendar"

    def list_events(self, query=None, time_min=None, time_max=None, max_results=50):
        _ = query, time_min, time_max
        now = datetime.utcnow()
        return [
            CalendarEvent(
                event_id="evt-1",
                title="Research meeting",
                start=now + timedelta(hours=1),
                end=now + timedelta(hours=2),
                attendees=["team@example.com"],
            )
        ][:max_results]


class FakeEmailProvider:
    name = "fake-email"

    def list_messages(
        self,
        query=None,
        from_address=None,
        to_address=None,
        unread_only=False,
        folder=None,
        after=None,
        before=None,
        max_results=50,
    ):
        _ = query, from_address, to_address, unread_only, folder, after, before
        return [
            EmailMessage(
                message_id="msg-1",
                subject="Follow up on Aquinas notes",
                sender="mentor@example.com",
                recipients=["user@example.com"],
                snippet="Please review the latest outline.",
                received_at=datetime.utcnow(),
                is_unread=True,
            )
        ][:max_results]

    def create_draft(self, recipients, subject, body, thread_id=None):
        return EmailDraft(
            draft_id="draft-1",
            recipients=list(recipients),
            subject=subject,
            body=body,
            thread_id=thread_id,
        )

    def send_draft(self, draft_id):
        _ = draft_id


class FakeNewsProvider:
    name = "fake-news"

    def search(self, query, max_results=5):
        return [
            RemoteResult(
                title=f"News about {query or 'general'}",
                url="https://example.com/story",
                snippet="Story summary",
                score=0.7,
                metadata={"source_id": "story-1", "provider": self.name},
            )
        ][:max_results]


@dataclass
class FakeHTTPResponse:
    text: str
    status_code: int = 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class ScriptedHarnessProvider:
    name = "openai"
    model = "gpt-5.2"
    supports_function_calling = False

    def __init__(self) -> None:
        self.last_response_id = "harness-resp"

    def send_message(self, **kwargs):
        prompt = str(kwargs.get("input", ""))
        return json.dumps(self._payload_for_prompt(prompt))

    def _payload_for_prompt(self, prompt: str) -> dict[str, Any]:
        if "Phase: DECOMPOSE" in prompt:
            return self._done(json.dumps(["Origins of sin", "Aquinas on sin"]))
        if "Phase: RESEARCH SUMMARY" in prompt:
            return self._done("- Retrieved grounded findings\n- Preserved source-backed evidence")
        if "Research findings by subtopic" in prompt:
            return self._done("# Research Outline: Topic\n\n## Origins\n\n- Summary\n")
        if 'Return JSON: {"sufficient": true/false' in prompt:
            return self._done(json.dumps({"sufficient": True, "suggestions": []}))
        if "Phase: OUTLINE" in prompt:
            return self._done("## Outline\n\n- Point one\n- Point two")
        if "Phase: DRAFT" in prompt:
            return self._done("# Deliverable\n\nDrafted content.")
        if "Revise the draft for clarity and factual grounding" in prompt:
            return self._done("# Deliverable\n\nReviewed content.")
        if "Phase: PLAN" in prompt:
            return self._done("Plan recorded.")
        if "Phase: EVIDENCE" in prompt:
            return self._done("Evidence gathered.")
        if "Based on the evidence gathered above" in prompt:
            return self._done("# Synthesis\n\nGrounded synthesis.")
        if "Select corpus curation scope" in prompt:
            return self._done(
                json.dumps(
                    {
                        "scope_path": ".",
                        "stale_days": 365,
                        "near_duplicate_threshold": 0.9,
                        "top_near_duplicates": 25,
                    }
                )
            )
        if "Produce concise actionable corpus curation recommendations" in prompt:
            return self._done(json.dumps(["Review duplicate notes.", "Clean placeholder files."]))
        return self._done("Completed.")

    @staticmethod
    def _done(response: str) -> dict[str, Any]:
        return {"response": response, "action": "DONE", "action_arguments": {}}

    def estimate_cost(self, input_tokens, output_tokens):
        _ = input_tokens, output_tokens
        return 0.0


def build_pipeline(config) -> RetrievalPipeline:
    return RetrievalPipeline(
        db=FakeVectorCollection(),
        embedder=FakeEmbedder(),
        config=config,
        llm_provider=FakePipelineLLMProvider(),
    )
