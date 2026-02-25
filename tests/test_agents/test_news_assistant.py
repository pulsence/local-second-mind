from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from lsm.agents.assistants.news_assistant import NewsAssistantAgent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw
from lsm.remote.base import RemoteResult


@dataclass
class StubNewsProvider:
    name: str
    results: List[RemoteResult]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        _ = query
        return self.results[:max_results]


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 8000,
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_news_assistant_filters_by_topic_and_time_window(tmp_path: Path) -> None:
    now = datetime(2024, 3, 1, 12, 0, 0)
    provider = StubNewsProvider(
        name="news-a",
        results=[
            RemoteResult(
                title="AI breakthrough",
                url="http://example.com/ai",
                snippet="AI is advancing",
                metadata={"published_date": (now - timedelta(hours=1)).isoformat()},
            ),
            RemoteResult(
                title="Sports update",
                url="http://example.com/sports",
                snippet="Game highlights",
                metadata={"published_date": (now - timedelta(hours=1)).isoformat()},
            ),
        ],
    )
    provider_b = StubNewsProvider(
        name="news-b",
        results=[
            RemoteResult(
                title="AI policy",
                url="http://example.com/policy",
                snippet="Policy news",
                metadata={"published_date": (now - timedelta(days=2)).isoformat()},
            )
        ],
    )
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = NewsAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        agent_overrides={
            "provider_instances": [provider, provider_b],
            "now": now.isoformat(),
        },
    )

    payload = {"topics": ["ai"], "window_hours": 24}
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    summary = json.loads(agent.last_result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["total_stories"] == 1
    assert summary["stories"][0]["title"] == "AI breakthrough"


def test_news_assistant_aggregates_sources(tmp_path: Path) -> None:
    now = datetime(2024, 3, 2, 12, 0, 0)
    provider_a = StubNewsProvider(
        name="news-a",
        results=[
            RemoteResult(
                title="Market update",
                url="http://example.com/market",
                snippet="Stocks are up",
                metadata={"published_date": now.isoformat()},
            )
        ],
    )
    provider_b = StubNewsProvider(
        name="news-b",
        results=[
            RemoteResult(
                title="Weather alert",
                url="http://example.com/weather",
                snippet="Storm incoming",
                metadata={"published_date": now.isoformat()},
            )
        ],
    )
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = NewsAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        agent_overrides={
            "provider_instances": [provider_a, provider_b],
            "now": now.isoformat(),
        },
    )

    payload = {"topics": ["general"]}
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    summary = json.loads(agent.last_result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["total_stories"] == 2

    markdown = agent.last_result.summary_path.read_text(encoding="utf-8")
    assert "News Briefing" in markdown
    assert "Top Stories" in markdown
