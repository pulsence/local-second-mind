from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lsm.agents.assistants.email_assistant import EmailAssistantAgent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw
from lsm.remote.providers.communication.models import EmailDraft, EmailMessage


class AskUserStub(BaseTool):
    name = "ask_user"
    description = "Stub approval tool."
    input_schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}, "context": {"type": "string"}},
        "required": ["prompt"],
    }

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses = responses or ["yes"]
        self.calls: List[Dict[str, Any]] = []
        self._idx = 0

    def execute(self, args: dict) -> str:
        self.calls.append(dict(args))
        if self._idx >= len(self.responses):
            return self.responses[-1]
        response = self.responses[self._idx]
        self._idx += 1
        return response


@dataclass
class StubEmailProvider:
    messages: List[EmailMessage]

    def __post_init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.sent_drafts: List[str] = []
        self.created_drafts: List[EmailDraft] = []

    @property
    def name(self) -> str:
        return "stub_email"

    def search_messages(
        self,
        query: str,
        *,
        max_results: int = 10,
        unread_only: bool = False,
        from_address: Optional[str] = None,
        to_address: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        folder: Optional[str] = None,
    ) -> List[EmailMessage]:
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "unread_only": unread_only,
                "from_address": from_address,
                "to_address": to_address,
                "after": after,
                "before": before,
                "folder": folder,
            }
        )
        return self.messages[:max_results]

    def create_draft(
        self,
        *,
        recipients: List[str],
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> EmailDraft:
        draft = EmailDraft(
            draft_id=f"draft-{len(self.created_drafts)+1}",
            subject=subject,
            recipients=recipients,
            body=body,
            thread_id=thread_id,
        )
        self.created_drafts.append(draft)
        return draft

    def send_draft(self, draft_id: str) -> None:
        self.sent_drafts.append(draft_id)


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "path": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "path": str(tmp_path / ".chroma"),
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


def test_email_assistant_applies_time_window_and_filters(tmp_path: Path) -> None:
    provider = StubEmailProvider(messages=[])
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    registry.register(AskUserStub())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = EmailAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
        agent_overrides={
            "provider_instance": provider,
            "now": "2024-01-01T12:00:00",
        },
    )

    payload = {
        "action": "summary",
        "time_window": "1h",
        "filters": {
            "query": "status",
            "from": "alice@example.com",
            "to": "me@example.com",
            "unread_only": True,
            "folder": "inbox",
        },
    }
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))

    assert provider.calls
    call = provider.calls[0]
    assert call["query"] == "status"
    assert call["from_address"] == "alice@example.com"
    assert call["to_address"] == "me@example.com"
    assert call["unread_only"] is True
    assert call["folder"] == "inbox"
    assert call["after"] == datetime(2024, 1, 1, 11, 0, 0)
    assert call["before"] == datetime(2024, 1, 1, 12, 0, 0)


def test_email_assistant_generates_summary_and_tasks(tmp_path: Path) -> None:
    now = datetime(2024, 1, 2, 12, 0, 0)
    messages = [
        EmailMessage(
            message_id="m1",
            subject="Urgent: Please review",
            sender="boss@example.com",
            recipients=["me@example.com"],
            snippet="Please review the document.",
            received_at=now - timedelta(hours=1),
            is_unread=True,
        ),
        EmailMessage(
            message_id="m2",
            subject="FYI update",
            sender="team@example.com",
            recipients=["me@example.com"],
            snippet="Just for your awareness.",
            received_at=now - timedelta(hours=5),
            is_unread=False,
        ),
    ]
    provider = StubEmailProvider(messages=messages)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    registry.register(AskUserStub())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = EmailAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
        agent_overrides={"provider_instance": provider, "now": now.isoformat()},
    )

    agent.run(AgentContext(messages=[{"role": "user", "content": "Summarize"}]))
    summary_path = agent.last_result.summary_json_path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["total_messages"] == 2
    assert payload["importance"]["high"]
    assert payload["importance"]["high"][0]["message_id"] == "m1"
    assert payload["importance"]["low"]
    task_ids = [task["message_id"] for task in payload["tasks"]]
    assert "m1" in task_ids


def test_email_assistant_requires_approval_before_send(tmp_path: Path) -> None:
    provider = StubEmailProvider(messages=[])

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    registry.register(AskUserStub(responses=["no"]))
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = EmailAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
        agent_overrides={"provider_instance": provider},
    )

    payload = {
        "action": "draft",
        "send": True,
        "draft": {
            "recipients": ["me@example.com"],
            "subject": "Hello",
            "body": "Body text",
        },
    }
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    assert provider.created_drafts
    assert not provider.sent_drafts

    registry = ToolRegistry()
    registry.register(AskUserStub(responses=["yes"]))
    sandbox = ToolSandbox(config.agents.sandbox)
    provider = StubEmailProvider(messages=[])
    agent = EmailAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
        agent_overrides={"provider_instance": provider},
    )
    agent.run(AgentContext(messages=[{"role": "user", "content": json.dumps(payload)}]))
    assert provider.sent_drafts


def test_email_assistant_output_in_artifacts_dir(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    provider = StubEmailProvider(messages=[])
    agent = EmailAssistantAgent(
        config.llm, registry, sandbox, config.agents,
        lsm_config=config,
        agent_overrides={"provider_instance": provider},
    )
    agent.run(AgentContext(messages=[{"role": "user", "content": "Email summary"}]))
    assert agent.last_result is not None
    assert agent.last_result.summary_path.parent.name == "artifacts"
    assert agent.last_result.summary_json_path.parent.name == "artifacts"


def test_email_assistant_has_no_tokens_used_attribute(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    provider = StubEmailProvider(messages=[])
    agent = EmailAssistantAgent(
        config.llm, registry, sandbox, config.agents,
        lsm_config=config,
        agent_overrides={"provider_instance": provider},
    )
    agent.run(AgentContext(messages=[{"role": "user", "content": "Email summary"}]))

    with pytest.raises(AttributeError):
        getattr(agent, "_tokens_used")


# ---------------------------------------------------------------------------
# Source-inspection tests
# ---------------------------------------------------------------------------

def test_email_assistant_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.assistants.email_assistant as email_module

    source = inspect.getsource(email_module)
    assert "AgentHarness(" not in source


def test_email_assistant_has_no_direct_sandbox_execute_call() -> None:
    import lsm.agents.assistants.email_assistant as email_module

    source = inspect.getsource(email_module)
    assert "sandbox.execute(" not in source


def test_email_assistant_bind_interaction_tools_wires_ask_user(tmp_path: Path) -> None:
    """_bind_interaction_tools() binds ask_user to the interaction channel via _Harness stub."""
    from lsm.agents.tools.ask_user import AskUserTool

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None

    ask_user_tool = AskUserTool()
    registry = ToolRegistry()
    registry.register(ask_user_tool)
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = EmailAssistantAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )
    _ = agent  # construction triggers _bind_interaction_tools()

    # After construction, the ask_user tool should have been bound to a _Harness stub
    assert ask_user_tool._harness is not None, (
        "_bind_interaction_tools() must bind ask_user to a harness stub"
    )
    assert ask_user_tool._harness.agent_config is config.agents
    assert ask_user_tool._harness.interaction_channel is sandbox.interaction_channel
