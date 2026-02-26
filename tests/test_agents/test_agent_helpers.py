from __future__ import annotations

from lsm.agents.base import AgentState, BaseAgent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry


class HelperAgent(BaseAgent):
    name = "helper"
    description = "Helper test agent."
    tool_allowlist = {"alpha", "beta"}

    def run(self, initial_context: AgentContext) -> AgentState:
        _ = initial_context
        return self.state


class AlphaTool(BaseTool):
    name = "alpha"
    description = "Alpha description."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("query", ""))


class BetaTool(BaseTool):
    name = "beta"
    description = "Beta description."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("path", ""))


def test_base_agent_helpers_track_budget_parse_json_and_log() -> None:
    agent = HelperAgent()
    agent.max_tokens_budget = 10

    parsed = agent._parse_json('{"ok": true}')
    assert parsed == {"ok": True}
    assert agent._parse_json("not-json") is None

    used_1 = agent._consume_tokens("x" * 20)
    assert used_1 == 5
    assert agent._budget_exhausted() is False

    used_2 = agent._consume_tokens("y" * 24)
    assert used_2 == 6
    assert agent._budget_exhausted() is True

    agent._log("hello", actor="agent")
    assert len(agent.state.log_entries) == 1
    assert agent.state.log_entries[0].content == "hello"
    assert agent.state.log_entries[0].actor == "agent"


def test_base_agent_helpers_format_and_parse_tool_selection() -> None:
    agent = HelperAgent()
    registry = ToolRegistry()
    registry.register(AlphaTool())
    registry.register(BetaTool())

    formatted = agent._format_tool_definitions_for_prompt(registry)
    assert "Alpha description." in formatted
    assert "Beta description." in formatted
    assert '"required": [' in formatted

    agent.tool_allowlist = {"beta"}
    filtered = agent._get_tool_definitions(registry)
    assert len(filtered) == 1
    assert filtered[0]["name"] == "beta"

    selected = agent._parse_tool_selection(
        '{"tools": ["BETA", "alpha", "unknown", "alpha"]}',
        ["alpha", "beta"],
    )
    assert selected == ["beta", "alpha"]

    assert agent._parse_tool_selection("invalid-json", ["alpha", "beta"]) == []
