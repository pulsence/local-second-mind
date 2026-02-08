"""
Base abstractions for agent tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """
    Abstract base class for tools callable by agents.
    """

    name: str = "base_tool"
    description: str = "Base tool"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    requires_permission: bool = False

    def get_definition(self) -> Dict[str, Any]:
        """
        Build LLM-facing tool metadata.

        Returns:
            Tool definition dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "requires_permission": self.requires_permission,
        }

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> str:
        """
        Execute tool logic.

        Args:
            args: Parsed JSON arguments.

        Returns:
            Tool output as a string.
        """


class ToolRegistry:
    """
    Registry for agent tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If tool name is empty or duplicated.
        """
        name = str(tool.name).strip()
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = tool

    def lookup(self, name: str) -> BaseTool:
        """
        Get a registered tool by name.

        Args:
            name: Tool name.

        Returns:
            Registered tool instance.

        Raises:
            KeyError: If name is unknown.
        """
        return self._tools[name]

    def list_tools(self) -> List[BaseTool]:
        """
        List registered tools.

        Returns:
            Tool instances sorted by name.
        """
        return [self._tools[name] for name in sorted(self._tools.keys())]

    def list_definitions(self) -> List[Dict[str, Any]]:
        """
        List tool definitions for LLM prompts.

        Returns:
            List of tool definition dictionaries.
        """
        return [tool.get_definition() for tool in self.list_tools()]

