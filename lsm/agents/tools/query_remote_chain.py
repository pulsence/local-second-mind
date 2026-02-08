"""
Tool for executing configured remote provider chains.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.config.models import LSMConfig
from lsm.remote.chain import RemoteProviderChain

from .base import BaseTool


class QueryRemoteChainTool(BaseTool):
    """Execute a named remote provider chain."""

    name = "query_remote_chain"
    description = "Run a configured remote provider chain using structured input."
    input_schema = {
        "type": "object",
        "properties": {
            "chain": {"type": "string", "description": "Configured chain name."},
            "input": {"type": "object", "description": "Chain input object."},
            "max_results": {"type": "integer", "description": "Result limit per link."},
        },
        "required": ["chain", "input"],
    }

    def __init__(self, config: LSMConfig) -> None:
        self.config = config

    def execute(self, args: Dict[str, Any]) -> str:
        chain_name = str(args.get("chain", "")).strip()
        if not chain_name:
            raise ValueError("chain is required")

        chain_input = args.get("input", {})
        if not isinstance(chain_input, dict):
            raise ValueError("input must be an object")

        chain_config = self.config.get_remote_provider_chain(chain_name)
        if chain_config is None:
            raise ValueError(f"Remote provider chain is not configured: {chain_name}")

        max_results = int(args.get("max_results", 5))
        chain = RemoteProviderChain(self.config, chain_config)
        results = chain.execute(chain_input, max_results=max_results)
        return json.dumps(results, indent=2)

