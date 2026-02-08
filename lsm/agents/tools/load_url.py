"""
Tool for loading URL content.
"""

from __future__ import annotations

from typing import Any, Dict

import requests

from .base import BaseTool


class LoadURLTool(BaseTool):
    """Fetch textual content from a URL."""

    name = "load_url"
    description = "Fetch text content from a URL."
    requires_permission = True
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
            "timeout": {"type": "number", "description": "Timeout in seconds."},
            "max_chars": {"type": "integer", "description": "Maximum output length."},
        },
        "required": ["url"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        url = str(args.get("url", "")).strip()
        if not url:
            raise ValueError("url is required")
        timeout = float(args.get("timeout", 10))
        max_chars = int(args.get("max_chars", 10_000))
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        text = response.text
        if max_chars > 0:
            text = text[:max_chars]
        return text

