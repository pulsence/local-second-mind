"""
Environment scrubbing for sandboxed tool execution.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Mapping, Optional

_MINIMAL_ENV_KEYS = ("PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "LANG")
_SECRET_ENV_NAME = re.compile(r"(_API_KEY$|_SECRET|_TOKEN|_PASSWORD)", re.IGNORECASE)


def scrub_environment(source_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """
    Build a minimal environment safe for tool execution.

    Args:
        source_env: Optional source environment mapping. Defaults to ``os.environ``.

    Returns:
        Scrubbed environment dictionary containing only minimal non-secret keys.
    """
    env = source_env or os.environ
    clean: Dict[str, str] = {}
    for key in _MINIMAL_ENV_KEYS:
        value = env.get(key)
        if value is None:
            continue
        if _SECRET_ENV_NAME.search(key):
            continue
        clean[key] = str(value)
    return clean
