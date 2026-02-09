from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.4.0"

_LAZY_EXPORTS = {
    "ingest": ("lsm.ingest", "ingest"),
    "query": ("lsm.query", "query"),
}

__all__ = ["__version__", "ingest", "query"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lsm' has no attribute '{name}'")
