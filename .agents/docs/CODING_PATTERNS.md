# Coding Patterns

This document defines the coding patterns and conventions for Local Second Mind.

## Design Pattern

Core modules (`lsm/query/`, `lsm/ingest/`) contain pure business logic that returns results. UI modules handle command parsing, user interaction, and display formatting. Backwards compatibility is not maintained — no deprecated methods or classes are preserved.

Implementing features should follow a **Test-Driven Design** pattern. When the user reports an error, determine if a test case should be created to cover the error and write a test duplicating the error before fixing it.

## Dataclasses

Use `__post_init__` for normalization and `validate()` for validation:

```python
@dataclass
class MyConfig:
    field: str
    
    def __post_init__(self):
        self.field = self.field.strip()
    
    def validate(self):
        if not self.field:
            raise ValueError("field required")
```

## Providers

Extend ABC base classes:
- `lsm/providers/base.py` - LLM providers
- `lsm/remote/base.py` - Remote source providers
- `lsm/vectordb/base.py` - Vector DB providers

## Naming Conventions

- **snake_case** - functions, variables
- **PascalCase** - classes
- **UPPER_SNAKE_CASE** - constants

## Type Hints

Required on all function signatures. Use `from __future__ import annotations`.

## Docstrings

Google style with Args, Returns, Raises sections:

```python
def function(param: str) -> int:
    """Short description.

    Args:
        param: Description of param.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something fails.
    """
```

## Logging

```python
from lsm.logging import get_logger
logger = get_logger(__name__)
```

## File Organization

- Core business logic in `lsm/query/` and `lsm/ingest/`
- UI code in `lsm/ui/`
- Commands return strings, no print statements
- Display utilities return formatted strings

## Testing

- Tests use pytest
- No mocks or stubs in tests
- Write tests that verify rejection behavior for security tests
- Use fixtures from `conftest.py`
