from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseResult:
    final_text: str
    tool_calls: list[dict]
    stop_reason: str