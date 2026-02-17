"""
Cost tracking utilities for query sessions.

Tracks estimated tokens and costs per provider and supports CSV export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from lsm.config.models import LSMConfig
from lsm.providers import create_provider
from lsm.query.session import SessionState, Candidate


@dataclass
class CostEntry:
    """Single cost record for an LLM call."""

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: Optional[float]
    kind: str


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate from text length.

    Uses ~4 characters per token as a heuristic.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def estimate_output_tokens(text: Optional[str], max_tokens: Optional[int]) -> int:
    """Estimate output tokens from text or fallback to max_tokens."""
    if text:
        return estimate_tokens(text)
    if max_tokens:
        return max_tokens
    return 0


def estimate_synthesis_cost(
    provider,
    question: str,
    context: str,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    """Estimate cost for synthesis step."""
    input_tokens = estimate_tokens(f"{question}\n{context}")
    output_tokens = max_tokens or 0
    cost = provider.estimate_cost(input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def estimate_rerank_cost(
    provider,
    question: str,
    candidates: List[Candidate],
    k: int,
) -> Dict[str, Any]:
    """Estimate cost for reranking step."""
    combined = question + "\n" + "\n".join(c.text or "" for c in candidates[:k])
    input_tokens = estimate_tokens(combined)
    output_tokens = max(50, k * 30)
    cost = provider.estimate_cost(input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def estimate_query_cost(
    question: str,
    config: LSMConfig,
    state: SessionState,
    embedder,
    collection,
) -> float:
    """
    Estimate costs for a query without invoking the LLM.
    """
    from lsm.query.planning import prepare_local_candidates

    mode_config = config.get_mode_config()
    local_policy = mode_config.source_policy.local
    remote_policy = mode_config.source_policy.remote

    plan = prepare_local_candidates(question, config, state, embedder, collection)

    if not local_policy.enabled:
        query_config = config.llm.get_query_config()
        synthesis_provider = create_provider(query_config)
        synth_est = estimate_synthesis_cost(
            synthesis_provider,
            question,
            "",
            query_config.max_tokens,
        )
        return synth_est["cost"] or 0.0

    if not plan.candidates and not remote_policy.enabled:
        return 0.0

    if not plan.filtered and not remote_policy.enabled:
        return 0.0

    if plan.relevance < plan.min_relevance and not remote_policy.enabled:
        return 0.0

    total_estimated = 0.0

    if plan.should_llm_rerank:
        ranking_config = config.llm.get_ranking_config()
        rerank_provider = create_provider(ranking_config)
        chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
        rerank_est = estimate_rerank_cost(
            rerank_provider,
            question,
            chosen,
            k=min(plan.k_rerank, len(chosen)),
        )
        total_estimated += rerank_est["cost"] or 0.0

    from lsm.query.context import build_context_block

    chosen = plan.filtered[: min(plan.k_rerank, len(plan.filtered))]
    context_block, _ = build_context_block(chosen)
    query_config = config.llm.get_query_config()
    synthesis_provider = create_provider(query_config)
    synth_est = estimate_synthesis_cost(
        synthesis_provider,
        question,
        context_block,
        query_config.max_tokens,
    )
    total_estimated += synth_est["cost"] or 0.0

    return total_estimated


@dataclass
class CostTracker:
    """Session-level cost tracker."""

    budget_limit: Optional[float] = None
    warn_threshold: float = 0.8
    entries: List[CostEntry] = field(default_factory=list)

    def add_entry(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float],
        kind: str,
    ) -> None:
        self.entries.append(
            CostEntry(
                timestamp=datetime.utcnow(),
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                kind=kind,
            )
        )

    def total_cost(self) -> float:
        return sum(entry.cost or 0.0 for entry in self.entries)

    def total_calls(self) -> int:
        return len(self.entries)

    def provider_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for entry in self.entries:
            stats = summary.setdefault(entry.provider, {"calls": 0, "cost": 0.0})
            stats["calls"] += 1
            stats["cost"] += entry.cost or 0.0
        return summary

    def period_summary(self, period: str) -> Dict[str, float]:
        buckets: Dict[str, float] = {}
        for entry in self.entries:
            ts = entry.timestamp
            if period == "daily":
                key = ts.strftime("%Y-%m-%d")
            elif period == "weekly":
                key = f"{ts.strftime('%Y')}-W{ts.isocalendar().week:02d}"
            elif period == "monthly":
                key = ts.strftime("%Y-%m")
            else:
                continue
            buckets[key] = buckets.get(key, 0.0) + (entry.cost or 0.0)
        return buckets

    def budget_status(self) -> Optional[str]:
        if self.budget_limit is None:
            return None
        total = self.total_cost()
        if total >= self.budget_limit:
            return f"Budget exceeded (${total:.4f} >= ${self.budget_limit:.4f})"
        if total >= self.budget_limit * self.warn_threshold:
            return f"Budget warning (${total:.4f} / ${self.budget_limit:.4f})"
        return f"Budget ok (${total:.4f} / ${self.budget_limit:.4f})"

    def format_summary(self) -> str:
        lines = []
        total = self.total_cost()
        lines.append("SESSION COST SUMMARY")
        lines.append(f"Total calls: {self.total_calls()}")
        lines.append(f"Total cost:  ${total:.4f}")
        status = self.budget_status()
        if status:
            lines.append(status)

        if self.entries:
            lines.append("")
            lines.append("By provider:")
            for provider, stats in sorted(self.provider_summary().items()):
                lines.append(
                    f"- {provider}: calls={int(stats['calls'])} cost=${stats['cost']:.4f}"
                )

        for period in ("daily", "weekly", "monthly"):
            buckets = self.period_summary(period)
            if buckets:
                lines.append("")
                lines.append(f"{period.capitalize()} totals:")
                for key, cost in sorted(buckets.items()):
                    lines.append(f"- {key}: ${cost:.4f}")

        return "\n".join(lines)

    def export_csv(self, path: Path) -> Path:
        lines = ["timestamp,provider,model,kind,input_tokens,output_tokens,cost"]
        for entry in self.entries:
            lines.append(
                f"{entry.timestamp.isoformat()},{entry.provider},{entry.model},"
                f"{entry.kind},{entry.input_tokens},{entry.output_tokens},"
                f"{entry.cost if entry.cost is not None else ''}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
