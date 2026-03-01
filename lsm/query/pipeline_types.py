"""
Core data types for the RetrievalPipeline.

Defines QueryRequest, ContextPackage, QueryResponse, and supporting types
used across all pipeline stages.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lsm.config.models.modes import GROUNDED_MODE


@dataclass
class FilterSet:
    """Metadata filters applied during retrieval."""

    path_contains: Optional[List[str]] = None
    """Substring filters on source_path."""

    ext_allow: Optional[List[str]] = None
    """Allowed file extensions."""

    ext_deny: Optional[List[str]] = None
    """Denied file extensions."""

    def is_active(self) -> bool:
        return bool(self.path_contains or self.ext_allow or self.ext_deny)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_contains": self.path_contains,
            "ext_allow": self.ext_allow,
            "ext_deny": self.ext_deny,
        }


@dataclass
class ScoreBreakdown:
    """Per-candidate score breakdown across retrieval stages."""

    dense_score: Optional[float] = None
    """Cosine similarity score from dense retrieval."""

    dense_rank: Optional[int] = None
    """Rank position from dense retrieval (1-based)."""

    sparse_score: Optional[float] = None
    """BM25 score from FTS5 sparse retrieval."""

    sparse_rank: Optional[int] = None
    """Rank position from sparse retrieval (1-based)."""

    fused_score: Optional[float] = None
    """Score after RRF fusion."""

    rerank_score: Optional[float] = None
    """Score after cross-encoder or LLM reranking."""

    temporal_boost: Optional[float] = None
    """Temporal recency boost factor."""

    graph_expansion_score: Optional[float] = None
    """Score from graph-augmented retrieval expansion."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in {
                "dense_score": self.dense_score,
                "dense_rank": self.dense_rank,
                "sparse_score": self.sparse_score,
                "sparse_rank": self.sparse_rank,
                "fused_score": self.fused_score,
                "rerank_score": self.rerank_score,
                "temporal_boost": self.temporal_boost,
                "graph_expansion_score": self.graph_expansion_score,
            }.items()
            if v is not None
        }


@dataclass
class Citation:
    """A citation linking an answer claim to a source."""

    chunk_id: str
    """Chunk ID in the vector database."""

    source_path: str
    """File path or URL of the source."""

    heading: Optional[str] = None
    """Section heading within the source."""

    page_number: Optional[int] = None
    """Page number for PDF sources."""

    url_or_doi: Optional[str] = None
    """URL or DOI for remote sources."""

    snippet: Optional[str] = None
    """Relevant text excerpt from the source."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
        }
        if self.heading is not None:
            d["heading"] = self.heading
        if self.page_number is not None:
            d["page_number"] = self.page_number
        if self.url_or_doi is not None:
            d["url_or_doi"] = self.url_or_doi
        if self.snippet is not None:
            d["snippet"] = self.snippet
        return d


@dataclass
class StageTimings:
    """Timing for a single pipeline stage."""

    stage: str
    """Stage name (e.g. 'dense_recall', 'rrf_fusion')."""

    duration_ms: float
    """Duration in milliseconds."""


@dataclass
class RetrievalTrace:
    """Diagnostic trace of retrieval pipeline execution."""

    stages_executed: List[str] = field(default_factory=list)
    """Ordered list of stage names that ran."""

    timings: List[StageTimings] = field(default_factory=list)
    """Per-stage timing data."""

    dense_candidates_count: int = 0
    """Number of candidates from dense recall."""

    sparse_candidates_count: int = 0
    """Number of candidates from sparse recall."""

    fused_candidates_count: int = 0
    """Number of candidates after fusion."""

    reranked_candidates_count: int = 0
    """Number of candidates after reranking."""

    hyde_documents: Optional[List[str]] = None
    """HyDE-generated hypothetical documents (if used)."""

    retrieval_profile: Optional[str] = None
    """Name of the retrieval profile used."""

    def total_duration_ms(self) -> float:
        return sum(t.duration_ms for t in self.timings)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "stages_executed": self.stages_executed,
            "timings": [
                {"stage": t.stage, "duration_ms": round(t.duration_ms, 2)}
                for t in self.timings
            ],
            "dense_candidates_count": self.dense_candidates_count,
            "sparse_candidates_count": self.sparse_candidates_count,
            "fused_candidates_count": self.fused_candidates_count,
            "reranked_candidates_count": self.reranked_candidates_count,
            "total_duration_ms": round(self.total_duration_ms(), 2),
        }
        if self.retrieval_profile:
            d["retrieval_profile"] = self.retrieval_profile
        if self.hyde_documents:
            d["hyde_documents"] = self.hyde_documents
        return d


@dataclass
class CostEntry:
    """Cost record for a single LLM operation."""

    provider: str
    """LLM provider name."""

    model: str
    """Model identifier."""

    input_tokens: int = 0
    """Estimated input tokens."""

    output_tokens: int = 0
    """Estimated output tokens."""

    cost: float = 0.0
    """Estimated monetary cost."""

    kind: str = "synthesis"
    """Operation type: 'synthesis', 'rerank', 'hyde', etc."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "kind": self.kind,
        }


@dataclass
class RemoteSource:
    """A result from a remote provider."""

    provider: str
    """Remote provider name."""

    title: str = ""
    """Result title."""

    url: str = ""
    """Result URL."""

    snippet: str = ""
    """Result text snippet."""

    score: float = 0.5
    """Relevance score from the provider."""

    weight: float = 1.0
    """Provider weight for ranking."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata."""

    def weighted_score(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.weighted_score(),
            "metadata": self.metadata,
        }


@dataclass
class QueryRequest:
    """Input to the retrieval pipeline."""

    question: str
    """The user's query text."""

    mode: Optional[str] = None
    """Mode name. None resolves to 'grounded'."""

    filters: Optional[FilterSet] = None
    """Metadata filters for retrieval."""

    k: Optional[int] = None
    """Override number of results to retrieve."""

    starting_prompt: Optional[str] = None
    """Explicit synthesis prompt (highest priority)."""

    conversation_id: Optional[str] = None
    """Identifier for the conversation session."""

    prior_response_id: Optional[str] = None
    """Response ID from the previous turn (for server cache chaining)."""

    conversation_history: Optional[List[Dict[str, str]]] = None
    """Previous turns in the conversation."""

    chat_mode: str = "single"
    """'single' or 'chat'."""

    pinned_chunks: Optional[List[str]] = None
    """Chunk IDs forced into context."""

    context_documents: Optional[List[str]] = None
    """Document paths as context anchors."""

    context_chunks: Optional[List[str]] = None
    """Chunk IDs as context anchors."""

    model_override: Optional[str] = None
    """Override the synthesis model."""

    @property
    def resolved_mode(self) -> str:
        """Resolve mode name, defaulting to 'grounded'."""
        return self.mode or "grounded"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question": self.question,
            "mode": self.resolved_mode,
            "chat_mode": self.chat_mode,
        }
        if self.filters and self.filters.is_active():
            d["filters"] = self.filters.to_dict()
        if self.k is not None:
            d["k"] = self.k
        if self.starting_prompt is not None:
            d["starting_prompt"] = self.starting_prompt
        if self.conversation_id is not None:
            d["conversation_id"] = self.conversation_id
        if self.prior_response_id is not None:
            d["prior_response_id"] = self.prior_response_id
        if self.model_override is not None:
            d["model_override"] = self.model_override
        return d


@dataclass
class ContextPackage:
    """Intermediate result after source retrieval and context assembly."""

    request: QueryRequest
    """The original query request."""

    candidates: List[Any] = field(default_factory=list)
    """Final chosen candidates (Candidate objects)."""

    remote_sources: List[RemoteSource] = field(default_factory=list)
    """Remote source results."""

    retrieval_trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    """Diagnostic trace of retrieval stages."""

    costs: List[CostEntry] = field(default_factory=list)
    """Cost entries accumulated during retrieval."""

    # Fields populated by synthesize_context()
    context_block: Optional[str] = None
    """Formatted context string for LLM input."""

    source_labels: Optional[Dict[str, Any]] = None
    """Mapping from 'S1', 'S2', ... to candidate/source info."""

    starting_prompt: Optional[str] = None
    """Resolved synthesis prompt."""

    prior_response_id: Optional[str] = None
    """Response ID for server cache continuation."""

    # Internal state
    all_candidates: List[Any] = field(default_factory=list)
    """All candidates before filtering/reranking."""

    filtered_candidates: List[Any] = field(default_factory=list)
    """Candidates after filtering but before final selection."""

    relevance: float = 0.0
    """Best relevance score among candidates."""

    local_enabled: bool = True
    """Whether local retrieval was enabled."""

    def total_cost(self) -> float:
        return sum(c.cost for c in self.costs)


@dataclass
class QueryResponse:
    """Final output from the retrieval pipeline."""

    answer: str
    """Synthesized answer text."""

    package: ContextPackage
    """The context package used for synthesis."""

    citations: List[Citation] = field(default_factory=list)
    """Parsed citations from the answer."""

    costs: List[CostEntry] = field(default_factory=list)
    """Cost entries from synthesis."""

    conversation_id: Optional[str] = None
    """Conversation session identifier."""

    response_id: Optional[str] = None
    """Provider response ID for cache chaining."""

    timestamp: float = field(default_factory=time.time)
    """Response creation timestamp."""

    def total_cost(self) -> float:
        """Total cost across retrieval and synthesis."""
        return self.package.total_cost() + sum(c.cost for c in self.costs)

    @property
    def candidates(self) -> List[Any]:
        """Shortcut to package candidates."""
        return self.package.candidates

    @property
    def remote_sources(self) -> List[RemoteSource]:
        """Shortcut to package remote sources."""
        return self.package.remote_sources

    @property
    def retrieval_trace(self) -> RetrievalTrace:
        """Shortcut to package retrieval trace."""
        return self.package.retrieval_trace

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for agent tool transport."""
        d: Dict[str, Any] = {
            "answer": self.answer,
            "total_cost": self.total_cost(),
            "timestamp": self.timestamp,
        }
        if self.citations:
            d["citations"] = [c.to_dict() for c in self.citations]
        if self.conversation_id:
            d["conversation_id"] = self.conversation_id
        if self.response_id:
            d["response_id"] = self.response_id
        d["retrieval_trace"] = self.retrieval_trace.to_dict()
        d["costs"] = [c.to_dict() for c in self.costs]
        return d
