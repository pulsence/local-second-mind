"""
Multi-hop retrieval strategies.

Supports parallel (decompose → retrieve all → merge) and iterative
(hop-by-hop with LLM-guided refinement) approaches.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from lsm.logging import get_logger
from lsm.query.pipeline_types import (
    ContextPackage,
    QueryRequest,
    QueryResponse,
)
from lsm.query.session import Candidate

if TYPE_CHECKING:
    from lsm.query.pipeline import RetrievalPipeline

logger = get_logger(__name__)

_JSON_BLOCK_RE = re.compile(r"\[[\s\S]*\]")

_DECOMPOSE_PROMPT = """Decompose the following complex question into 2-4 simpler sub-questions that can each be answered independently. Return ONLY a JSON array of strings, no markdown or explanation.

Question: {question}"""

_ITERATIVE_REFINE_PROMPT = """Based on the partial answer so far, generate the next follow-up question to deepen understanding. Return ONLY the follow-up question as a single string, no JSON or explanation.

Original question: {question}
Partial answer: {partial_answer}"""


@dataclass
class MultiHopRequest:
    """Request for multi-hop retrieval."""

    query: str
    max_hops: int = 3
    strategy: str = "parallel"
    mode: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass
class MultiHopResult:
    """Result from multi-hop retrieval."""

    answer: str
    sub_questions: List[str]
    hop_responses: List[QueryResponse]
    merged_candidates: List[Candidate]
    total_hops: int


def parallel_multi_hop(
    request: MultiHopRequest,
    pipeline: "RetrievalPipeline",
    llm_provider: Any,
) -> MultiHopResult:
    """Parallel multi-hop: decompose → retrieve all sub-questions → merge → synthesize.

    Args:
        request: Multi-hop request with query and max_hops.
        pipeline: RetrievalPipeline instance.
        llm_provider: LLM provider for decomposition and synthesis.

    Returns:
        MultiHopResult with merged answer and all sub-question responses.
    """
    # 1. Decompose query into sub-questions
    sub_questions = _decompose_query(request.query, llm_provider, request.max_hops)
    if not sub_questions:
        sub_questions = [request.query]

    # 2. Retrieve for each sub-question
    hop_responses: List[QueryResponse] = []
    all_candidates: List[Candidate] = []
    seen_cids: set = set()

    for sq in sub_questions:
        sub_request = QueryRequest(
            question=sq,
            mode=request.mode,
            conversation_id=request.conversation_id,
        )
        package = pipeline.build_sources(sub_request)
        package = pipeline.synthesize_context(package)

        # Merge candidates (deduplicated)
        for c in package.candidates:
            if c.cid not in seen_cids:
                seen_cids.add(c.cid)
                all_candidates.append(c)

        # Get individual response
        response = pipeline.execute(package)
        hop_responses.append(response)

    # 3. Synthesize final answer from all sub-answers
    combined_answer = _merge_answers(
        request.query, sub_questions, hop_responses, llm_provider,
    )

    return MultiHopResult(
        answer=combined_answer,
        sub_questions=sub_questions,
        hop_responses=hop_responses,
        merged_candidates=all_candidates,
        total_hops=len(sub_questions),
    )


def iterative_multi_hop(
    request: MultiHopRequest,
    pipeline: "RetrievalPipeline",
    llm_provider: Any,
) -> MultiHopResult:
    """Iterative multi-hop: each hop informs the next via partial answers.

    Args:
        request: Multi-hop request with query and max_hops.
        pipeline: RetrievalPipeline instance.
        llm_provider: LLM provider for refinement and synthesis.

    Returns:
        MultiHopResult with final answer built from iterative hops.
    """
    sub_questions: List[str] = [request.query]
    hop_responses: List[QueryResponse] = []
    all_candidates: List[Candidate] = []
    seen_cids: set = set()
    partial_answer = ""
    prior_response_id = None

    for hop in range(request.max_hops):
        current_question = sub_questions[-1]

        sub_request = QueryRequest(
            question=current_question,
            mode=request.mode,
            conversation_id=request.conversation_id,
            prior_response_id=prior_response_id,
        )
        package = pipeline.build_sources(sub_request)
        package = pipeline.synthesize_context(package)
        response = pipeline.execute(package)

        hop_responses.append(response)
        prior_response_id = response.response_id

        # Merge candidates
        for c in response.package.candidates:
            if c.cid not in seen_cids:
                seen_cids.add(c.cid)
                all_candidates.append(c)

        partial_answer = response.answer

        # Check if we've reached max hops or have enough info
        if hop >= request.max_hops - 1:
            break

        # Generate next sub-question
        next_question = _generate_followup(
            request.query, partial_answer, llm_provider,
        )
        if not next_question or next_question.lower().strip() == current_question.lower().strip():
            break
        sub_questions.append(next_question)

    return MultiHopResult(
        answer=partial_answer,
        sub_questions=sub_questions,
        hop_responses=hop_responses,
        merged_candidates=all_candidates,
        total_hops=len(hop_responses),
    )


def _decompose_query(
    query: str,
    llm_provider: Any,
    max_sub_questions: int = 4,
) -> List[str]:
    """Decompose a query into sub-questions using LLM."""
    try:
        prompt = _DECOMPOSE_PROMPT.format(question=query)
        response = llm_provider.send_message(input=prompt)
        return _parse_sub_questions(response, max_sub_questions)
    except Exception as exc:
        logger.debug("Query decomposition failed: %s", exc)
        return [query]


def _parse_sub_questions(response: str, max_count: int) -> List[str]:
    """Parse LLM response into a list of sub-questions."""
    text = (response or "").strip()
    if not text:
        return []

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()][:max_count]
    except Exception:
        pass

    # Try extracting JSON array
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()][:max_count]
        except Exception:
            pass

    # Fallback: split by newlines
    lines = [l.strip().lstrip("0123456789.-) ") for l in text.split("\n") if l.strip()]
    return [l for l in lines if len(l) > 10][:max_count]


def _generate_followup(
    original_query: str,
    partial_answer: str,
    llm_provider: Any,
) -> Optional[str]:
    """Generate a follow-up question based on partial answer."""
    try:
        prompt = _ITERATIVE_REFINE_PROMPT.format(
            question=original_query,
            partial_answer=partial_answer[:500],
        )
        response = llm_provider.send_message(input=prompt)
        result = (response or "").strip().strip('"').strip("'")
        return result if result else None
    except Exception as exc:
        logger.debug("Follow-up generation failed: %s", exc)
        return None


def _merge_answers(
    original_query: str,
    sub_questions: List[str],
    responses: List[QueryResponse],
    llm_provider: Any,
) -> str:
    """Merge sub-question answers into a comprehensive final answer."""
    if len(responses) == 1:
        return responses[0].answer

    parts = []
    for i, (sq, resp) in enumerate(zip(sub_questions, responses), 1):
        parts.append(f"Sub-question {i}: {sq}\nAnswer: {resp.answer}")

    merge_prompt = (
        f"Original question: {original_query}\n\n"
        "Below are answers to sub-questions. Synthesize them into a single "
        "comprehensive answer. Preserve any [S1], [S2] etc. citations.\n\n"
        + "\n\n".join(parts)
    )

    try:
        return llm_provider.send_message(input=merge_prompt)
    except Exception:
        # Fallback: concatenate
        return "\n\n".join(r.answer for r in responses)
