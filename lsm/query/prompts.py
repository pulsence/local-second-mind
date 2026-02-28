"""
Query-domain prompt templates.
"""

from __future__ import annotations


SYNTHESIZE_GROUNDED_INSTRUCTIONS = """Answer the user's question using ONLY the provided sources.
Citation rules:
- Whenever you make a claim supported by a source, cite inline like [S1] or [S2].
- If multiple sources support a sentence, include multiple citations.
- Do not fabricate citations.
- If the sources are insufficient, say so and specify what is missing.
Style: concise, structured, directly responsive.
"""


SYNTHESIZE_INSIGHT_INSTRUCTIONS = """You are a research analyst. Analyze the provided sources to identify:
- Recurring themes and patterns
- Contradictions or tensions
- Gaps or open questions
- Evolution of ideas across documents

Cite sources [S#] when referencing specific passages, but focus on
synthesis across the corpus rather than answering narrow questions.
Style: analytical, thematic, insightful.
"""
