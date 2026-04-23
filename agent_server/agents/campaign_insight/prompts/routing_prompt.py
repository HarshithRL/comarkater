"""Keyword dictionaries + optional LLM prompt for the Query Router."""
from __future__ import annotations


GENIE_DIRECT_KEYWORDS: tuple[str, ...] = (
    "compare",
    "top",
    " by ",
    "list ",
    "rank",
    "show me",
    "group by",
    "breakdown",
    "distribution",
    "per ",
    "each ",
)

HYBRID_KEYWORDS: tuple[str, ...] = (
    "performance of",
    "how did",
    "what was",
    "overview",
    "summary",
    "summarise",
    "summarize",
    "overall",
)

DECOMPOSE_KEYWORDS: tuple[str, ...] = (
    "why",
    "explain",
    "recommend",
    "recommendation",
    "drill",
    "reasons",
    "suggest",
    "root cause",
    "and then",
    "followed by",
)


ROUTING_SYSTEM_PROMPT = """You are the Query Router for a marketing analytics agent.

Classify the user's question into exactly one of three execution strategies:

1. GENIE_DIRECT - a single aggregated SQL query answers the whole question.
   Examples: "compare CTR by segment", "top 10 campaigns by conversions".

2. HYBRID - one aggregated query plus light interpretation of the numbers.
   Examples: "how did email perform last week", "summary of last month".

3. AGENT_DECOMPOSE - requires multi-step reasoning, drill-downs, cross-dimension
   comparisons, or causal explanation.
   Examples: "why did open rate drop", "find worst campaigns and recommend fixes".

Rules:
- Prefer the simplest strategy that still answers the question.
- If the question asks "why", "explain", "recommend", or requires more than one
  Genie query, pick AGENT_DECOMPOSE.
- If the user wants only data (comparisons, rankings, groupings), pick GENIE_DIRECT.
- Output a short reason (<= 20 words).
"""
