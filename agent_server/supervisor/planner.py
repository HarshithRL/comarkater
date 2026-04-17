"""Supervisor planner â€” decomposes complex queries into sub-tasks.

Only invoked for ``complexity == "complex"`` queries. Simple/medium queries
delegate directly to the subagent without a supervisor-level plan.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from supervisor.domain_context import SupervisorDomainContext

logger = logging.getLogger(__name__)


class _PlanModel(BaseModel):
    """Structured-output schema for supervisor plans."""

    steps: list[str] = Field(default_factory=list, description="Ordered list of plain-English sub-tasks.")


_PROMPT = """You are planning a multi-step marketing analytics investigation.

Break the user's question into a small, ordered set of plain-English sub-tasks
that the downstream analytics agent can execute in sequence.

---

{domain_context}

---

## DIMENSION CONTROL (CRITICAL)

The system supports three dimensions:

* Campaign (ALWAYS baseline: metrics, filters, time)
* Audience (ONLY if requires_audience = true)
* Content (ONLY if requires_content = true)

---

## INPUT

User question: {query}
Classified intent: {intent}

From intent:

* requires_audience = intent.requires_audience
* requires_content = intent.requires_content

---

## STRICT RULES (MUST FOLLOW)

* Campaign dimension is ALWAYS included

* ONLY include Audience if requires_audience = true

* ONLY include Content if requires_content = true

* DO NOT introduce audience concepts (segments, lifecycle, users) unless requires_audience = true

* DO NOT introduce content concepts (template, header, CTA, keywords) unless requires_content = true

* DO NOT change the problem type

* DO NOT hallucinate new dimensions

---

## PLANNING RULES

* Produce between 2 and 5 steps

* Each step is ONE sentence describing a concrete analytics task

* Steps should build logically (filter → analyze → compare → summarize)

* Preserve:

  * channel (e.g., WhatsApp)
  * time range
  * filters (e.g., header type)

---

## EXAMPLES

### Content-only

User:
"Show WhatsApp campaigns last 3 months that used a text header instead of an image header"

requires_content = true
requires_audience = false

Correct:

1. Filter WhatsApp campaigns from the last 3 months by header type (text vs image).
2. Compare performance metrics such as delivered, clicks, and CTR across header types.
3. Summarize which header type performs better.

❌ DO NOT add audience segmentation

---

### Audience-only

User:
"Which audience segments have highest CTR?"

Correct:

1. Group campaign performance by audience segment.
2. Calculate CTR for each segment.
3. Identify top-performing segments.

---

### Mixed

User:
"Which segments respond best to different content types?"

Correct:

1. Group performance by audience segment and content type.
2. Compare CTR across segment-content combinations.
3. Identify best-performing combinations.

---

## OUTPUT

Return ONLY the list of steps. Do not include explanations.
"""


class SupervisorPlanner:
    """Generates a lightweight plan for complex queries only."""

    def __init__(self, llm: Any, domain_context: SupervisorDomainContext) -> None:
        """Store LLM and domain context.

        Args:
            llm: LangChain chat model with ``with_structured_output``.
            domain_context: Condensed domain reference supplier.
        """
        self._llm = llm
        self._domain_context = domain_context

    def plan(self, query: str, intent: dict) -> list[str]:
        """Return a list of plain-English sub-tasks, or [] if not needed.

        Args:
            query: The user query.
            intent: Intent dict produced by :class:`IntentClassifier`.

        Returns:
            A list of up to 5 sub-task strings; empty list when the query is
            not complex.
        """
        if (intent or {}).get("complexity") != "complex":
            return []

        prompt = _PROMPT.format(
            domain_context=self._domain_context.format_for_prompt(),
            query=(query or "").strip(),
            intent=intent,
        )

        try:
            structured = self._llm.with_structured_output(_PlanModel)
            result: _PlanModel = structured.invoke(prompt)
            steps = [s.strip() for s in (result.steps or []) if isinstance(s, str) and s.strip()]
            return steps[:5] if steps else [query]
        except Exception as exc:
            logger.warning("SupervisorPlanner fallback (LLM error): %s", exc)
            return [query] if query else []
