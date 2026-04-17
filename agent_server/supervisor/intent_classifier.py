"""Supervisor intent classifier — 1 LLM call with a regex greeting fast-path."""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from supervisor.domain_context import SupervisorDomainContext

logger = logging.getLogger(__name__)

# Reuse the greeting regex shape from agents/supervisor.py (kept local to avoid
# a cross-package dependency and to keep this module self-contained).
_GREETING_PATTERNS = re.compile(
    r"^(?:hi+|hey+|hello+|howdy|hola|yo+|sup|good\s*(?:morning|afternoon|evening|night)"
    r"|what'?s?\s*up|greetings|namaste|heya?"
    r"|thanks?|thank\s*you|thx"
    r"|bye|goodbye|see\s*ya|cheers"
    r"|how\s*are\s*you|how'?s?\s*it\s*going"
    r"|welcome|ok|okay|cool|great|got\s*it|sure)\s*[?!.,]*$",
    re.IGNORECASE,
)
_GREETING_WITH_NAME = re.compile(
    r"^(hi+|hey+|hello|good\s*(morning|afternoon|evening|day))\s*[,!]?\s*\w+[\s!.,?]*$",
    re.IGNORECASE,
)


def _is_greeting(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 50:
        return False
    return bool(_GREETING_PATTERNS.match(stripped) or _GREETING_WITH_NAME.match(stripped))


class _IntentModel(BaseModel):
    """Pydantic schema for structured intent classification output."""

    intent_type: str = Field(description="Intent name from the taxonomy, or 'clarification' / 'out_of_scope'.")
    complexity: str = Field(default="simple", description="One of: simple, medium, complex.")
    channels_mentioned: list[str] = Field(default_factory=list)
    metrics_mentioned: list[str] = Field(default_factory=list)
    time_context: Optional[str] = Field(default=None)
    requires_audience: bool = Field(default=False)
    requires_content: bool = Field(default=False)


_PROMPT = """You are the supervisor for a marketing analytics assistant.

The system supports THREE analysis dimensions:

1. Campaign Performance (ALWAYS INCLUDED)
   - delivery, engagement, conversions, trends, rankings

2. Audience Analysis (ONLY when explicitly requested)
   - segments, lifecycle stage, intent, demographics, device, communication health
   - keywords: "segment", "lifecycle", "who opened", "audience", "user", "intent"

3. Content Analysis (ONLY when explicitly requested)
   - messaging, templates, keywords, CTA, emotions, content quality, benchmarks
   - keywords: "content", "template", "CTA", "keyword", "emotion", "message", "subject line"

4. Mixed Analysis
   - when both audience AND content are referenced

A question is IN-SCOPE if it relates to ANY of the above dimensions.

A question is OUT-OF-SCOPE ONLY if it is NOT related to marketing analytics
(e.g., weather, pricing, HR, general knowledge, jokes).

---

Classification Rules:

- If the question is purely conversational (greeting, thanks), use intent_type="greeting"
- If the question is ambiguous or incomplete, use intent_type="clarification"
- If the question is NOT related to marketing analytics, use intent_type="out_of_scope"
- Otherwise, use one of the intent names from the taxonomy

---

Dimension Signals:

- requires_audience = true ONLY if audience/segment-related concepts are explicitly mentioned
- requires_content = true ONLY if content/messaging-related concepts are explicitly mentioned

IMPORTANT:

- Audience queries are VALID — do NOT mark them out_of_scope
- Content queries are VALID — do NOT mark them out_of_scope
- Mixed queries are VALID — do NOT mark them out_of_scope
- Campaign is always implicitly included

---

{domain_context}

{history_block}

User question: {query}
"""


class IntentClassifier:
    """Single-call supervisor intent classifier."""

    def __init__(self, llm: Any, domain_context: SupervisorDomainContext) -> None:
        """Store the LLM and domain context.

        Args:
            llm: A LangChain chat model with ``with_structured_output`` support.
            domain_context: Condensed domain reference supplier.
        """
        self._llm = llm
        self._domain_context = domain_context

    def classify(self, query: str, conversation_history: list | None = None) -> dict:
        """Classify a user query into an intent dict.

        Args:
            query: Raw user question.
            conversation_history: Optional list of prior turns (last 3 used).

        Returns:
            Dict with keys: intent_type, complexity, channels_mentioned,
            metrics_mentioned, time_context, requires_audience,
            requires_content, target_agent.
        """
        q = (query or "").strip()
        logger.info(f"[INTENT_CLASSIFIER] classify() start | query={q[:120]!r} | history_turns={len(conversation_history or [])}")
        if _is_greeting(q):
            logger.info("[INTENT_CLASSIFIER] regex greeting fast-path → intent=greeting")
            return {
                "intent_type": "greeting",
                "complexity": "simple",
                "channels_mentioned": [],
                "metrics_mentioned": [],
                "time_context": None,
                "requires_audience": False,
                "requires_content": False,
                "target_agent": None,
            }

        history_block = ""
        if conversation_history:
            recent = conversation_history[-3:]
            rendered: list[str] = []
            for turn in recent:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = str(turn.get("content", ""))[:200]
                    rendered.append(f"{role}: {content}")
                else:
                    rendered.append(str(turn)[:200])
            if rendered:
                history_block = "Recent conversation:\n" + "\n".join(rendered) + "\n\n"

        prompt = _PROMPT.format(
            domain_context=self._domain_context.format_for_prompt(),
            history_block=history_block,
            query=q,
        )

        try:
            logger.info(f"[INTENT_CLASSIFIER] LLM call | prompt_len={len(prompt)}ch")
            structured = self._llm.with_structured_output(_IntentModel)
            result: _IntentModel = structured.invoke(prompt)
            intent_type = (result.intent_type or "performance_lookup").strip()
            complexity = (result.complexity or "simple").strip().lower()
            if complexity not in ("simple", "medium", "complex"):
                complexity = "simple"
            target_agent: Optional[str]
            if intent_type in ("greeting", "clarification", "out_of_scope"):
                target_agent = None
            else:
                target_agent = "campaign_insight"
            logger.info(
                f"[INTENT_CLASSIFIER] result | intent_type={intent_type} complexity={complexity} "
                f"channels={list(result.channels_mentioned or [])} metrics={list(result.metrics_mentioned or [])} "
                f"time={result.time_context!r} requires_audience={bool(result.requires_audience)} "
                f"requires_content={bool(result.requires_content)} target_agent={target_agent}"
            )
            return {
                "intent_type": intent_type,
                "complexity": complexity,
                "channels_mentioned": list(result.channels_mentioned or []),
                "metrics_mentioned": list(result.metrics_mentioned or []),
                "time_context": result.time_context,
                "requires_audience": bool(result.requires_audience),
                "requires_content": bool(result.requires_content),
                "target_agent": target_agent,
            }
        except Exception as exc:
            logger.warning("IntentClassifier fallback (LLM error): %s", exc)
            return {
                "intent_type": "performance_lookup",
                "complexity": "simple",
                "channels_mentioned": [],
                "metrics_mentioned": [],
                "time_context": None,
                "requires_audience": False,
                "requires_content": False,
                "target_agent": "campaign_insight",
            }
