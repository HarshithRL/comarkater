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


# Strip leading punctuation/whitespace so noisy prefixes like ":", "-", ">"
# don't make the LLM treat the query as malformed/incomplete.
_LEADING_NOISE = re.compile(r"^[\s:;,.\-_>*~`'\"\\/|=+]+")


def _normalize_query(text: str) -> str:
    """Strip leading punctuation noise and collapse internal whitespace."""
    if not text:
        return ""
    cleaned = _LEADING_NOISE.sub("", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


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
   - delivery, engagement, conversions, trends, rankings, comparisons,
     anomalies, recurring patterns, top/bottom performers, drop-offs

2. Audience Analysis (ONLY when explicitly requested)
   - segments, lifecycle stage, intent, demographics, device, communication health
   - keywords: "segment", "lifecycle", "who opened", "audience", "user", "intent",
     "cohort", "persona"

3. Content Analysis (ONLY when explicitly requested)
   - messaging, templates, keywords, CTA, emotions, content quality, benchmarks
   - keywords: "content", "template", "CTA", "keyword", "emotion", "message",
     "subject line", "theme", "topic", "category", "creative", "copy",
     "campaign type", "tag", "headline"

4. Mixed Analysis
   - when both audience AND content are referenced

A question is IN-SCOPE if it relates to ANY of the above dimensions.

A question is OUT-OF-SCOPE ONLY if it is NOT related to marketing analytics
at all (e.g., weather, stock prices, HR, recipes, sports, general knowledge,
jokes, coding help).

The following are IN-SCOPE — never mark them out_of_scope:

- Forward-looking / predictive framings ("predict", "forecast", "expected",
  "what could perform best", "will this work", "lookalike to our best
  campaigns", "if we ran X, how would it do"). The agent answers with
  HISTORICAL benchmarks and patterns, but the question itself is in-scope.
- Recommendation / advice framings ("what should we do", "suggest",
  "recommend", "how can we improve", "what's the best approach for X").
- Diagnostic framings ("why did X drop", "what caused the spike",
  "which campaigns are underperforming", "anomalies").
- Benchmark / comparison framings ("how do we compare to last quarter",
  "lookalike", "similar to top performers").
- Open-ended insight asks ("anything interesting", "what stands out",
  "highlights", "give me a summary").

---

Classification Rules (apply in order):

1. Purely conversational (greeting, thanks, "ok", "got it") → intent_type="greeting"
2. Question is META about the conversation itself — refers to prior turns,
   asks "what did we just discuss", "summarize what you said", "what filters
   are applied", "clarify your last answer" → intent_type="clarification"
3. Question is NOT related to marketing analytics → intent_type="out_of_scope"
4. Otherwise (ANY new data/insight question, even if vague) → pick an intent
   from the taxonomy. DEFAULT to "performance_lookup" when uncertain.

DO NOT use "clarification" just because the question is short, vague, or uses
unfamiliar terminology. If the user is asking for NEW data or insights —
even with words like "themes", "patterns", "anything interesting", "what's
working", "best/worst performers", "trends", "highlights" — classify it as
a data intent (default: performance_lookup) and let the downstream agent
handle the specifics.

"clarification" is reserved for questions that ONLY make sense in the context
of the prior conversation. A standalone analytical question is NEVER clarification.

---

Dimension Signals:

- requires_audience = true ONLY if audience/segment-related concepts are explicitly mentioned
- requires_content = true ONLY if content/messaging-related concepts (incl.
  "theme", "topic", "category", "creative", "campaign type") are mentioned

IMPORTANT:

- Audience queries are VALID — do NOT mark them out_of_scope
- Content queries are VALID — do NOT mark them out_of_scope
- Mixed queries are VALID — do NOT mark them out_of_scope
- Campaign is always implicitly included
- Vague-but-analytical queries are VALID — do NOT mark them clarification

---

Examples:

Q: "Identify recurring good or bad performing themes for the past 6 months"
→ intent_type="performance_lookup", requires_content=true, time="past 6 months"
   (theme = content dimension; recurring patterns = campaign performance)

Q: "What's working and what's not?"
→ intent_type="performance_lookup" (vague but analytical, NOT clarification)

Q: "Anything interesting in last week's campaigns?"
→ intent_type="performance_lookup" (open-ended insight ask, NOT clarification)

Q: "Show me best and worst performing categories"
→ intent_type="performance_lookup", requires_content=true

Q: "Which segments converted best?"
→ intent_type="performance_lookup", requires_audience=true

Q: "Summarize what we just discussed"
→ intent_type="clarification" (refers to prior conversation)

Q: "What channels did you mention earlier?"
→ intent_type="clarification" (meta about prior turn)

Q: "hi"
→ intent_type="greeting"

Q: "What's the weather?"
→ intent_type="out_of_scope"

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
        raw = (query or "").strip()
        q = _normalize_query(raw)
        if q != raw:
            logger.info(f"[INTENT_CLASSIFIER] normalized query | raw={raw[:80]!r} → q={q[:80]!r}")
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
