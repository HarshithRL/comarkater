"""Follow-up question generation via LLM.

Generates 3 contextual follow-up questions based on the user's query
and the analysis summary. Falls back to static suggestions on failure.

Ported from legacy deployment code: lines 402-436.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_STATIC_FALLBACK = [
    "Which channel has the highest ROI?",
    "What optimization levers should I focus on?",
    "How does performance compare month over month?",
]


def generate_follow_ups(question: str, summary: str, llm) -> list[str]:
    """Generate 3 contextual follow-up questions using a fast LLM call.

    Args:
        question: The original user question.
        summary: The analysis summary (first 400 chars used).
        llm: LangChain-compatible LLM instance.

    Returns:
        List of 3 follow-up question strings. Falls back to static on failure.
    """
    try:
        prompt = (
            "You are a marketing analytics assistant.\n"
            "Given the user's question and the analysis summary, "
            "suggest exactly 3 short follow-up questions the user might ask next.\n\n"
            f"User question: {question}\n"
            f"Analysis summary (first 400 chars): {summary[:400]}\n\n"
            "Rules:\n"
            "- Each question must be under 12 words\n"
            "- Make them specific to campaign analytics (channels, metrics, optimization)\n"
            "- Return ONLY a JSON array of 3 strings — no preamble, no markdown fences\n"
            'Example: ["How does WhatsApp compare?", "Which day performs best?", "What drives revenue?"]'
        )
        response = llm.invoke(prompt)
        raw = (response.content or "").strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list) and len(suggestions) > 0:
            logger.info(f"FOLLOW_UPS: Generated {len(suggestions)} suggestions")
            return [str(s) for s in suggestions[:3]]
    except Exception as e:
        logger.warning(f"FOLLOW_UPS: Generation failed, using fallback | error={e}")

    return _STATIC_FALLBACK.copy()
