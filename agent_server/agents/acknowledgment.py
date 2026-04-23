"""Acknowledgment system — generates a short "working on it" message.

Yields an acknowledgment event before the graph executes for analytics queries.
Skipped entirely for greetings (detected via _is_greeting).

Ported from legacy: lines 2087-2126.
"""

import logging
import uuid

from supervisor.intent_classifier import _is_greeting
from prompts.acknowledgment_prompt import ACKNOWLEDGMENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Fallback when LLM fails
_FALLBACK_ACK = "Looking into that for you — preparing the analysis now."


def create_acknowledgment(user_query: str, custom_inputs, llm) -> dict | None:
    """Generate an LLM-based acknowledgment message before graph execution.

    Args:
        user_query: The raw user question.
        custom_inputs: CustomInputs instance with user context.
        llm: LangChain-compatible LLM (ChatOpenAI or ChatDatabricks).

    Returns:
        Structured dict with items + custom_outputs, or None for greetings.
    """
    # Skip acknowledgment for greetings
    if _is_greeting(user_query):
        return None

    ack_text = _FALLBACK_ACK
    try:
        prompt = ACKNOWLEDGMENT_PROMPT_TEMPLATE.format(user_query=user_query)
        result = llm.invoke(prompt)
        raw_text = (result.content or "").strip()

        if raw_text:
            # Truncate to 3 sentences max
            sentences = raw_text.split('.')
            if len(sentences) > 3:
                raw_text = '.'.join(sentences[:3]) + '.'
            ack_text = raw_text

        logger.info(f"ACK: Generated | len={len(ack_text)}")
    except Exception as e:
        logger.warning(f"ACK: LLM failed, using fallback | error={e}")

    return {
        "items": [{
            "type": "text",
            "id": str(uuid.uuid4()),
            "value": ack_text,
        }],
        "custom_outputs": {
            "user_name": getattr(custom_inputs, "user_name", "User"),
            "user_id": getattr(custom_inputs, "user_id", ""),
            "thread_id": getattr(custom_inputs, "thread_id", ""),
            "conversation_id": getattr(custom_inputs, "conversation_id", ""),
            "task_type": getattr(custom_inputs, "task_type", "general"),
            "agent_id": "ACKNOWLEDGMENT",
            "type": "observation",
        },
    }
