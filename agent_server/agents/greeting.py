"""Greeting agent — handles greeting/thanks/bye intents.

No LLM calls. Returns canned responses. <0.1s latency. Zero cost.
Supervisor routes here when regex matches a greeting pattern.
"""

import logging
import uuid

from langchain_core.messages import AIMessage

from core.state import AgentState

logger = logging.getLogger(__name__)


def greeting_node(state: AgentState) -> dict:
    """Handle greeting intents with a canned response.

    Returns:
        Partial state update dict.
    """
    client_name = state.get("client_name", "there")
    request_id = state.get("request_id", "unknown")
    question = state.get("original_question", "")

    logger.info(f"GREETING.node: start | client={client_name} | request_id={request_id} | q={question[:80]}")

    response_text = (
        f"Hello {client_name}! I'm CoMarketer, your campaign analytics assistant.\n\n"
        "I can help you with:\n"
        "- Campaign performance analysis across Email, WhatsApp, SMS, APN, BPN\n"
        "- Optimization recommendations with trade-off analysis\n"
        "- Cross-channel data queries and comparisons\n\n"
        "Just ask me a question about your campaign data!"
    )

    response_items = [
        {"type": "text", "id": str(uuid.uuid4()), "value": response_text},
    ]

    logger.info(f"GREETING.node: done | response_len={len(response_text)} | items={len(response_items)} | request_id={request_id}")

    return {
        "messages": [AIMessage(content=response_text)],
        "response_text": response_text,
        "response_items": response_items,
        "llm_call_count": state.get("llm_call_count", 0),
    }
