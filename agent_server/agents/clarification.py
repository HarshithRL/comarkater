"""Clarification agent — answers meta/follow-up questions from conversation context.

No Genie call. No charts. No recommendations. Just a concise text answer
using conversation history and LTM context. Fast and cheap (~2s, 1 LLM call).

Supervisor routes here when the question is about the conversation itself
(e.g., "what channels am I interested in?", "summarize what we discussed",
"can you clarify that?") rather than a new data query.
"""

import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import mlflow

from core.config import settings
from core.state import AgentState

logger = logging.getLogger(__name__)

CLARIFICATION_PROMPT = """You are CoMarketer, a campaign analytics assistant. The user is asking a follow-up or clarification question about the current conversation — NOT requesting new data.

Answer their question using ONLY the conversation history below. Be concise and direct.

{conversation_history}

Rules:
- Answer from conversation context only — do NOT make up data
- If you can infer the answer from prior exchanges (e.g., channels mentioned, metrics discussed), state it clearly
- If you genuinely cannot answer from context, say so and suggest what data query they could ask instead
- Keep it to 2-4 sentences max
- Do NOT generate tables, charts, SQL, or strategic recommendations

User question: {question}
Answer:"""


def clarification_node(state: AgentState, config: RunnableConfig) -> dict:
    """Answer meta/clarification questions from conversation context.

    No Genie API call. Returns plain text only.
    """
    request_id = state.get("request_id", "unknown")
    question = state.get("original_question", "")
    messages = state.get("messages", [])
    ltm_context = state.get("ltm_context", "") or ""

    logger.info(f"CLARIFICATION.node: start | request_id={request_id} | q={question[:80]} | history={len(messages)}")

    # Build conversation history for prompt
    history_lines = []
    for msg in messages[:-1]:  # Exclude current message
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content[:300] if hasattr(msg, "content") else str(msg)[:300]
        history_lines.append(f"{role}: {content}")

    conversation_history = "\n".join(history_lines) if history_lines else "(No prior conversation)"
    if ltm_context:
        conversation_history += f"\n\nLong-term memory context:\n{ltm_context[:500]}"

    sp_token = config["configurable"]["sp_token"]
    try:
        with mlflow.start_span(name="clarification_answer") as span:
            span.set_attributes({
                "request_id": request_id,
                "history_messages": len(messages),
                "question": question[:200],
            })

            llm = ChatOpenAI(
                model=settings.LLM_ENDPOINT_NAME,
                api_key=sp_token,
                base_url=settings.AI_GATEWAY_URL,
                temperature=0.0,
            )

            prompt = CLARIFICATION_PROMPT.format(
                conversation_history=conversation_history,
                question=question,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip()

            span.set_attributes({"answer_length": len(answer)})
            logger.info(f"CLARIFICATION.node: done | request_id={request_id} | answer_len={len(answer)}")

    except Exception as e:
        logger.error(f"CLARIFICATION.node: LLM failed | request_id={request_id} | error={e}")
        answer = "I couldn't process that follow-up. Could you rephrase, or ask a specific data question?"

    response_items = [
        {"type": "text", "id": str(uuid.uuid4()), "value": answer},
    ]

    return {
        "messages": [AIMessage(content=answer)],
        "response_text": answer,
        "response_items": response_items,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
    }
