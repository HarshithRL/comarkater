"""Genie analysis node — LLM analysis of Genie query results.

.. deprecated::
    This node is part of the legacy simple path (genie_data → genie_analysis).
    New complex queries use the insight_agent subgraph which combines data
    retrieval and analysis in a single ReAct loop.
    Kept for the simple data_query path and backward compatibility.

Runs AFTER genie_data_node. Reads raw columns, data_array, and SQL
from state. Produces executive summary + analytical text.

This is a separate node so the table streams to the user immediately
from genie_data_node, and the analysis arrives as a second event.
"""

import logging
import uuid

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from core.config import settings
from core.state import AgentState
from prompts.genie_agent_prompt import GENIE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


def _format_columns_description(columns: list[dict]) -> str:
    """Format column metadata for the analysis prompt."""
    if not columns:
        return "No column information available."
    parts = []
    for col in columns:
        name = col.get("name", "unknown")
        type_name = col.get("type_name") or col.get("type_text", "STRING")
        parts.append(f"- {name} ({type_name})")
    return "\n".join(parts)


def genie_analysis_node(state: AgentState, config: RunnableConfig) -> dict:
    """LLM analysis of Genie results. Reads data from state, returns summary."""
    request_id = state.get("request_id", "unknown")
    question = state.get("rewritten_question") or state.get("original_question", "")
    token = config["configurable"]["sp_token"]
    client_name = state.get("client_name", "there")
    prior_llm_calls = state.get("llm_call_count", 0)

    # Read Genie data from state (set by genie_data_node)
    columns = state.get("genie_columns") or []
    data_array = state.get("genie_data_array") or []
    sql_query = state.get("genie_sql", "")
    description = state.get("genie_query_description", "")
    table_text = state.get("genie_table", "")
    text_content = state.get("genie_text_content", "")

    logger.info(
        "GENIE_ANALYSIS.node: START | request_id=%s | rows=%d | cols=%d",
        request_id, len(data_array), len(columns),
    )

    # Skip analysis if no data was returned
    if not columns and not data_array:
        fallback = text_content or "No data available for analysis."
        logger.info("GENIE_ANALYSIS.node: no data, using fallback | request_id=%s", request_id)
        return {
            "genie_summary": fallback,
            "genie_insights": fallback,
            "response_text": fallback,
            "response_items": [{"type": "text", "id": str(uuid.uuid4()), "value": fallback}],
            "llm_call_count": prior_llm_calls,
        }

    summary = ""
    try:
        with mlflow.start_span(name="genie_analysis") as span:
            span.set_inputs({"question": question[:200], "sql": sql_query[:200], "row_count": len(data_array)})
            span.set_attributes({"model_used": settings.LLM_ENDPOINT_NAME, "request_id": request_id})

            llm = ChatOpenAI(
                model=settings.LLM_ENDPOINT_NAME,
                api_key=token,
                base_url=settings.AI_GATEWAY_URL,
                temperature=0.0,
            )

            columns_desc = _format_columns_description(columns)
            prompt_text = GENIE_ANALYSIS_PROMPT.format(
                original_question=question,
                genie_sql=sql_query or "N/A",
                genie_description=description or "N/A",
                genie_columns=columns_desc,
                genie_data=table_text,
                row_count=len(data_array),
                client_name=client_name,
            )

            # Emit status before LLM call
            try:
                from langgraph.config import get_stream_writer
                _writer = get_stream_writer()
                _writer({"event_type": "node_started", "node": "genie_analysis", "message": "Analyzing your data..."})
            except Exception:
                pass

            user_msg = question
            if text_content:
                user_msg += f"\n\n[Data summary for reference]: {text_content}"

            response = llm.invoke([
                SystemMessage(content=prompt_text),
                HumanMessage(content=user_msg),
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            summary = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip()

            span.set_outputs({"nl_analysis": summary[:2000]})
            logger.info("GENIE_ANALYSIS.node: done | summary_len=%d | request_id=%s", len(summary), request_id)

    except Exception as e:
        logger.error("GENIE_ANALYSIS.node: LLM failed | request_id=%s | error=%s", request_id, e, exc_info=True)
        if text_content:
            summary = text_content
        else:
            summary = f"Data retrieved for {client_name}: {len(data_array)} rows returned."
            if description:
                summary += f"\n\n{description}"

    if not summary:
        summary = "No analysis generated."

    response_text = f"**Analysis for {client_name}:**\n\n{summary}"

    response_items = []
    if summary:
        response_items.append({"type": "text", "id": str(uuid.uuid4()), "value": summary})

    logger.info("GENIE_ANALYSIS.node: DONE | request_id=%s | summary=%dch", request_id, len(summary))

    return {
        "genie_summary": summary[:2000],
        "genie_insights": summary[:2000],
        "response_text": response_text,
        "response_items": response_items,
        "llm_call_count": prior_llm_calls + 1,
    }
