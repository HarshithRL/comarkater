"""Genie data node — calls Databricks Genie API, builds tables, emits via StreamWriter.

.. deprecated::
    This node is part of the legacy simple path (genie_data → genie_analysis).
    New complex queries use the insight_agent subgraph instead.
    Kept for the simple data_query path and backward compatibility.

Split from the original monolith: this node handles ONLY the Genie API call
and table construction. LLM analysis is a separate node (genie_analysis.py)
so the table streams to the user immediately without waiting for LLM.

Flow:
  1. Genie API (submit → poll → fetch)
  2. Build 2D array from columns + data_array
  3. EMIT TABLE via StreamWriter (user sees it before analysis)
  4. Return state with raw data for genie_analysis_node
"""

import logging
import time
import uuid

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from core.config import settings
from core.state import AgentState
from tools.genie_client import genie_client, STATUS_COMPLETED, STATUS_FAILED, STATUS_FEEDBACK
from parsers.table_truncator import truncate_table_for_llm, typed_value, BUDGET_ANALYSIS

logger = logging.getLogger(__name__)


# ── Data conversion utilities ──

def _build_table_2d(columns: list[dict], data_array: list[list], max_rows: int = 50) -> dict:
    """Genie {columns, data_array} → {tableHeaders, data, alignment} for UI."""
    if not columns or not data_array:
        return {"tableHeaders": [], "data": [], "alignment": []}

    headers = [col.get("name", f"col_{i}") for i, col in enumerate(columns)]
    data = [[typed_value(v) for v in row] for row in data_array[:max_rows]]

    return {
        "tableHeaders": headers,
        "data": data,
        "alignment": ["left"] * len(headers),
    }


def _build_table_text(columns: list[dict], data_array: list[list], max_rows: int = 50) -> str:
    """Genie {columns, data_array} → readable text for LLM prompts."""
    if not columns or not data_array:
        return "No data returned."

    headers = [col.get("name", "unknown") for col in columns]
    lines = [" | ".join(headers)]
    for row in data_array[:max_rows]:
        lines.append(" | ".join(str(v) if v is not None else "NA" for v in row))

    return "\n".join(lines)


def _extract_feedback_text(status_data: dict) -> str:
    """Extract human-readable feedback text from a FEEDBACK_NEEDED response."""
    raw = status_data.get("raw", {})
    for att in raw.get("attachments", []):
        if "text" in att:
            content = att["text"].get("content", "")
            if content:
                return content
    return "I need more information to answer your question. Could you please specify the time period, metric, and channel you're interested in?"


def _error_response(message: str, request_id: str, prior_llm_calls: int) -> dict:
    """Build a standard error response dict."""
    logger.warning("GENIE_DATA._error: request_id=%s | message=%s", request_id, message[:200])
    return {
        "response_text": f"I encountered an issue while analyzing your data.\n\n{message}",
        "response_items": [{"type": "text", "id": str(uuid.uuid4()), "value": message}],
        "genie_summary": "",
        "genie_sql": "",
        "genie_table": "",
        "genie_tables": [],
        "genie_columns": [],
        "genie_data_array": [],
        "genie_text_content": "",
        "genie_insights": "",
        "genie_query_description": "",
        "follow_up_suggestions": [],
        "error": message,
        "llm_call_count": prior_llm_calls,
    }


# ── Graph node ──

def genie_data_node(state: AgentState, config: RunnableConfig) -> dict:
    """Genie API call + table build + StreamWriter emit. No LLM.

    Emits table mid-node via StreamWriter so the user sees data before
    genie_analysis_node runs. Passes raw columns/data_array in state
    for the analysis node to use.
    """
    request_id = state.get("request_id", "unknown")
    question = state.get("rewritten_question") or state.get("original_question", "")
    token = config["configurable"]["sp_token"]
    prior_llm_calls = state.get("llm_call_count", 0)
    space_id = settings.GENIE_SPACE_ID
    writer = get_stream_writer()

    logger.info(
        "GENIE_DATA.node: START | request_id=%s | question=%s | space=%s",
        request_id, question[:120], space_id,
    )

    if not token:
        return _error_response("Authentication error: no SP token available.", request_id, prior_llm_calls)

    if not space_id:
        return _error_response("Genie space not configured.", request_id, prior_llm_calls)

    # ── Genie API (submit → poll → fetch) ──
    conv_id, msg_id = "", ""
    sql_query, description, text_content = "", "", ""
    columns, data_array = [], []
    suggested_questions = []

    try:
        with mlflow.start_span(name="genie_call") as span:
            span.set_inputs({"query_sent_to_genie": question, "space_id": space_id, "request_id": request_id})

            # Submit
            try:
                conv_id, msg_id = genie_client.start_conversation(question, space_id, token)
            except Exception as e:
                logger.error("GENIE_DATA.node: submit failed | request_id=%s | error=%s", request_id, e, exc_info=True)
                return _error_response(f"Genie connection failed: {str(e)[:150]}", request_id, prior_llm_calls)

            # Poll with exponential backoff
            start_time = time.time()
            wait = 2
            consecutive_errors = 0
            status_data = None
            prev_status = None

            while True:
                elapsed = time.time() - start_time
                if elapsed > settings.GENIE_POLL_TIMEOUT:
                    logger.warning("GENIE_DATA.node: TIMEOUT after %.0fs | request_id=%s", elapsed, request_id)
                    span.set_outputs({"status": "TIMEOUT", "elapsed": elapsed})
                    return _error_response(
                        f"Query timed out after {settings.GENIE_POLL_TIMEOUT}s. Try a simpler question.",
                        request_id, prior_llm_calls,
                    )

                try:
                    status_data = genie_client.check_status(space_id, conv_id, msg_id, token)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning("GENIE_DATA.node: poll error #%d | request_id=%s | error=%s", consecutive_errors, request_id, e)
                    if consecutive_errors >= 3:
                        return _error_response("Genie polling failed after 3 retries.", request_id, prior_llm_calls)
                    time.sleep(wait)
                    wait = min(wait * 2, 8)
                    continue

                status = status_data["status"]

                # Emit poll status via StreamWriter (transient UI indicator)
                if status != prev_status:
                    try:
                        writer({"event_type": "genie_status", "status": status, "elapsed": round(elapsed, 1)})
                    except Exception:
                        pass
                    prev_status = status

                if status == STATUS_COMPLETED:
                    break
                elif status == STATUS_FAILED:
                    error_msg = "Genie query failed."
                    raw = status_data.get("raw", {})
                    err = raw.get("error", {})
                    if isinstance(err, dict) and err.get("message"):
                        error_msg = err["message"]
                    logger.error("GENIE_DATA.node: FAILED | error=%s | request_id=%s", error_msg, request_id)
                    span.set_outputs({"status": "FAILED", "error": error_msg[:500]})
                    return _error_response(error_msg, request_id, prior_llm_calls)
                elif status == STATUS_FEEDBACK:
                    feedback = _extract_feedback_text(status_data)
                    logger.info("GENIE_DATA.node: FEEDBACK_NEEDED | request_id=%s", request_id)
                    span.set_outputs({"status": "feedback_needed", "feedback": feedback[:500]})
                    return {
                        "response_text": feedback,
                        "response_items": [{"type": "text", "id": str(uuid.uuid4()), "value": feedback}],
                        "genie_summary": feedback,
                        "genie_sql": "",
                        "genie_table": "",
                        "genie_tables": [],
                        "genie_columns": [],
                        "genie_data_array": [],
                        "genie_text_content": "",
                        "genie_insights": "",
                        "genie_query_description": "",
                        "follow_up_suggestions": [],
                        "genie_trace_id": f"{conv_id}/{msg_id}",
                        "llm_call_count": prior_llm_calls,
                    }

                time.sleep(wait)
                wait = min(wait * 2, 8)

            # COMPLETED — extract attachments
            sql_query = status_data.get("sql") or ""
            raw = status_data.get("raw", {})

            for att in raw.get("attachments", []):
                if "text" in att:
                    text_content = att["text"].get("content", "")
            suggested_questions = raw.get("suggested_questions", [])

            # Fetch query result
            for att in raw.get("attachments", []):
                if "query" in att:
                    att_id = att.get("attachment_id") or att.get("id", "")
                    description = att["query"].get("description", "")
                    if att_id:
                        try:
                            result = genie_client.fetch_result(space_id, conv_id, msg_id, att_id, token)
                            data_array = result.get("statement_response", {}).get("result", {}).get("data_array", [])
                            columns = result.get("statement_response", {}).get("manifest", {}).get("schema", {}).get("columns", [])
                            logger.info("GENIE_DATA.node: data fetched | rows=%d | cols=%d | request_id=%s", len(data_array), len(columns), request_id)
                        except Exception as e:
                            logger.error("GENIE_DATA.node: fetch_result failed | request_id=%s | error=%s", request_id, e, exc_info=True)
                            return _error_response(f"Failed to fetch query results: {str(e)[:150]}", request_id, prior_llm_calls)
                    break

            span.set_attributes({
                "genie_status": STATUS_COMPLETED, "genie_conv_id": conv_id,
                "genie_msg_id": msg_id, "genie_space_id": space_id,
                "request_id": request_id, "row_count": len(data_array),
                "has_sql": bool(sql_query), "table_count": 1 if data_array else 0,
            })
            span.set_outputs({
                "sql_query": sql_query[:500], "data_rows": str(data_array[:3])[:2000],
                "row_count": len(data_array), "column_count": len(columns),
                "genie_trace_id": f"{conv_id}/{msg_id}",
            })

    except Exception as e:
        logger.error("GENIE_DATA.node: API error | request_id=%s | error=%s", request_id, e, exc_info=True)
        return _error_response(f"Genie query failed: {str(e)[:200]}", request_id, prior_llm_calls)

    # ── Build 2D array + emit table via StreamWriter ──
    table_2d = _build_table_2d(columns, data_array)
    table_text = truncate_table_for_llm(columns, data_array, char_budget=BUDGET_ANALYSIS)

    # Emit table mid-node — user sees it BEFORE analysis node runs
    if table_2d.get("data"):
        try:
            writer({"event_type": "table_ready", "table": table_2d, "row_count": len(data_array), "genie_trace_id": f"{conv_id}/{msg_id}"})
            logger.info("GENIE_DATA.node: table emitted via StreamWriter | rows=%d | request_id=%s", len(data_array), request_id)
        except Exception as e:
            logger.warning("GENIE_DATA.node: StreamWriter failed (non-fatal) | error=%s", e)

    genie_tables = [table_2d] if table_2d.get("data") else []

    logger.info(
        "GENIE_DATA.node: DONE | request_id=%s | rows=%d | cols=%d | sql=%s | suggestions=%d | trace=%s/%s",
        request_id, len(data_array), len(columns), bool(sql_query), len(suggested_questions), conv_id, msg_id,
    )

    return {
        "genie_sql": sql_query,
        "genie_table": table_text,
        "genie_tables": genie_tables,
        "genie_columns": columns,
        "genie_data_array": data_array,
        "genie_text_content": text_content,
        "genie_query_description": description,
        "follow_up_suggestions": suggested_questions,
        "genie_trace_id": f"{conv_id}/{msg_id}",
        "llm_call_count": prior_llm_calls,
    }
