"""Genie worker node — Send-spawned parallel worker for complex queries.

.. deprecated::
    The parallel worker pattern is superseded by the insight_agent subgraph
    which executes steps sequentially with a ReAct loop. This node is kept
    in the graph for backward compatibility and direct testing, but is no
    longer wired into the default complex query path.

Each worker executes a single Genie sub-question independently.
Spawned via Send API from the planner fan-out edge.

Reuses genie_client (API calls) and genie_agent utilities (table building).
Returns result in worker_results list for reducer accumulation.

NOTE: Worker receives a CUSTOM state dict from Send, NOT AgentState.
"""

import logging
import time

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from core.config import settings
from tools.genie_client import genie_client, STATUS_COMPLETED, STATUS_FAILED, STATUS_FEEDBACK
from agents.genie_agent import _build_table_2d
from parsers.table_truncator import truncate_table_for_llm, BUDGET_WORKER

logger = logging.getLogger(__name__)


def _worker_error(sub_question: str, purpose: str, message: str, worker_index: int) -> dict:
    """Build a worker error result for accumulation."""
    logger.warning("GENIE_WORKER[%d]: error | message=%s", worker_index, message[:200])
    return {
        "worker_results": [{
            "sub_question": sub_question,
            "purpose": purpose,
            "worker_index": worker_index,
            "genie_sql": "",
            "genie_table": "",
            "genie_tables": [],
            "genie_columns": [],
            "genie_data_array": [],
            "genie_text_content": "",
            "genie_query_description": "",
            "genie_trace_id": "",
            "error": message,
        }],
    }


def genie_worker_node(state: dict, config: RunnableConfig) -> dict:
    """Execute a single Genie sub-question. Spawned via Send API.

    state is a custom dict from Send:
      {sub_question, purpose, worker_index, request_id}

    Returns worker_results (list with one dict) for operator.add reducer.
    Emits table via StreamWriter mid-node.
    """
    sub_question = state.get("sub_question", "")
    purpose = state.get("purpose", "")
    worker_index = state.get("worker_index", 0)
    request_id = state.get("request_id", "unknown")

    token = config["configurable"]["sp_token"]
    space_id = settings.GENIE_SPACE_ID
    writer = get_stream_writer()

    logger.info(
        "GENIE_WORKER[%d]: START | request_id=%s | q=%s",
        worker_index, request_id, sub_question[:120],
    )

    if not token:
        return _worker_error(sub_question, purpose, "No SP token available.", worker_index)
    if not space_id:
        return _worker_error(sub_question, purpose, "Genie space not configured.", worker_index)

    # Stagger submissions to avoid Genie API 429 rate limiting.
    # Workers 0-4 go immediately (within Genie's burst tolerance),
    # workers 5+ wait 1s per index beyond 4 to stay under the rate limit.
    if worker_index >= 5:
        stagger_delay = (worker_index - 4) * 1.0
        logger.info("GENIE_WORKER[%d]: stagger delay %.1fs | request_id=%s", worker_index, stagger_delay, request_id)
        time.sleep(stagger_delay)

    # ── Genie API (submit → poll → fetch) — same pattern as genie_data_node ──
    conv_id, msg_id = "", ""
    sql_query, description, text_content = "", "", ""
    columns, data_array = [], []

    try:
        with mlflow.start_span(name=f"genie_worker_{worker_index}") as span:
            span.set_inputs({
                "sub_question": sub_question,
                "purpose": purpose,
                "worker_index": worker_index,
                "space_id": space_id,
                "request_id": request_id,
            })

            # Submit (retry once on 429 rate limit)
            for attempt in range(2):
                try:
                    conv_id, msg_id = genie_client.start_conversation(sub_question, space_id, token)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt == 0:
                        retry_wait = 5 + worker_index * 2  # Stagger retries by worker index
                        logger.warning("GENIE_WORKER[%d]: 429 rate limit, retrying in %ds", worker_index, retry_wait)
                        time.sleep(retry_wait)
                        continue
                    logger.error("GENIE_WORKER[%d]: submit failed | error=%s", worker_index, e)
                    return _worker_error(sub_question, purpose, f"Genie connection failed: {str(e)[:150]}", worker_index)

            # Poll with exponential backoff
            start_time = time.time()
            wait = 2
            consecutive_errors = 0
            prev_status = None

            while True:
                elapsed = time.time() - start_time
                if elapsed > settings.GENIE_POLL_TIMEOUT:
                    logger.warning("GENIE_WORKER[%d]: TIMEOUT | request_id=%s", worker_index, request_id)
                    return _worker_error(sub_question, purpose, "Query timed out.", worker_index)

                try:
                    status_data = genie_client.check_status(space_id, conv_id, msg_id, token)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        return _worker_error(sub_question, purpose, "Polling failed after 3 retries.", worker_index)
                    time.sleep(wait)
                    wait = min(wait * 2, 8)
                    continue

                status = status_data["status"]

                # Emit poll status via StreamWriter (transient UI indicator)
                if status != prev_status:
                    try:
                        writer({"event_type": "genie_status", "status": status, "elapsed": round(elapsed, 1), "worker_index": worker_index})
                    except Exception:
                        pass
                    prev_status = status

                if status == STATUS_COMPLETED:
                    break
                elif status == STATUS_FAILED:
                    error_msg = status_data.get("raw", {}).get("error", {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", "Genie query failed.")
                    return _worker_error(sub_question, purpose, str(error_msg), worker_index)
                elif status == STATUS_FEEDBACK:
                    # Genie needs clarification — treat as partial result
                    for att in status_data.get("raw", {}).get("attachments", []):
                        if "text" in att:
                            text_content = att["text"].get("content", "")
                    return _worker_error(sub_question, purpose, text_content or "Genie needs more info.", worker_index)

                time.sleep(wait)
                wait = min(wait * 2, 8)

            # COMPLETED — extract attachments
            sql_query = status_data.get("sql") or ""
            raw = status_data.get("raw", {})

            for att in raw.get("attachments", []):
                if "text" in att:
                    text_content = att["text"].get("content", "")

            for att in raw.get("attachments", []):
                if "query" in att:
                    att_id = att.get("attachment_id") or att.get("id", "")
                    description = att["query"].get("description", "")
                    if att_id:
                        try:
                            result = genie_client.fetch_result(space_id, conv_id, msg_id, att_id, token)
                            data_array = result.get("statement_response", {}).get("result", {}).get("data_array", [])
                            columns = result.get("statement_response", {}).get("manifest", {}).get("schema", {}).get("columns", [])
                        except Exception as e:
                            logger.error("GENIE_WORKER[%d]: fetch failed | error=%s", worker_index, e)
                            return _worker_error(sub_question, purpose, f"Fetch failed: {str(e)[:150]}", worker_index)
                    break

            span.set_outputs({
                "sql_query": sql_query[:500],
                "row_count": len(data_array),
                "column_count": len(columns),
                "genie_trace_id": f"{conv_id}/{msg_id}",
            })

    except Exception as e:
        logger.error("GENIE_WORKER[%d]: API error | error=%s", worker_index, e, exc_info=True)
        return _worker_error(sub_question, purpose, f"Genie query failed: {str(e)[:200]}", worker_index)

    # ── Build table + emit via StreamWriter ──
    table_2d = _build_table_2d(columns, data_array)
    table_text = truncate_table_for_llm(columns, data_array, char_budget=BUDGET_WORKER)

    if table_2d.get("data"):
        try:
            writer({
                "event_type": "worker_table_ready",
                "worker_index": worker_index,
                "sub_question": sub_question,
                "table": table_2d,
                "row_count": len(data_array),
                "genie_trace_id": f"{conv_id}/{msg_id}",
            })
            logger.info("GENIE_WORKER[%d]: table emitted | rows=%d", worker_index, len(data_array))
        except Exception as e:
            logger.warning("GENIE_WORKER[%d]: StreamWriter failed (non-fatal) | error=%s", worker_index, e)

    genie_tables = [table_2d] if table_2d.get("data") else []

    logger.info(
        "GENIE_WORKER[%d]: DONE | rows=%d | cols=%d | request_id=%s",
        worker_index, len(data_array), len(columns), request_id,
    )

    return {
        "worker_results": [{
            "sub_question": sub_question,
            "purpose": purpose,
            "worker_index": worker_index,
            "genie_sql": sql_query,
            "genie_table": table_text,
            "genie_tables": genie_tables,
            "genie_columns": columns,
            "genie_data_array": data_array,
            "genie_text_content": text_content,
            "genie_query_description": description,
            "genie_trace_id": f"{conv_id}/{msg_id}",
            "error": None,
        }],
    }
