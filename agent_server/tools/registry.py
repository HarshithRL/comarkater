"""Tool registry — defines tools ONCE, resolves by name for capability agents.

All Genie tools wrap the existing GenieClient. Adding a new tool means
adding ONE @tool function here and registering it in ALL_TOOLS.

Tools receive cid for RLS context but do NOT add WHERE cid clauses —
Unity Catalog enforces RLS at infrastructure level.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List

from langchain_core.tools import tool

from tools.genie_client import (
    genie_client,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_FEEDBACK,
)
from core.config import settings
from parsers.table_truncator import truncate_table_for_llm, BUDGET_ANALYSIS

logger = logging.getLogger(__name__)


def _run_genie_query(question: str, token: str) -> dict:
    """Shared Genie API flow: submit → poll → fetch. Returns raw result dict.

    Raises on failure so callers can handle errors with context.
    """
    space_id = settings.GENIE_SPACE_ID
    if not space_id:
        raise ValueError("GENIE_SPACE_ID not configured")

    conv_id, msg_id = genie_client.start_conversation(question, space_id, token)

    # Poll with exponential backoff
    start_time = time.time()
    wait = 2
    consecutive_errors = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > settings.GENIE_POLL_TIMEOUT:
            raise TimeoutError(
                f"Genie query timed out after {settings.GENIE_POLL_TIMEOUT}s"
            )

        try:
            status_data = genie_client.check_status(space_id, conv_id, msg_id, token)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise ConnectionError("Genie polling failed after 3 retries") from e
            time.sleep(wait)
            wait = min(wait * 2, 8)
            continue

        status = status_data["status"]

        if status == STATUS_COMPLETED:
            break
        elif status == STATUS_FAILED:
            raw = status_data.get("raw", {})
            err = raw.get("error", {})
            error_msg = err.get("message", "Genie query failed.") if isinstance(err, dict) else "Genie query failed."
            raise RuntimeError(error_msg)
        elif status == STATUS_FEEDBACK:
            # Extract feedback text
            for att in status_data.get("raw", {}).get("attachments", []):
                if "text" in att:
                    feedback = att["text"].get("content", "")
                    if feedback:
                        return {"type": "feedback", "text": feedback}
            return {"type": "feedback", "text": "More information needed."}

        time.sleep(wait)
        wait = min(wait * 2, 8)

    # COMPLETED — extract data
    raw = status_data.get("raw", {})
    sql_query = status_data.get("sql") or ""
    text_content = ""
    columns, data_array = [], []
    description = ""

    for att in raw.get("attachments", []):
        if "text" in att:
            text_content = att["text"].get("content", "")

    for att in raw.get("attachments", []):
        if "query" in att:
            att_id = att.get("attachment_id") or att.get("id", "")
            description = att["query"].get("description", "")
            if att_id:
                result = genie_client.fetch_result(space_id, conv_id, msg_id, att_id, token)
                data_array = result.get("statement_response", {}).get("result", {}).get("data_array", [])
                columns = result.get("statement_response", {}).get("manifest", {}).get("schema", {}).get("columns", [])
            break

    return {
        "type": "data",
        "sql": sql_query,
        "columns": columns,
        "data_array": data_array,
        "description": description,
        "text_content": text_content,
        "trace_id": f"{conv_id}/{msg_id}",
    }


# ── Genie token accessor ──
# Tools can't access RunnableConfig directly. The insight agent node
# injects the SP token into a module-level holder before invoking the
# ReAct agent. This is safe because LangGraph nodes run sequentially
# within a single graph invocation.
_current_sp_token: str = ""


def set_current_sp_token(token: str) -> None:
    """Set the SP token for the current graph invocation."""
    global _current_sp_token
    _current_sp_token = token


def get_current_sp_token() -> str:
    """Get the SP token for the current graph invocation."""
    if not _current_sp_token:
        raise RuntimeError("SP token not set — call set_current_sp_token() first")
    return _current_sp_token


# ── Tool definitions ──

@tool
def genie_search(query: str) -> str:
    """Search campaign data using natural language via the analytics engine.

    Use this for questions like:
    - "Show email open rates for last month"
    - "Top 5 campaigns by click rate in March 2026"
    - "WhatsApp delivery rate trends for IGP"

    The engine translates your question to SQL and returns data.
    Do NOT include client ID filters — row-level security is automatic.

    Args:
        query: Natural language question about campaign data.

    Returns:
        Formatted data table with column headers and rows, or feedback
        if the query needs clarification.
    """
    token = get_current_sp_token()
    logger.info("TOOL.genie_search: query='%s'", query[:120])

    try:
        result = _run_genie_query(query, token)
    except Exception as e:
        logger.error("TOOL.genie_search: failed | error=%s", e)
        return (
            f"genie_search FAILED: {str(e)[:300]}. "
            "Do NOT retry the identical query. Change your approach: "
            "rephrase the question, try different column names, or use genie_query with explicit SQL."
        )

    if result["type"] == "feedback":
        return f"Clarification needed: {result['text']}"

    columns = result["columns"]
    data_array = result["data_array"]

    if not columns or not data_array:
        text = result.get("text_content", "")
        if text:
            return text
        return (
            f"genie_search returned NO ROWS for query: '{query[:200]}'. "
            "Do NOT retry this exact query. Possible causes: "
            "(1) filter too restrictive — try a broader date range or remove one filter; "
            "(2) value does not exist — verify with a simpler probe query first; "
            "(3) wrong column reference — use a discovery query like 'what channels exist in the data'. "
            "Change your approach before the next tool call."
        )

    # Format as readable table for the LLM
    table_text = truncate_table_for_llm(
        columns, data_array, char_budget=BUDGET_ANALYSIS
    )
    desc = result.get("description", "")
    row_count = len(data_array)

    output = f"Query: {query}\nRows: {row_count}\n"
    if desc:
        output += f"Description: {desc}\n"
    output += f"\n{table_text}"

    return output


@tool
def genie_query(sql: str) -> str:
    """Execute a specific SQL-style query via the analytics engine.

    Use this when you need precise control over the query, such as:
    - Specific aggregations or filters
    - Queries with array_contains for tagname filtering
    - Date range filtering with specific dates

    Schema guidance:
    - Table: campaign_details (single source, RLS enforced)
    - Use array_contains(tagname, 'value') for tag filtering
    - Do NOT add WHERE cid = ... (RLS handles this)
    - Channels: Email, SMS, APN, BPN, WhatsApp
    - Date column: send_date (DATE type)

    Args:
        sql: SQL query or natural language description of the SQL you need.

    Returns:
        Formatted data table with column headers and rows.
    """
    token = get_current_sp_token()
    logger.info("TOOL.genie_query: sql='%s'", sql[:120])

    try:
        result = _run_genie_query(sql, token)
    except Exception as e:
        logger.error("TOOL.genie_query: failed | error=%s", e)
        return (
            f"genie_query FAILED: {str(e)[:300]}. "
            "Do NOT retry the identical SQL. Change your approach: "
            "verify column/table names with genie_search, simplify the WHERE clause, "
            "or try a smaller probe query first."
        )

    if result["type"] == "feedback":
        return f"Clarification needed: {result['text']}"

    columns = result["columns"]
    data_array = result["data_array"]

    if not columns or not data_array:
        text = result.get("text_content", "")
        if text:
            return text
        return (
            f"genie_query returned NO ROWS for SQL: '{sql[:200]}'. "
            "Do NOT retry this exact SQL. Possible causes: "
            "(1) filter too restrictive — broaden the date range or remove one condition; "
            "(2) wrong column value — e.g. the wave/channel name may differ from what you used; "
            "(3) array_contains needs exact tag match. "
            "Try a discovery query (SELECT DISTINCT wave FROM campaign_details LIMIT 20) before retrying."
        )

    table_text = truncate_table_for_llm(
        columns, data_array, char_budget=BUDGET_ANALYSIS
    )

    return f"SQL result ({len(data_array)} rows):\n{table_text}"


# ── Registry ──

ALL_TOOLS = {
    "genie_search": genie_search,
    "genie_query": genie_query,
}


def get_tools(tool_names: List[str]) -> List[Callable]:
    """Resolve tool names to tool instances.

    Args:
        tool_names: List of tool names to resolve (must exist in ALL_TOOLS).

    Returns:
        List of tool instances ready for use with create_react_agent.

    Raises:
        ValueError: If a tool name is not found in the registry.
    """
    tools = []
    for name in tool_names:
        if name not in ALL_TOOLS:
            available = ", ".join(ALL_TOOLS.keys())
            raise ValueError(
                f"Tool '{name}' not found in registry. Available: {available}"
            )
        tools.append(ALL_TOOLS[name])
    return tools


