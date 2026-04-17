"""AgentBricks sub-agent — calls the AgentBricks endpoint for data queries.

Sends the rewritten question to the AgentBricks serving endpoint,
parses the rich response to extract structured data:
  - Executive summary (markdown analysis)
  - Pipe table (data table)
  - SQL query
  - Campaign insights/recommendations
  - Follow-up suggestions
  - Query description
"""

import json
import logging
import re
import uuid

import mlflow
from langchain_core.runnables import RunnableConfig
from openai import OpenAI
from core.config import settings
from core.state import AgentState

logger = logging.getLogger(__name__)

# Pattern to detect routing tags like <name>agent-campaign-analysis-v2</name>
_NAME_TAG_RE = re.compile(r"^\s*<name>.*</name>\s*$", re.DOTALL)


def _json_decode_attr(raw: str) -> str:
    """Decode a span attribute that may be JSON-encoded (e.g. '"value"')."""
    if raw and isinstance(raw, str) and raw.startswith('"'):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return raw


def _get_text(item) -> str:
    """Safely extract text from a message output item."""
    content = getattr(item, "content", None) or []
    if isinstance(content, list) and content:
        block = content[0]
        return (getattr(block, "text", "") or "").strip()
    return ""


def _is_pipe_table(text: str) -> bool:
    """Check if text contains a markdown pipe table."""
    lines = text.strip().split("\n")
    pipe_count = sum(1 for l in lines if l.strip().startswith("|") and l.strip().endswith("|"))
    return pipe_count >= 2


def _parse_ab_response(response) -> dict:
    """Parse AgentBricks response to extract all structured data.

    AgentBricks returns 8 output items in a consistent pattern:
      [0] message: acknowledgment
      [1] function_call: get_campaign_insights
      [2] function_call: agent-campaign-analysis-v2 (genie query)
      [3] function_call_output: insights JSON
      [4] message: <name>...</name> routing tag
      [5] message: pipe table with data
      [6] message: <name>...</name> routing tag
      [7] message: executive summary (longest text)

    Plus trace spans with sql_query, query_description, follow-ups.

    Returns:
        dict with keys: executive_summary, table_markdown, sql_query,
        insights_raw, query_description, follow_ups, acknowledgment
    """
    result = {
        "executive_summary": "",
        "table_markdown": "",     # backward-compat: first table or labeled join
        "table_markdowns": [],    # all tables as separate strings
        "sql_query": "",
        "insights_raw": "",
        "query_description": "",
        "follow_ups": [],
        "acknowledgment": "",
    }

    # ── Step 1: Extract from output items ──
    try:
        output_items = getattr(response, "output", None) or []
        logger.info(f"AGENTBRICKS.parse: total output_items={len(output_items)}")
        messages = []  # collect non-tag message texts

        for idx, item in enumerate(output_items):
            item_type = getattr(item, "type", None)
            logger.debug(f"AGENTBRICKS.parse: item[{idx}] type={item_type}")

            # function_call_output — insights JSON
            if item_type == "function_call_output":
                name = getattr(item, "name", "") or ""
                raw_output = getattr(item, "output", "") or ""
                logger.info(f"AGENTBRICKS.parse: item[{idx}] function_call_output name={name} | output_len={len(raw_output)}")
                if "insights" in name.lower() and raw_output:
                    result["insights_raw"] = raw_output
                    logger.info(f"AGENTBRICKS.parse: item[{idx}] captured insights_raw ({len(raw_output)}ch)")
                continue

            # function_call items — log tool name
            if item_type == "function_call":
                fc_name = getattr(item, "name", "") or ""
                logger.info(f"AGENTBRICKS.parse: item[{idx}] function_call name={fc_name}")
                continue

            # message items — classify by content
            if item_type == "message":
                text = _get_text(item)
                if not text:
                    logger.debug(f"AGENTBRICKS.parse: item[{idx}] message empty, skipping")
                    continue
                # Skip routing tags like <name>genie_agent</name>
                if _NAME_TAG_RE.match(text):
                    logger.debug(f"AGENTBRICKS.parse: item[{idx}] message is routing tag, skipping | text={text[:80]}")
                    continue
                logger.info(f"AGENTBRICKS.parse: item[{idx}] message collected | len={len(text)} | preview={text[:120]}")
                messages.append(text)

        logger.info(f"AGENTBRICKS.parse: collected {len(messages)} usable messages from output_items")

        # Classify collected messages:
        # - pipe table: text with |...|..| lines (may appear multiple times)
        # - executive summary: longest non-table text
        # - acknowledgment: first short message
        for i, text in enumerate(messages):
            if _is_pipe_table(text):
                result["table_markdowns"].append(text)   # collect ALL tables
                result["table_markdown"] = text          # keep for backward compat (last wins)
                logger.info(f"AGENTBRICKS.parse: msg[{i}] classified as TABLE ({len(text)}ch) | total_tables={len(result['table_markdowns'])}")
            elif len(text) > len(result["executive_summary"]):
                # If we already had a shorter text as summary, that was the acknowledgment
                if result["executive_summary"] and not result["acknowledgment"]:
                    result["acknowledgment"] = result["executive_summary"]
                result["executive_summary"] = text
                logger.info(f"AGENTBRICKS.parse: msg[{i}] classified as EXECUTIVE_SUMMARY ({len(text)}ch)")
            elif not result["acknowledgment"]:
                result["acknowledgment"] = text
                logger.info(f"AGENTBRICKS.parse: msg[{i}] classified as ACKNOWLEDGMENT ({len(text)}ch)")

        # If multiple tables found, build a labeled join for format_supervisor (LLM prompt)
        if len(result["table_markdowns"]) > 1:
            result["table_markdown"] = "\n\n".join(
                f"**Table {j + 1}:**\n{tbl}"
                for j, tbl in enumerate(result["table_markdowns"])
            )
            logger.info(f"AGENTBRICKS.parse: {len(result['table_markdowns'])} tables joined for format_supervisor")

    except Exception as e:
        logger.warning(f"AGENTBRICKS.parse: Output extraction failed: {e}", exc_info=True)

    # ── Step 2: Extract from trace spans ──
    try:
        db_output = getattr(response, "databricks_output", None)
        logger.info(f"AGENTBRICKS.parse: databricks_output present={db_output is not None}")
        if db_output and isinstance(db_output, dict):
            trace_data = db_output.get("trace", {}).get("data", {})
            spans = trace_data.get("spans", [])
            logger.info(f"AGENTBRICKS.parse: trace spans count={len(spans or [])}")

            for span_idx, span in enumerate(spans or []):
                if not isinstance(span, dict):
                    continue
                attrs = span.get("attributes", {})
                span_name = span.get("name", "")
                span_type_raw = _json_decode_attr(attrs.get("mlflow.spanType", ""))
                logger.debug(f"AGENTBRICKS.parse: span[{span_idx}] name={span_name} | type={span_type_raw}")

                # SQL query — from the genie/analysis tool span
                if not result["sql_query"]:
                    sql_raw = attrs.get("sql_query", "")
                    if sql_raw:
                        result["sql_query"] = _json_decode_attr(str(sql_raw))
                        logger.info(f"AGENTBRICKS.parse: span[{span_idx}] extracted SQL query ({len(result['sql_query'])}ch) from span={span_name}")

                # Query description — from the genie/analysis tool span
                if not result["query_description"]:
                    desc_raw = attrs.get("description", "")
                    if desc_raw:
                        result["query_description"] = _json_decode_attr(str(desc_raw))
                        logger.info(f"AGENTBRICKS.parse: span[{span_idx}] extracted query_description from span={span_name}")

                # Follow-up suggestions — from ask_question span outputs
                if not result["follow_ups"] and span_name == "ask_question":
                    outputs_raw = attrs.get("mlflow.spanOutputs", "")
                    if outputs_raw:
                        try:
                            parsed = json.loads(outputs_raw) if isinstance(outputs_raw, str) else outputs_raw
                            if isinstance(parsed, dict):
                                suggestions = parsed.get("suggested_questions", [])
                                if suggestions:
                                    result["follow_ups"] = suggestions
                                    logger.info(f"AGENTBRICKS.parse: span[{span_idx}] extracted {len(suggestions)} follow_ups from ask_question")
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"AGENTBRICKS.parse: span[{span_idx}] failed to parse ask_question outputs")

                # Fallback: executive summary from root AGENT span outputs
                if not result["executive_summary"]:
                    if span_type_raw == "AGENT":
                        outputs_raw = attrs.get("mlflow.spanOutputs", "")
                        if outputs_raw:
                            try:
                                parsed = json.loads(outputs_raw) if isinstance(outputs_raw, str) else outputs_raw
                                if isinstance(parsed, dict):
                                    final = parsed.get("final_response", "")
                                    if final:
                                        result["executive_summary"] = final
                                        logger.info(f"AGENTBRICKS.parse: span[{span_idx}] fallback executive_summary from AGENT span ({len(final)}ch)")
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(f"AGENTBRICKS.parse: span[{span_idx}] failed to parse AGENT span outputs")

    except Exception as e:
        logger.warning(f"AGENTBRICKS.parse: Span extraction failed: {e}", exc_info=True)

    logger.info(
        f"AGENTBRICKS.parse: summary={len(result['executive_summary'])}ch "
        f"| tables={len(result['table_markdowns'])} "
        f"| table_total_len={len(result['table_markdown'])}ch "
        f"| sql={bool(result['sql_query'])} "
        f"| insights={bool(result['insights_raw'])} "
        f"| follow_ups={len(result['follow_ups'])}"
    )
    return result



def agentbricks_node(state: AgentState, config: RunnableConfig) -> dict:
    """Call AgentBricks endpoint with the rewritten question.

    Reads SP token from config["configurable"]["sp_token"], NOT from state.

    Returns:
        Partial state update with structured response data.
    """
    request_id = state.get("request_id", "unknown")
    question = state.get("rewritten_question") or state.get("original_question", "")
    token = config["configurable"]["sp_token"]
    client_name = state.get("client_name", "there")
    prior_llm_calls = state.get("llm_call_count", 0)

    logger.info(
        f"AGENTBRICKS.node: START | request_id={request_id} | "
        f"question={question[:150]} | client_name={client_name} | "
        f"has_token={bool(token)} | prior_llm_calls={prior_llm_calls}"
    )

    if not token:
        logger.error(f"AGENTBRICKS.node: No SP token — aborting | request_id={request_id}")
        return _error_response("Authentication error: no SP token available.", request_id, prior_llm_calls)

    try:
        with mlflow.start_span(name="agentbricks_call") as span:
            ab_client = OpenAI(
                api_key=token,
                base_url=f"{settings.DATABRICKS_HOST}/serving-endpoints",
            )

            conversation_id = state.get("conversation_id", "") or state.get("request_id", "")
            sp_id = state.get("sp_id", "")
            client_id = state.get("client_id", "")
            user_id = state.get("user_id", "")

            span.set_inputs({
                "query_sent_to_agentbricks": question,
                "conversation_id": conversation_id,
                "client_id": client_id,
                "sp_id": sp_id,
            })

            request_payload = {
                "model": settings.AGENTBRICKS_ENDPOINT_NAME,
                "input": [{"role": "user", "content": question}],
                "databricks_options": {
                    "conversation_id": conversation_id,
                    "return_trace": True,
                    "long_task": True,
                },
                "context": {
                    "sp_id": sp_id,
                    "client_id": client_id,
                    "client_name": client_name,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                },
            }
            logger.info(
                f"AGENTBRICKS.node: REQUEST payload | request_id={request_id} | "
                f"endpoint={settings.AGENTBRICKS_ENDPOINT_NAME} | "
                f"base_url={settings.DATABRICKS_HOST}/serving-endpoints | "
                f"sp_id={sp_id} | client_id={client_id} | conversation_id={conversation_id} | "
                f"user_id={user_id} | question_len={len(question)}"
            )
            logger.debug(f"AGENTBRICKS.node: full request_payload | request_id={request_id} | payload={json.dumps(request_payload, default=str)}")

            response = ab_client.responses.create(
                model=settings.AGENTBRICKS_ENDPOINT_NAME,
                input=[{"role": "user", "content": question}],
                extra_body={
                    "databricks_options": {
                        "conversation_id": conversation_id,
                        "return_trace": True,
                        "long_task": True,
                    },
                    "context": {
                        "sp_id": sp_id,
                        "client_id": client_id,
                        "client_name": client_name,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "request_id": request_id,
                    },
                },
            )

            span.set_attributes({
                "endpoint": settings.AGENTBRICKS_ENDPOINT_NAME,
                "question": question[:200],
                "request_id": request_id,
                "sp_id": sp_id,
                "client_id": client_id,
                "client_name": client_name,
                "conversation_id": conversation_id,
                "user_id": user_id,
            })

            # Log raw response metadata
            resp_id = getattr(response, "id", "N/A")
            resp_output_count = len(getattr(response, "output", []) or [])
            resp_has_db_output = getattr(response, "databricks_output", None) is not None
            logger.info(
                f"AGENTBRICKS.node: RAW RESPONSE received | request_id={request_id} | "
                f"response_id={resp_id} | output_items={resp_output_count} | "
                f"has_databricks_output={resp_has_db_output}"
            )

            # Log each raw output item type for traceability
            for ri, r_item in enumerate(getattr(response, "output", []) or []):
                r_type = getattr(r_item, "type", "unknown")
                r_text_preview = ""
                if r_type == "message":
                    r_text_preview = _get_text(r_item)[:100]
                elif r_type == "function_call_output":
                    r_text_preview = (getattr(r_item, "output", "") or "")[:100]
                elif r_type == "function_call":
                    r_text_preview = getattr(r_item, "name", "") or ""
                logger.info(f"AGENTBRICKS.node: raw_item[{ri}] type={r_type} | preview={r_text_preview} | request_id={request_id}")

            # Extract AgentBricks trace ID from response
            ab_trace_id = ""
            try:
                db_output = getattr(response, "databricks_output", None)
                if db_output and isinstance(db_output, dict):
                    ab_trace_id = db_output.get("trace", {}).get("trace_id", "")
                if not ab_trace_id:
                    ab_trace_id = getattr(response, "id", "") or ""
            except Exception:
                pass

            logger.info(f"AGENTBRICKS.node: ab_trace_id={ab_trace_id} | request_id={request_id}")

            # Parse the response
            parsed = _parse_ab_response(response)

            # Evaluation-canonical keys (used by scorers in evaluation notebooks):
            #   sql_query   → scored by sql_has_client_id_filter, sql_no_select_star, etc.
            #   nl_analysis → scored by grounding judge (recommendation_grounding)
            #   data_rows   → scored by agentbricks_returned_data
            # Existing ab_* keys kept for backward compatibility.
            _sql = parsed.get("sql_query") or ""
            _nl = (parsed.get("executive_summary") or "")[:1000]
            _table_md = (parsed.get("table_markdown") or "")
            _table_markdowns = parsed.get("table_markdowns") or []
            _row_count = max(len(_table_md.strip().split("\n")) - 2, 0) if _table_md else 0
            span.set_outputs({
                # ── Evaluation-canonical (scorers look for these) ──
                "sql_query": _sql,
                "nl_analysis": _nl,
                "data_rows": _table_md[:2000],
                # ── Legacy ab_* keys (backward-compatible) ──
                "ab_summary": _nl,
                "ab_table": _table_md[:2000],
                "ab_sql": _sql,
                "ab_query_description": (parsed.get("query_description") or "")[:500],
                "ab_trace_id": ab_trace_id,
                "ab_has_table": bool(_table_md),
                "ab_has_sql": bool(_sql),
                "ab_table_count": len(_table_markdowns),
            })
            span.set_attributes({
                "row_count": _row_count,
                "has_sql": bool(_sql),
                "table_count": len(_table_markdowns),
            })
        summary = parsed["executive_summary"] or "No analysis returned from AgentBricks."
        sql_query = parsed["sql_query"]
        table_md = parsed["table_markdown"]
        table_markdowns = parsed.get("table_markdowns") or ([] if not table_md else [table_md])

        # Build formatted response_text (backward-compatible flat markdown)
        response_text = f"**Analysis for {client_name}:**\n\n{summary}"
        if table_md:
            response_text += f"\n\n---\n**Data:**\n{table_md}"
        if sql_query:
            response_text += f"\n\n---\n**SQL Query Used:**\n```sql\n{sql_query}\n```"

        # Build structured response_items for frontend
        response_items = []
        if summary:
            response_items.append({"type": "text", "id": str(uuid.uuid4()), "value": summary})
        if table_md:
            response_items.append({"type": "table", "id": str(uuid.uuid4()), "value": table_md})
        if sql_query:
            response_items.append({"type": "sql", "id": str(uuid.uuid4()), "value": sql_query})
        if parsed["insights_raw"]:
            response_items.append({"type": "insights", "id": str(uuid.uuid4()), "value": parsed["insights_raw"]})

        logger.info(
            f"AGENTBRICKS.node: DONE | request_id={request_id} | items={len(response_items)} | "
            f"summary_len={len(summary)} | table={bool(table_md)} ({len(table_md)}ch) | "
            f"sql={bool(sql_query)} ({len(sql_query)}ch) | "
            f"insights={bool(parsed['insights_raw'])} | follow_ups={len(parsed['follow_ups'])} | "
            f"ab_trace_id={ab_trace_id}"
        )

        # Log what flows back to CoMarketer graph state
        logger.info(
            f"AGENTBRICKS.node: → COMARKETER state update | request_id={request_id} | "
            f"response_text_len={len(response_text)} | response_items_count={len(response_items)} | "
            f"item_types={[it['type'] for it in response_items]} | "
            f"ab_summary_len={len(summary[:2000])} | ab_sql_len={len(sql_query)} | "
            f"ab_table_len={len(table_md)} | ab_query_desc={bool(parsed['query_description'])} | "
            f"follow_ups={parsed['follow_ups'][:3]}"
        )
        logger.debug(
            f"AGENTBRICKS.node: → COMARKETER response_text preview | request_id={request_id} | "
            f"text={response_text[:300]}"
        )

        return {
            "response_text": response_text,
            "response_items": response_items,
            "ab_summary": summary[:2000],
            "ab_sql": sql_query,
            "ab_table": table_md,           # backward-compat: joined string for format_supervisor
            "ab_tables": table_markdowns,   # NEW: individual tables for streaming renderer
            "ab_insights": parsed["insights_raw"],
            "ab_query_description": parsed["query_description"],
            "follow_up_suggestions": parsed["follow_ups"],
            "ab_trace_id": ab_trace_id,
            "llm_call_count": prior_llm_calls + 1,
        }

    except Exception as e:
        logger.error(f"AGENTBRICKS.node: Error | request_id={request_id} | error={e}", exc_info=True)
        return _error_response(f"AgentBricks query failed: {str(e)[:200]}", request_id, prior_llm_calls)


def _error_response(message: str, request_id: str, prior_llm_calls: int) -> dict:
    """Build a standard error response dict."""
    logger.warning(f"AGENTBRICKS._error_response: → COMARKETER error state | request_id={request_id} | message={message[:200]}")
    return {
        "response_text": f"I encountered an issue while analyzing your data.\n\n{message}",
        "response_items": [{"type": "text", "id": str(uuid.uuid4()), "value": message}],
        "ab_summary": "",
        "ab_sql": "",
        "ab_table": "",
        "ab_tables": [],
        "ab_insights": "",
        "ab_query_description": "",
        "follow_up_suggestions": [],
        "error": message,
        "llm_call_count": prior_llm_calls,
    }
