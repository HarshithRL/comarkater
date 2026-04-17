"""Synthesizer node — combines results from capability steps OR parallel workers.

Supports two input modes:
  1. step_results (from insight_agent sequential execution — new path)
  2. worker_results (from parallel genie_workers — legacy path)

Writes to the SAME genie_* fields that format_supervisor reads,
so format_supervisor needs ZERO changes.
"""
from __future__ import annotations

import logging
import uuid

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from core.config import settings
from core.state import AgentState
from prompts.synthesizer_prompt import SYNTHESIZER_PROMPT
from parsers.table_truncator import truncate_table_for_llm, BUDGET_SYNTHESIZER_PER_WORKER

logger = logging.getLogger(__name__)


def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Combine step_results or worker_results into unified genie_* fields.

    Reads: step_results, worker_results, original_question, plan
    Writes: genie_summary, genie_tables, genie_table, genie_columns,
            genie_data_array, genie_sql, genie_text_content, genie_insights,
            follow_up_suggestions, response_text, response_items
    """
    request_id = state.get("request_id", "unknown")
    question = state.get("rewritten_question") or state.get("original_question", "")
    client_name = state.get("client_name", "there")
    step_results = state.get("step_results") or []
    worker_results = state.get("worker_results") or []
    prior_llm_calls = state.get("llm_call_count", 0)

    token = config["configurable"]["sp_token"]

    # Determine which path we're on
    if step_results:
        return _synthesize_step_results(
            step_results, question, client_name, request_id, token, prior_llm_calls,
        )
    elif worker_results:
        return _synthesize_worker_results(
            state, question, client_name, request_id, token, prior_llm_calls,
        )
    else:
        error_msg = "No results to synthesize."
        logger.error("SYNTHESIZER: no results | request_id=%s", request_id)
        return _empty_result(error_msg, prior_llm_calls)


def _synthesize_step_results(
    step_results: list,
    question: str,
    client_name: str,
    request_id: str,
    token: str,
    prior_llm_calls: int,
) -> dict:
    """Synthesize results from insight_agent sequential steps."""

    logger.info(
        "SYNTHESIZER: step_results path | steps=%d | request_id=%s",
        len(step_results), request_id,
    )

    successful = [r for r in step_results if not r.get("error")]
    failed = [r for r in step_results if r.get("error")]

    if failed:
        logger.warning(
            "SYNTHESIZER: %d/%d steps failed | request_id=%s",
            len(failed), len(step_results), request_id,
        )

    if not successful:
        return _empty_result("All analysis steps failed.", prior_llm_calls)

    # If only one step, use its result directly — no synthesis needed
    if len(successful) == 1:
        result_text = successful[0].get("result", "")
        return _build_output(result_text, question, client_name, prior_llm_calls)

    # Multiple steps — synthesize via LLM
    results_block = ""
    for i, r in enumerate(successful):
        results_block += f"\n### Analysis {i + 1}: {r.get('task', 'N/A')}\n"
        results_block += f"{r.get('result', 'No result.')}\n"

    summary = _llm_synthesize(results_block, question, token, request_id)
    return _build_output(summary, question, client_name, prior_llm_calls + 1)


def _synthesize_worker_results(
    state: AgentState,
    question: str,
    client_name: str,
    request_id: str,
    token: str,
    prior_llm_calls: int,
) -> dict:
    """Synthesize results from parallel genie_workers (legacy path)."""
    worker_results = state.get("worker_results") or []

    logger.info(
        "SYNTHESIZER: worker_results path | workers=%d | request_id=%s",
        len(worker_results), request_id,
    )

    successful = [r for r in worker_results if not r.get("error")]
    failed = [r for r in worker_results if r.get("error")]

    if failed:
        logger.warning(
            "SYNTHESIZER: %d/%d workers failed | request_id=%s | errors=%s",
            len(failed), len(worker_results), request_id,
            [f"[{r.get('worker_index')}] {r.get('error', '')[:80]}" for r in failed],
        )

    if not successful:
        error_msg = "All data queries failed. Please try rephrasing your question."
        logger.error("SYNTHESIZER: ALL workers failed | request_id=%s", request_id)
        return _empty_result(error_msg, prior_llm_calls)

    # Aggregate results from successful workers
    all_tables = []
    all_table_text_parts = []
    all_sql_parts = []
    all_columns = []
    all_data_array = []

    for r in sorted(successful, key=lambda x: x.get("worker_index", 0)):
        for t in r.get("genie_tables", []):
            if isinstance(t, dict) and t.get("data"):
                all_tables.append(t)
        table_text = r.get("genie_table", "")
        if table_text:
            label = r.get("sub_question", f"Query {r.get('worker_index', '?')}")
            all_table_text_parts.append(f"--- {label} ---\n{table_text}")
        sql = r.get("genie_sql", "")
        if sql:
            all_sql_parts.append(sql)
        if not all_columns and r.get("genie_columns"):
            all_columns = r["genie_columns"]
        all_data_array.extend(r.get("genie_data_array") or [])

    combined_table_text = "\n\n".join(all_table_text_parts)
    combined_sql = "\n---\n".join(all_sql_parts)

    # LLM synthesis
    results_block = ""
    for i, r in enumerate(sorted(successful, key=lambda x: x.get("worker_index", 0))):
        results_block += f"\n### Data Set {i + 1}: {r.get('sub_question', 'N/A')}\n"
        results_block += f"Purpose: {r.get('purpose', 'N/A')}\n"
        worker_cols = r.get("genie_columns") or []
        worker_data = r.get("genie_data_array") or []
        if worker_cols and worker_data:
            worker_table = truncate_table_for_llm(
                worker_cols, worker_data,
                char_budget=BUDGET_SYNTHESIZER_PER_WORKER,
                include_stats=True,
            )
        else:
            worker_table = r.get("genie_table", "No data.")[:BUDGET_SYNTHESIZER_PER_WORKER]
        results_block += f"Data:\n{worker_table}\n"
        desc = r.get("genie_query_description", "")
        if desc:
            results_block += f"Description: {desc}\n"

    summary = _llm_synthesize(results_block, question, token, request_id)
    if not summary:
        fallback_parts = [r.get("genie_text_content", "") for r in successful if r.get("genie_text_content")]
        summary = "\n\n".join(fallback_parts) if fallback_parts else f"Data retrieved: {len(all_data_array)} total rows across {len(successful)} queries."

    response_text = f"**Analysis for {client_name}:**\n\n{summary}"
    response_items = [{"type": "text", "id": str(uuid.uuid4()), "value": summary}] if summary else []

    logger.info(
        "SYNTHESIZER: DONE | request_id=%s | tables=%d | summary=%dch",
        request_id, len(all_tables), len(summary),
    )

    return {
        "genie_summary": summary[:3000],
        "genie_insights": summary[:3000],
        "genie_table": combined_table_text,
        "genie_tables": all_tables,
        "genie_columns": all_columns,
        "genie_data_array": all_data_array,
        "genie_sql": combined_sql,
        "genie_text_content": "",
        "genie_query_description": question,
        "follow_up_suggestions": [],
        "genie_trace_id": ", ".join(r.get("genie_trace_id", "") for r in successful if r.get("genie_trace_id")),
        "response_text": response_text,
        "response_items": response_items,
        "llm_call_count": prior_llm_calls + 1,
    }


def _llm_synthesize(results_block: str, question: str, token: str, request_id: str) -> str:
    """Run LLM synthesis on combined results. Returns summary text."""
    try:
        with mlflow.start_span(name="synthesizer_combine") as span:
            span.set_attributes({"request_id": request_id})

            llm = ChatOpenAI(
                model=settings.LLM_ENDPOINT_NAME,
                api_key=token,
                base_url=settings.AI_GATEWAY_URL,
                temperature=0.0,
            )

            prompt_text = SYNTHESIZER_PROMPT.format(
                result_count=results_block.count("###"),
                question=question,
                results_block=results_block,
            )

            span.set_inputs({"prompt_length": len(prompt_text), "question": question[:200]})

            try:
                from langgraph.config import get_stream_writer
                _writer = get_stream_writer()
                _writer({"event_type": "node_started", "node": "synthesizer", "message": "Synthesizing results across queries..."})
            except Exception:
                pass

            response = llm.invoke([
                SystemMessage(content=prompt_text),
                HumanMessage(content=question),
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            summary = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip()

            span.set_outputs({"synthesis_length": len(summary), "synthesis": summary[:2000]})
            logger.info("SYNTHESIZER: LLM done | summary=%dch | request_id=%s", len(summary), request_id)
            return summary

    except Exception as e:
        logger.error("SYNTHESIZER: LLM failed | request_id=%s | error=%s", request_id, e)
        return ""


def _build_output(summary: str, question: str, client_name: str, llm_call_count: int) -> dict:
    """Build synthesizer output dict from a summary string (step_results path)."""
    if not summary:
        summary = "Analysis complete."

    return {
        "genie_summary": summary[:3000],
        "genie_insights": summary[:3000],
        "genie_table": "",
        "genie_tables": [],
        "genie_columns": [],
        "genie_data_array": [],
        "genie_sql": "",
        "genie_text_content": "",
        "genie_query_description": question,
        "follow_up_suggestions": [],
        "response_text": f"**Analysis for {client_name}:**\n\n{summary}",
        "response_items": [{"type": "text", "id": str(uuid.uuid4()), "value": summary}],
        "llm_call_count": llm_call_count,
    }


def _empty_result(error_msg: str, prior_llm_calls: int) -> dict:
    """Build empty error result."""
    return {
        "genie_summary": error_msg,
        "genie_insights": error_msg,
        "genie_table": "",
        "genie_tables": [],
        "genie_columns": [],
        "genie_data_array": [],
        "genie_sql": "",
        "genie_text_content": "",
        "genie_query_description": "",
        "follow_up_suggestions": [],
        "response_text": error_msg,
        "response_items": [{"type": "text", "id": str(uuid.uuid4()), "value": error_msg}],
        "llm_call_count": prior_llm_calls,
        "error": error_msg,
    }
