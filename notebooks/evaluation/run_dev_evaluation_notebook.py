# Databricks notebook source
# MAGIC %md
# MAGIC # CoMarketer — Dev Evaluation (Test Questions)
# MAGIC
# MAGIC Runs the deployed agent against a set of test questions, captures MLflow traces,
# MAGIC then evaluates them with all 16 scorers (code-based + LLM judges).
# MAGIC
# MAGIC **Usage:** Run All Cells. Add questions to the QUESTIONS list below.

# COMMAND ----------

# MAGIC %md ## Config — Test Questions & Client

# COMMAND ----------

# ═══════════════════════════════════════════════════════════════
#  TEST QUESTIONS — ADD YOUR QUESTIONS HERE
# ═══════════════════════════════════════════════════════════════

QUESTIONS = [
    "Show me top 5 email campaigns by open rate last month",
    # ──────────────────────────────────────────────────────
    # ADD MORE QUESTIONS BELOW (one per line):
    # "Compare Email vs WhatsApp performance in February 2026",
    # "What was SMS revenue last quarter?",
    # "Show me APN campaigns from January 1 to January 15 2026",
    # "Average click rate across all channels this month",
    # "Worst 5 BPN campaigns by delivery rate last month",
    # "How are my campaigns doing?",
    # "Hello",
    # ──────────────────────────────────────────────────────
]

# Which client to test with
SP_ID = "sp-igp-001"  # Options: sp-igp-001, sp-pepe-002, sp-crocs-003

# App URL
APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"

# Run LLM judges? (True = all 16 scorers, False = 10 code-based only)
RUN_LLM_JUDGES = True

print(f"Questions: {len(QUESTIONS)}")
print(f"Client: {SP_ID}")
print(f"LLM Judges: {'enabled' if RUN_LLM_JUDGES else 'disabled'}")

# COMMAND ----------

# MAGIC %md ## Step 1 — Call Agent for Each Question

# COMMAND ----------

import time
import json
import requests
from datetime import datetime, timezone

import mlflow

EXPERIMENT_PATH = "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
mlflow.set_experiment(EXPERIMENT_PATH)

# Get auth token for calling the Databricks App
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
token = w.config.token

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

print(f"\nCalling agent at {APP_URL}/invocations")
print(f"{'=' * 70}\n")

eval_start = datetime.now(timezone.utc)

for i, question in enumerate(QUESTIONS, 1):
    print(f"[{i}/{len(QUESTIONS)}] \"{question[:80]}\"")

    payload = {
        "input": [{"role": "user", "content": question}],
        "custom_inputs": {
            "sp_id": SP_ID,
            "user_name": "Dev Evaluator",
            "user_id": "dev-eval",
            "conversation_id": f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{i}",
            "task_type": "analytics",
        },
    }

    try:
        t0 = time.time()
        resp = requests.post(
            f"{APP_URL}/invocations",
            headers=headers,
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - t0

        if resp.ok:
            data = resp.json()
            custom_out = data.get("custom_outputs", {})
            if custom_out.get("error"):
                print(f"   ERROR ({elapsed:.1f}s): {custom_out['error'][:100]}")
            else:
                print(f"   OK ({elapsed:.1f}s) | intent={custom_out.get('intent', '?')}")
        else:
            print(f"   HTTP {resp.status_code} ({elapsed:.1f}s): {resp.text[:100]}")

    except Exception as e:
        print(f"   FAILED: {e}")

    if i < len(QUESTIONS):
        time.sleep(3)

print(f"\nAll {len(QUESTIONS)} questions sent. Waiting 10s for traces to flush...")
time.sleep(10)

# COMMAND ----------

# MAGIC %md ## Step 2 — Pull Fresh Traces

# COMMAND ----------

traces_df = mlflow.search_traces(
    filter_string="trace.status = 'OK'",
    max_results=len(QUESTIONS) * 2,
    order_by=["timestamp DESC"],
)

# Keep only the most recent N traces (the ones we just created)
if hasattr(traces_df, '__len__') and len(traces_df) > len(QUESTIONS):
    traces_df = traces_df[:len(QUESTIONS)]

print(f"Found {len(traces_df)} traces for evaluation")

# COMMAND ----------

# MAGIC %md ## Step 3 — Define Scorers (All 16)

# COMMAND ----------

from mlflow.genai.scorers import scorer


@scorer
def genie_completed(trace):
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="n/a", rationale="No genie_call span")
    status = (spans[0].attributes or {}).get("genie_status", "UNKNOWN")
    return _F(value="yes" if status == "COMPLETED" else "no", rationale=f"Status: {status}")


@scorer
def data_returned(trace):
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="n/a", rationale="No genie_call span")
    row_count = (spans[0].attributes or {}).get("row_count", 0) or 0
    if row_count == 0 and spans[0].outputs:
        row_count = spans[0].outputs.get("row_count", 0) or 0
    return _F(value="yes" if row_count > 0 else "no", rationale=f"Rows: {row_count}")


@scorer
def table_data_quality(trace):
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_call span")
    rows = spans[0].outputs.get("row_count", 0) or 0
    cols = spans[0].outputs.get("column_count", 0) or 0
    return _F(value="yes" if rows > 0 and cols > 0 else "no", rationale=f"rows={rows}, cols={cols}")


@scorer
def genie_poll_duration(trace):
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="n/a", rationale="No genie_call span")
    span = spans[0]
    duration_ms = 0
    if hasattr(span, "end_time_ns") and hasattr(span, "start_time_ns"):
        if span.end_time_ns and span.start_time_ns:
            duration_ms = (span.end_time_ns - span.start_time_ns) / 1_000_000
    return _F(value="yes" if duration_ms < 60_000 else "no", rationale=f"Poll: {duration_ms:.0f}ms")


@scorer
def sql_no_select_star(trace):
    from mlflow.entities import Feedback as _F
    import re as _re
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_call span")
    sql = (spans[0].outputs.get("sql_query") or "").lower()
    if not sql:
        return _F(value="n/a", rationale="No SQL")
    has_star = bool(_re.search(r"select\s+\*", sql))
    return _F(value="no" if has_star else "yes", rationale="SELECT *" if has_star else "Specific columns")


@scorer
def sql_has_where_clause(trace):
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_call span")
    sql = (spans[0].outputs.get("sql_query") or "").lower()
    if not sql:
        return _F(value="n/a", rationale="No SQL")
    return _F(value="yes" if "where" in sql else "no",
              rationale="Has WHERE" if "where" in sql else "MISSING WHERE")


@scorer
def sql_references_valid_table(trace):
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_call span")
    sql = (spans[0].outputs.get("sql_query") or "").lower()
    if not sql:
        return _F(value="n/a", rationale="No SQL")
    valid = ["campaign_details", "campaign_insights"]
    found = [t for t in valid if t in sql]
    return _F(value="yes" if found else "no", rationale=f"Tables: {found or 'NONE'}")


@scorer
def analysis_no_tables(trace):
    from mlflow.entities import Feedback as _F
    import re as _re
    spans = trace.search_spans(name="genie_analysis")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_analysis span")
    analysis = spans[0].outputs.get("nl_analysis") or ""
    if not analysis:
        return _F(value="n/a", rationale="No analysis text")
    has_table = bool(_re.search(r"\|.*\|.*\|", analysis))
    return _F(value="yes" if not has_table else "no",
              rationale="Pure text" if not has_table else "Contains pipe table")


@scorer
def response_latency_within_sla(trace):
    from mlflow.entities import Feedback as _F
    ms = trace.info.execution_duration or 0
    return _F(value="yes" if ms <= 60_000 else "no", rationale=f"{ms}ms (SLA: 60s)")


code_scorers = [
    genie_completed,
    data_returned,
    table_data_quality,
    genie_poll_duration,
    sql_no_select_star,
    sql_has_where_clause,
    sql_references_valid_table,
    analysis_no_tables,
    response_latency_within_sla,
]

# ── LLM Judges ──
llm_scorers = []
if RUN_LLM_JUDGES:
    from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines
    from mlflow.genai.judges import make_judge
    from typing import Literal

    llm_scorers = [
        RelevanceToQuery(),
        Safety(),
        Guidelines(
            name="no_internal_system_details",
            guidelines=(
                "Response must not expose internal system details, SQL queries, "
                "table names, Genie space IDs, or Databricks terminology."
            ),
        ),
        make_judge(
            name="recommendation_grounding",
            instructions=(
                "Analyze the {{ trace }} to find the genie_call span outputs "
                "(data_rows, nl_analysis) and the recommendation_gen span outputs "
                "(recommendation).\n\n"
                "Is the recommendation faithfully grounded in the data? "
                "Are numbers/percentages traceable to the Genie output?"
            ),
            feedback_value_type=Literal["grounded", "partially_grounded", "hallucinated"],
            model="databricks:/databricks-gpt-5-2",
        ),
        make_judge(
            name="sql_semantic_correctness",
            instructions=(
                "Analyze the {{ trace }} to find the SQL in genie_call span "
                "(outputs.sql_query) and the user's question in {{ inputs }}.\n\n"
                "Does the SQL correctly address the question? "
                "Right columns? Correct filters? Proper aggregations?"
            ),
            feedback_value_type=Literal["correct", "partially_correct", "incorrect"],
            model="databricks:/databricks-gpt-5-2",
        ),
        make_judge(
            name="response_completeness",
            instructions=(
                "Review the {{ trace }}: user question in {{ inputs }} "
                "and the final agent response.\n\n"
                "Does the response fully address all aspects of the question? "
                "All channels, metrics, and time periods covered?"
            ),
            feedback_value_type=Literal["complete", "partially_complete", "incomplete"],
            model="databricks:/databricks-gpt-5-2",
        ),
    ]

all_scorers = code_scorers + llm_scorers
print(f"Scorers: {len(code_scorers)} code + {len(llm_scorers)} LLM = {len(all_scorers)} total")

# COMMAND ----------

# MAGIC %md ## Step 4 — Run Evaluation

# COMMAND ----------

run_name = f"dev_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
print(f"Running: {run_name} | {len(all_scorers)} scorers | {len(traces_df)} traces\n")

with mlflow.start_run(run_name=run_name):
    results = mlflow.genai.evaluate(
        data=traces_df,
        scorers=all_scorers,
    )

print("Evaluation complete.")

# COMMAND ----------

# MAGIC %md ## Step 5 — Results Summary

# COMMAND ----------

print(f"{'=' * 70}")
print(f"  DEV EVALUATION — {len(QUESTIONS)} questions | Client: {SP_ID}")
print(f"{'=' * 70}\n")

for name, value in sorted(results.metrics.items()):
    bar = ""
    if isinstance(value, float) and 0 <= value <= 1:
        filled = int(value * 20)
        bar = f"  {'█' * filled}{'░' * (20 - filled)}"
    if isinstance(value, float):
        print(f"  {name:<50} {value:.3f}{bar}")
    else:
        print(f"  {name:<50} {value}")

# COMMAND ----------

# MAGIC %md ## Step 6 — Flagged Issues

# COMMAND ----------

eval_table = results.tables.get("eval_results_table")

if eval_table is not None and len(eval_table) > 0:
    def flag(col_keyword, bad_value, label):
        col = next((c for c in eval_table.columns if col_keyword in c and "value" in c), None)
        if col:
            bad = eval_table[eval_table[col] == bad_value]
            if len(bad) > 0:
                print(f"  FAIL  {label}: {len(bad)}/{len(eval_table)} traces")
            else:
                print(f"  PASS  {label}")

    flag("genie_completed",     "no",           "Genie query completion")
    flag("data_returned",       "no",           "Data returned")
    flag("analysis_no_tables",  "no",           "Analysis is pure text")
    flag("sql_no_select_star",  "no",           "SQL avoids SELECT *")
    flag("sql_has_where",       "no",           "SQL has WHERE clause")
    flag("latency_within_sla",  "no",           "Response within 60s SLA")
    if RUN_LLM_JUDGES:
        flag("grounding",       "hallucinated", "No hallucination")
        flag("sql_semantic",    "incorrect",    "SQL correctness")
        flag("completeness",    "incomplete",   "Response completeness")
else:
    print("No evaluation results table.")

# COMMAND ----------

# MAGIC %md ## Full Results Table

# COMMAND ----------

if eval_table is not None:
    display(eval_table)
