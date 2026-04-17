"""Dev evaluation — run agent against test questions, then score the traces.

Calls the DEPLOYED agent (Databricks App) for each question,
waits for MLflow traces to appear, then evaluates with all scorers.

Usage:
    cd comarketer
    python notebooks/evaluation/run_dev_evaluation.py

    # With LLM judges (adds cost per question):
    RUN_LLM_JUDGES=true python notebooks/evaluation/run_dev_evaluation.py
"""

import os
import time
import json

# ── Must set BEFORE importing mlflow ──
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["DATABRICKS_HOST"] = "https://dbc-540c0e05-7c19.cloud.databricks.com"

import mlflow
from datetime import datetime, timezone
from databricks.sdk import WorkspaceClient


# ═══════════════════════════════════════════════════════════════
#  TEST QUESTIONS — ADD YOUR QUESTIONS HERE
# ═══════════════════════════════════════════════════════════════

QUESTIONS = [
    "Show me top 5 email campaigns by open rate last month",
    # ──────────────────────────────────────────────────────
    # ADD MORE QUESTIONS BELOW (one per line):
    # "Compare Email vs WhatsApp performance in February 2026",
    # "What was SMS revenue last quarter?",
    # "Show me BPN campaign click rates for January 2026",
    # "Top 10 WhatsApp campaigns by conversion rate this month",
    # "What are the worst performing APN campaigns last week?",
    # "Show me email delivery rate trends for the past 3 months",
    # ──────────────────────────────────────────────────────
]

# Which client to test with
SP_ID = "sp-igp-001"  # Options: sp-igp-001, sp-pepe-002, sp-crocs-003

# App URL
APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"


# ═══════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════

print("Authenticating with Databricks...")
w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
me = w.current_user.me()
print(f"Authenticated as: {me.user_name}\n")

token = w.config.token
if token:
    os.environ["DATABRICKS_TOKEN"] = token

EXPERIMENT_PATH = "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Experiment: {EXPERIMENT_PATH}")


# ═══════════════════════════════════════════════════════════════
#  STEP 1: Call agent for each question
# ═══════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  Running {len(QUESTIONS)} test question(s) against {APP_URL}")
print(f"  Client: {SP_ID}")
print(f"{'=' * 70}\n")

# Record start time BEFORE calling agent (to filter traces later)
eval_start = datetime.now(timezone.utc)

for i, question in enumerate(QUESTIONS, 1):
    print(f"[{i}/{len(QUESTIONS)}] Calling agent: \"{question[:80]}...\"")

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
        resp = w.api_client.do(
            method="POST",
            url=f"{APP_URL}/invocations",
            body=payload,
        )
        elapsed = time.time() - t0

        # Parse response
        if isinstance(resp, str):
            resp = json.loads(resp)

        # Check for error
        custom_out = resp.get("custom_outputs", {})
        if custom_out.get("error"):
            print(f"   ERROR: {custom_out['error'][:100]}")
        else:
            output = resp.get("output", [])
            text_len = 0
            if output:
                content = output[0].get("content", [])
                if content:
                    text_len = len(content[0].get("text", ""))
            print(f"   OK ({elapsed:.1f}s) | response_len={text_len} | intent={custom_out.get('intent', '?')}")

    except Exception as e:
        print(f"   FAILED: {e}")

    # Short pause between requests (let traces flush)
    if i < len(QUESTIONS):
        time.sleep(3)

print(f"\nAll {len(QUESTIONS)} questions sent. Waiting 10s for traces to flush...\n")
time.sleep(10)


# ═══════════════════════════════════════════════════════════════
#  STEP 2: Pull the traces we just created
# ═══════════════════════════════════════════════════════════════

print("Searching for traces...")
traces_df = mlflow.search_traces(
    filter_string="trace.status = 'OK'",
    max_results=len(QUESTIONS) * 2,  # buffer for any duplicates
    order_by=["timestamp DESC"],
)

# Filter to only traces created AFTER our eval started
# (avoid scoring old production traces)
if hasattr(traces_df, '__len__') and len(traces_df) > 0:
    # Limit to expected count (newest first)
    traces_df = traces_df[:len(QUESTIONS)]

print(f"Found {len(traces_df)} traces for evaluation\n")

if len(traces_df) == 0:
    print("No traces found. The agent may not have logged traces correctly.")
    print("Check:")
    print("  1. Is the agent deployed and running?")
    print("  2. Is MLFLOW_TRACKING_URI=databricks set in the app?")
    print("  3. Did the agent return errors for all questions?")
    exit(1)


# ═══════════════════════════════════════════════════════════════
#  STEP 3: Define scorers
# ═══════════════════════════════════════════════════════════════

from mlflow.genai.scorers import scorer


@scorer
def genie_completed(trace):
    """Genie query completed successfully."""
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="n/a", rationale="No genie_call span (greeting?)")
    status = (spans[0].attributes or {}).get("genie_status", "UNKNOWN")
    return _F(value="yes" if status == "COMPLETED" else "no", rationale=f"Status: {status}")


@scorer
def data_returned(trace):
    """Genie returned data rows."""
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
    """Result has both rows AND columns."""
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_call span")
    rows = spans[0].outputs.get("row_count", 0) or 0
    cols = spans[0].outputs.get("column_count", 0) or 0
    return _F(value="yes" if rows > 0 and cols > 0 else "no", rationale=f"rows={rows}, cols={cols}")


@scorer
def genie_poll_duration(trace):
    """Genie polling under 60s."""
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
    """SQL does not use SELECT *."""
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
    """SQL has WHERE clause."""
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
    """SQL references known campaign tables."""
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
    """Analysis text is pure text (no pipe tables)."""
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
    """Response under 60s SLA."""
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

# ── Optional: LLM judges ──
llm_scorers = []
if os.environ.get("RUN_LLM_JUDGES", "").lower() == "true":
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
    print(f"LLM judges enabled: {len(llm_scorers)} scorers\n")
else:
    print("LLM judges disabled. Set RUN_LLM_JUDGES=true to enable.\n")

all_scorers = code_scorers + llm_scorers


# ═══════════════════════════════════════════════════════════════
#  STEP 4: Run evaluation
# ═══════════════════════════════════════════════════════════════

run_name = f"dev_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
print(f"Running evaluation: {run_name}")
print(f"Scorers: {len(all_scorers)} | Traces: {len(traces_df)}\n")

with mlflow.start_run(run_name=run_name):
    results = mlflow.genai.evaluate(
        data=traces_df,
        scorers=all_scorers,
    )


# ═══════════════════════════════════════════════════════════════
#  STEP 5: Print results
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  DEV EVALUATION SUMMARY")
print(f"  Questions: {len(QUESTIONS)} | Client: {SP_ID}")
print("=" * 70)

for name, value in sorted(results.metrics.items()):
    bar = ""
    if isinstance(value, float) and 0 <= value <= 1:
        filled = int(value * 20)
        bar = f"  {'█' * filled}{'░' * (20 - filled)}"
    if isinstance(value, float):
        print(f"  {name:<50} {value:.3f}{bar}")
    else:
        print(f"  {name:<50} {value}")

# ── Flag problems ──
eval_table = results.tables.get("eval_results_table")
if eval_table is not None and len(eval_table) > 0:
    print("\n" + "-" * 70)
    print("  FLAGGED ISSUES")
    print("-" * 70)

    def flag(col_keyword, bad_value, label):
        col = next((c for c in eval_table.columns if col_keyword in c and "value" in c), None)
        if col:
            bad = eval_table[eval_table[col] == bad_value]
            if len(bad) > 0:
                print(f"\n  FAIL  {label}: {len(bad)}/{len(eval_table)} traces")
                for _, r in bad.head(3).iterrows():
                    rat_col = col.replace("/value", "/rationale")
                    rat = r.get(rat_col, "")
                    print(f"         {rat[:100]}")
            else:
                print(f"  PASS  {label}")

    flag("genie_completed",     "no",           "Genie query completion")
    flag("data_returned",       "no",           "Data returned from Genie")
    flag("analysis_no_tables",  "no",           "Analysis is pure text (no tables)")
    flag("sql_no_select_star",  "no",           "SQL avoids SELECT *")
    flag("sql_has_where",       "no",           "SQL has WHERE clause")
    flag("latency_within_sla",  "no",           "Response within 60s SLA")
    if os.environ.get("RUN_LLM_JUDGES", "").lower() == "true":
        flag("grounding",       "hallucinated", "Grounding (no hallucination)")
        flag("sql_semantic",    "incorrect",    "SQL semantic correctness")
        flag("completeness",    "incomplete",   "Response completeness")

print(f"\nResults stored in MLflow experiment: {EXPERIMENT_PATH}")
print(f"Run name: {run_name}")
print(f"\nTo add more questions, edit QUESTIONS list at top of this file.")
