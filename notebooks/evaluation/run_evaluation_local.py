"""Run CoMarketer batch evaluation locally.

Uses databricks.sdk browser auth (same as other test notebooks).
Requires: mlflow[databricks]>=3.1 installed in your local venv.

Stage 3 — Genie API Architecture:
  Spans: genie_call, genie_analysis, recommendation_gen
  16 scorers: 10 code-based (free) + optional LLM judges

Usage:
    cd comarketer
    python notebooks/evaluation/run_evaluation_local.py

    # With LLM judges (adds cost):
    RUN_LLM_JUDGES=true python notebooks/evaluation/run_evaluation_local.py
"""

import os

# ── Must set BEFORE importing mlflow ──
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["DATABRICKS_HOST"] = "https://dbc-540c0e05-7c19.cloud.databricks.com"

import mlflow
from datetime import datetime, timedelta, timezone
from databricks.sdk import WorkspaceClient

# ── Auth via browser (same as other test notebooks) ──
print("Authenticating with Databricks...")
w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
me = w.current_user.me()
print(f"Authenticated as: {me.user_name}\n")

# Propagate the SDK token to MLflow
token = w.config.token
if token:
    os.environ["DATABRICKS_TOKEN"] = token

# ── Experiment ──
EXPERIMENT_PATH = "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Experiment: {EXPERIMENT_PATH}")

# ── Pull recent traces ──
LOOKBACK_HOURS = 72   # last 3 days
MAX_TRACES = 50       # keep low to avoid Windows thread pool limit

print(f"\nSearching traces (last {LOOKBACK_HOURS}h, max {MAX_TRACES})...")
traces_df = mlflow.search_traces(
    filter_string="trace.status = 'OK'",
    max_results=MAX_TRACES,
    order_by=["timestamp DESC"],
)
print(f"Found {len(traces_df)} traces\n")

if len(traces_df) == 0:
    print("No traces found. Send a few queries through the app first, then re-run.")
    exit(0)

# Preview traces
print("=== Recent traces ===")
for _, row in traces_df.head(5).iterrows():
    tid = row.get("trace_id", "?")[:12]
    tags = row.get("tags", {}) or {}
    client = tags.get("client_name", "?")
    intent = tags.get("intent", "?")
    status = tags.get("status", "?")
    route = tags.get("agent_route", "?")
    print(f"  {tid}...  client={client}  intent={intent}  status={status}  route={route}")
print()

# ── Define code-based scorers (no LLM cost) ──
from mlflow.genai.scorers import scorer


@scorer
def genie_completed(trace):
    """Check that Genie query completed successfully."""
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="n/a", rationale="No genie_call span (may be greeting)")
    status = (spans[0].attributes or {}).get("genie_status", "UNKNOWN")
    return _F(value="yes" if status == "COMPLETED" else "no", rationale=f"Status: {status}")


@scorer
def data_returned(trace):
    """Check that Genie returned data rows."""
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
    """Check result has both rows AND columns."""
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return _F(value="n/a", rationale="No genie_call span")
    rows = spans[0].outputs.get("row_count", 0) or 0
    cols = spans[0].outputs.get("column_count", 0) or 0
    return _F(value="yes" if rows > 0 and cols > 0 else "no", rationale=f"rows={rows}, cols={cols}")


@scorer
def genie_poll_duration(trace):
    """Flag if Genie polling took over 60s."""
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
    """SQL should not use SELECT *."""
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
    """SQL must have WHERE clause."""
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
    """SQL must reference known campaign tables."""
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
    """Analysis text must NOT contain pipe/markdown tables."""
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
    """Response must complete within 60 seconds."""
    from mlflow.entities import Feedback as _F
    ms = trace.info.execution_duration or 0
    return _F(value="yes" if ms <= 60_000 else "no", rationale=f"{ms}ms (SLA: 60s)")


@scorer
def has_genie_span(trace):
    """Verify genie_call span exists (confirms Genie architecture is deployed)."""
    from mlflow.entities import Feedback as _F
    genie_spans = trace.search_spans(name="genie_call")
    old_spans = trace.search_spans(name="agentbricks_call")
    if genie_spans:
        return _F(value="yes", rationale=f"genie_call span found ({len(genie_spans)})")
    if old_spans:
        return _F(value="no", rationale="Old agentbricks_call span — redeploy needed")
    return _F(value="n/a", rationale="No data span (greeting trace)")


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
    has_genie_span,
]

# ── Optional: LLM judges (set RUN_LLM_JUDGES=true to enable) ──
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
    ]
    print(f"LLM judges enabled: {len(llm_scorers)} scorers\n")
else:
    print("LLM judges disabled. Set RUN_LLM_JUDGES=true to enable (adds cost).\n")

all_scorers = code_scorers + llm_scorers

# ── Run evaluation ──
run_name = f"local_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
print(f"Running evaluation: {run_name}")
print(f"Scorers: {len(all_scorers)} | Traces: {len(traces_df)}\n")

with mlflow.start_run(run_name=run_name):
    results = mlflow.genai.evaluate(
        data=traces_df,
        scorers=all_scorers,
    )

# ── Summary ──
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
for name, value in sorted(results.metrics.items()):
    bar = ""
    if isinstance(value, float) and 0 <= value <= 1:
        filled = int(value * 20)
        bar = f"  {'█' * filled}{'░' * (20 - filled)}"
    print(f"  {name:<45} {value:.3f}{bar}" if isinstance(value, float) else f"  {name:<45} {value}")

# ── Flag problems ──
eval_table = results.tables.get("eval_results_table")
if eval_table is not None and len(eval_table) > 0:
    print("\n" + "=" * 60)
    print("FLAGGED TRACES")
    print("=" * 60)

    def flag(col_keyword, bad_value, label):
        col = next((c for c in eval_table.columns if col_keyword in c and "value" in c), None)
        if col:
            bad = eval_table[eval_table[col] == bad_value]
            if len(bad) > 0:
                print(f"\n  {label}: {len(bad)} traces")
                for _, r in bad.head(3).iterrows():
                    tid = r.get("trace_id", "?")
                    rat_col = col.replace("/value", "/rationale")
                    rat = r.get(rat_col, "")
                    print(f"     trace_id={str(tid)[:20]}  {rat[:80]}")
            else:
                print(f"  {label}: all pass")

    flag("genie_completed",     "no",           "Genie FAILED queries")
    flag("data_returned",       "no",           "No data returned")
    flag("analysis_no_tables",  "no",           "Analysis contains pipe tables")
    flag("grounding",           "hallucinated", "Hallucinated recommendations")
    flag("latency_within_sla",  "no",           "SLA violations (>60s)")
    flag("has_genie_span",      "no",           "Old agentbricks span (needs redeploy)")

print(f"\nEvaluation complete. Check MLflow UI for full results:")
print(f"   Experiment: {EXPERIMENT_PATH}")
print(f"   Run: {run_name}")
