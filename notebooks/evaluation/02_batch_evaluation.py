# Databricks notebook source
# MAGIC %md
# MAGIC # CoMarketer — Batch Evaluation Job
# MAGIC
# MAGIC Runs comprehensive evaluation against recent production traces.
# MAGIC Intended to be scheduled as a Databricks Job (daily or weekly).
# MAGIC
# MAGIC **Stage 3 — Genie API Architecture:**
# MAGIC - Spans: `genie_call`, `genie_analysis`, `recommendation_gen`
# MAGIC - 16 scorers: 10 code-based + 3 built-in LLM + 3 custom LLM judges

# COMMAND ----------

# %pip install --upgrade "mlflow[databricks]>=3.1"
# dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from datetime import datetime, timedelta

EXPERIMENT_PATH = "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
mlflow.set_experiment(EXPERIMENT_PATH)

# Configurable lookback window
LOOKBACK_HOURS = 24   # Change to 168 for weekly runs
MAX_TRACES = 500

print(f"Experiment: {EXPERIMENT_PATH}")
print(f"Lookback: {LOOKBACK_HOURS}h | Max traces: {MAX_TRACES}")

# COMMAND ----------
# MAGIC %md ## Step 1 — Pull Production Traces

# COMMAND ----------

cutoff = (datetime.utcnow() - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")

traces_df = mlflow.search_traces(
    filter_string="trace.status = 'OK'",
    max_results=MAX_TRACES,
    order_by=["timestamp DESC"],
)

print(f"Found {len(traces_df)} traces for evaluation (cutoff: {cutoff})")

# COMMAND ----------
# MAGIC %md ## Step 2 — Define Scorers (full suite — no production monitoring limits)

# COMMAND ----------

from mlflow.genai.scorers import scorer, RelevanceToQuery, Safety, Guidelines
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback
from typing import Literal

# ── Built-in LLM judges ──
built_in_scorers = [
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="no_internal_system_details",
        guidelines=(
            "Response must not expose internal system details, SQL queries, "
            "table names (like campaign_details, gold_channel), database schema, "
            "Genie space IDs, or Databricks-related terminology."
        ),
    ),
]

# ── Code-based Genie data quality scorers ──


@scorer
def genie_completed(trace):
    """Check that Genie query completed successfully."""
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="no", rationale="No genie_call span found")
    status = (spans[0].attributes or {}).get("genie_status", "UNKNOWN")
    return _F(value="yes" if status == "COMPLETED" else "no", rationale=f"Status: {status}")


@scorer
def data_returned(trace):
    """Check that Genie returned data rows."""
    from mlflow.entities import Feedback as _F
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return _F(value="no", rationale="No genie_call span found")
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


# ── Code-based SQL scorers ──


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
    return _F(value="yes" if "where" in sql else "no", rationale="WHERE present" if "where" in sql else "MISSING WHERE")


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
    valid_tables = ["campaign_details", "campaign_insights"]
    found = [t for t in valid_tables if t in sql]
    return _F(value="yes" if found else "no", rationale=f"Tables: {found or 'NONE valid'}")


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
    return _F(value="yes" if not has_table else "no", rationale="Pure text" if not has_table else "Contains pipe table")


@scorer
def response_latency_within_sla(trace):
    """Total response time must be under 60s."""
    from mlflow.entities import Feedback as _F
    duration_ms = trace.info.execution_duration or 0
    return _F(value="yes" if duration_ms <= 60_000 else "no", rationale=f"{duration_ms}ms (SLA: 60s)")


@scorer(aggregations=["mean", "p50", "p90", "max"])
def response_latency_ms(trace):
    """Track latency distribution (ms)."""
    from mlflow.entities import Feedback as _F
    duration_ms = trace.info.execution_duration or 0
    return _F(value=float(duration_ms), rationale=f"{duration_ms}ms")


code_based_scorers = [
    genie_completed,
    data_returned,
    table_data_quality,
    genie_poll_duration,
    sql_no_select_star,
    sql_has_where_clause,
    sql_references_valid_table,
    analysis_no_tables,
    response_latency_within_sla,
    response_latency_ms,
]

# ── LLM Judge scorers (batch-only — too expensive for production monitoring) ──

grounding_judge = make_judge(
    name="recommendation_grounding",
    instructions=(
        "Analyze the {{ trace }} to find:\n"
        "1. The data returned by the genie_call span "
        "(outputs.data_rows and outputs.nl_analysis)\n"
        "2. The final recommendation in the recommendation_gen span "
        "(outputs.recommendation)\n\n"
        "Check:\n"
        "- Are all numbers in the recommendation traceable to the Genie data?\n"
        "- Does the recommendation avoid claiming facts not in the data?\n"
        "- Are comparative claims supported?\n\n"
        "Rate the grounding."
    ),
    feedback_value_type=Literal["grounded", "partially_grounded", "hallucinated"],
    model="databricks:/databricks-gpt-5-2",
)

sql_semantic_judge = make_judge(
    name="sql_semantic_correctness",
    instructions=(
        "Analyze the {{ trace }} to find the SQL query generated by the "
        "genie_call span (outputs.sql_query).\n\n"
        "Evaluate whether the SQL correctly addresses the user's question "
        "in {{ inputs }}.\n\n"
        "Check:\n"
        "1. Does the SQL query the right columns for the metric asked about?\n"
        "2. Are WHERE clauses appropriate for the question's filters?\n"
        "3. Are aggregations correct (SUM for totals, AVG for averages)?\n"
        "4. Does the query avoid retrieving unrelated data?\n\n"
        "Rate the SQL."
    ),
    feedback_value_type=Literal["correct", "partially_correct", "incorrect"],
    model="databricks:/databricks-gpt-5-2",
)

completeness_judge = make_judge(
    name="response_completeness",
    instructions=(
        "Review the {{ trace }} to understand the user's question (in {{ inputs }}) "
        "and the final agent response.\n\n"
        "Evaluate whether the response fully addresses all aspects of the question:\n"
        "1. Are all channels/metrics mentioned in the question covered?\n"
        "2. Are all time periods referenced in the question addressed?\n"
        "3. Does the response answer the implicit intent?\n\n"
        "Rate the completeness."
    ),
    feedback_value_type=Literal["complete", "partially_complete", "incomplete"],
    model="databricks:/databricks-gpt-5-2",
)

llm_judge_scorers = [grounding_judge, sql_semantic_judge, completeness_judge]

all_scorers = built_in_scorers + code_based_scorers + llm_judge_scorers
print(f"Total scorers: {len(all_scorers)}")

# COMMAND ----------
# MAGIC %md ## Step 3 — Run Evaluation

# COMMAND ----------

run_name = f"batch_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
print(f"Starting evaluation run: {run_name}")

with mlflow.start_run(run_name=run_name):
    results = mlflow.genai.evaluate(
        data=traces_df,
        scorers=all_scorers,
    )

print("Evaluation complete.")

# COMMAND ----------
# MAGIC %md ## Step 4 — Summary Metrics

# COMMAND ----------

print("=== Evaluation Summary ===")
for metric_name, value in sorted(results.metrics.items()):
    if isinstance(value, float):
        print(f"  {metric_name}: {value:.3f}")
    else:
        print(f"  {metric_name}: {value}")

# COMMAND ----------
# MAGIC %md ## Step 5 — Flag Problematic Traces

# COMMAND ----------

eval_table = results.tables.get("eval_results_table")

if eval_table is not None and len(eval_table) > 0:
    # Hallucinated recommendations
    grounding_col = next(
        (c for c in eval_table.columns if "grounding" in c.lower() and "value" in c.lower()),
        None,
    )
    if grounding_col:
        hallucinated = eval_table[eval_table[grounding_col] == "hallucinated"]
        if len(hallucinated) > 0:
            print(f"  {len(hallucinated)} traces with HALLUCINATED recommendations!")
            print(hallucinated[["trace_id", grounding_col]].to_string())
        else:
            print("  No hallucinated recommendations detected.")

    # Genie failures
    genie_col = next(
        (c for c in eval_table.columns if "genie_completed" in c.lower() and "value" in c.lower()),
        None,
    )
    if genie_col:
        failed = eval_table[eval_table[genie_col] == "no"]
        if len(failed) > 0:
            print(f"  {len(failed)} traces where Genie FAILED!")
        else:
            print("  All Genie queries completed successfully.")

    # No data returned
    data_col = next(
        (c for c in eval_table.columns if "data_returned" in c.lower() and "value" in c.lower()),
        None,
    )
    if data_col:
        no_data = eval_table[eval_table[data_col] == "no"]
        if len(no_data) > 0:
            print(f"  {len(no_data)} traces with NO data returned!")
        else:
            print("  All queries returned data.")

    # Analysis contains tables (should be pure text)
    table_col = next(
        (c for c in eval_table.columns if "analysis_no_tables" in c.lower() and "value" in c.lower()),
        None,
    )
    if table_col:
        has_tables = eval_table[eval_table[table_col] == "no"]
        if len(has_tables) > 0:
            print(f"  {len(has_tables)} traces where analysis contains pipe tables!")
        else:
            print("  All analysis outputs are pure text.")

    # SLA violations
    sla_col = next(
        (c for c in eval_table.columns if "latency_within_sla" in c.lower() and "value" in c.lower()),
        None,
    )
    if sla_col:
        slow = eval_table[eval_table[sla_col] == "no"]
        if len(slow) > 0:
            print(f"  {len(slow)} traces exceeded 60s SLA!")
        else:
            print("  All traces within 60s SLA.")
else:
    print("No evaluation results table available.")

# COMMAND ----------
# MAGIC %md ## Eval Results Table (sample)

# COMMAND ----------

if eval_table is not None:
    display(eval_table.head(20))
