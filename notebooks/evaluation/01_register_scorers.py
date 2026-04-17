# Databricks notebook source
# MAGIC %md
# MAGIC # CoMarketer — Register Production Monitoring Scorers
# MAGIC
# MAGIC Run this notebook ONCE to register scheduled scorers on the production experiment.
# MAGIC Re-run to update sample rates or add new scorers.
# MAGIC
# MAGIC **Stage 3 — Genie API Architecture:**
# MAGIC - Span names: `genie_call`, `genie_analysis`, `recommendation_gen`
# MAGIC - 16 scorers: 10 code-based (free) + 3 built-in LLM + 3 custom LLM judges
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - The App's Service Principal has "Can manage" on the experiment
# MAGIC - `mlflow[databricks]>=3.1` installed

# COMMAND ----------

# %pip install --upgrade "mlflow[databricks]>=3.1"
# dbutils.library.restartPython()

# COMMAND ----------

import mlflow

EXPERIMENT_PATH = "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Experiment set: {EXPERIMENT_PATH}")

# COMMAND ----------
# MAGIC %md ## Category 1 — Safety & Guardrails

# COMMAND ----------

from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig

# Built-in safety — checks for harmful/dangerous content
safety_scorer = Safety()
_s = safety_scorer.register(name="safety")
_s.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
print("Registered: safety (sample_rate=0.5)")

# Guidelines: no internal system details in response
internal_detail_check = Guidelines(
    name="no_internal_system_details",
    guidelines=(
        "Response must not expose internal system details, SQL queries, "
        "table names (like campaign_details, gold_channel), database schema, "
        "Genie space IDs, or Databricks-related terminology to the user. "
        "Technical implementation details must remain hidden."
    ),
)
_g = internal_detail_check.register(name="no_internal_system_details")
_g.start(sampling_config=ScorerSamplingConfig(sample_rate=0.3))
print("Registered: no_internal_system_details (sample_rate=0.3)")

# COMMAND ----------
# MAGIC %md ## Category 2 — Response Quality

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery

# Built-in relevance — does the output address the question?
relevance_scorer = RelevanceToQuery()
_r = relevance_scorer.register(name="relevance_to_query")
_r.start(sampling_config=ScorerSamplingConfig(sample_rate=0.3))
print("Registered: relevance_to_query (sample_rate=0.3)")

# COMMAND ----------
# MAGIC %md ## Category 3 — Genie Data Quality (Code-Based)
# MAGIC
# MAGIC These run cheap checks against the `genie_call` span outputs.
# MAGIC All imports are INLINE — required for production scorer serialization.

# COMMAND ----------

from mlflow.genai.scorers import scorer


@scorer
def genie_completed(trace):
    """Check that Genie query completed successfully (not FAILED/TIMEOUT)."""
    from mlflow.entities import Feedback
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return Feedback(value="no", rationale="No genie_call span found")
    status = (spans[0].attributes or {}).get("genie_status", "UNKNOWN")
    return Feedback(
        value="yes" if status == "COMPLETED" else "no",
        rationale=f"Genie status: {status}",
    )


@scorer
def data_returned(trace):
    """Check that Genie actually returned data rows."""
    from mlflow.entities import Feedback
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return Feedback(value="no", rationale="No genie_call span found")
    span = spans[0]
    row_count = (span.attributes or {}).get("row_count", 0) or 0
    if row_count == 0 and span.outputs:
        row_count = span.outputs.get("row_count", 0) or 0
    return Feedback(
        value="yes" if row_count > 0 else "no",
        rationale=f"Genie returned {row_count} data rows",
    )


@scorer
def table_data_quality(trace):
    """Check that result has both rows AND columns (not partial)."""
    from mlflow.entities import Feedback
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return Feedback(value="n/a", rationale="No genie_call span found")
    outputs = spans[0].outputs
    rows = outputs.get("row_count", 0) or 0
    cols = outputs.get("column_count", 0) or 0
    ok = rows > 0 and cols > 0
    return Feedback(
        value="yes" if ok else "no",
        rationale=f"rows={rows}, columns={cols}",
    )


@scorer
def genie_poll_duration(trace):
    """Flag if Genie polling took over 60s (performance issue)."""
    from mlflow.entities import Feedback
    spans = trace.search_spans(name="genie_call")
    if not spans:
        return Feedback(value="n/a", rationale="No genie_call span found")
    span = spans[0]
    # Span duration in milliseconds
    duration_ms = 0
    if hasattr(span, "end_time_ns") and hasattr(span, "start_time_ns"):
        if span.end_time_ns and span.start_time_ns:
            duration_ms = (span.end_time_ns - span.start_time_ns) / 1_000_000
    ok = duration_ms < 60_000
    return Feedback(
        value="yes" if ok else "no",
        rationale=f"Genie poll took {duration_ms:.0f}ms {'(OK)' if ok else '(SLOW)'}",
    )


# COMMAND ----------
# MAGIC %md ## Category 4 — SQL Sanity (Code-Based)

# COMMAND ----------


@scorer
def sql_no_select_star(trace):
    """SQL should not use SELECT * — must select specific columns."""
    from mlflow.entities import Feedback
    import re as _re
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return Feedback(value="n/a", rationale="No genie_call span found")
    sql = (spans[0].outputs.get("sql_query") or "").lower()
    if not sql:
        return Feedback(value="n/a", rationale="No SQL query in span")
    has_star = bool(_re.search(r"select\s+\*", sql))
    return Feedback(
        value="no" if has_star else "yes",
        rationale="SQL uses SELECT * (inefficient)" if has_star else "SQL selects specific columns",
    )


@scorer
def sql_has_where_clause(trace):
    """SQL must have WHERE clause — unbounded queries are dangerous."""
    from mlflow.entities import Feedback
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return Feedback(value="n/a", rationale="No genie_call span found")
    sql = (spans[0].outputs.get("sql_query") or "").lower()
    if not sql:
        return Feedback(value="n/a", rationale="No SQL query in span")
    has_where = "where" in sql
    return Feedback(
        value="yes" if has_where else "no",
        rationale="SQL has WHERE clause" if has_where else "SQL MISSING WHERE clause",
    )


@scorer
def sql_references_valid_table(trace):
    """SQL must reference known campaign tables only."""
    from mlflow.entities import Feedback
    spans = trace.search_spans(name="genie_call")
    if not spans or not spans[0].outputs:
        return Feedback(value="n/a", rationale="No genie_call span found")
    sql = (spans[0].outputs.get("sql_query") or "").lower()
    if not sql:
        return Feedback(value="n/a", rationale="No SQL query in span")
    valid_tables = ["campaign_details", "campaign_insights"]
    found = [t for t in valid_tables if t in sql]
    return Feedback(
        value="yes" if found else "no",
        rationale=f"Tables: {found if found else 'NONE of the valid tables'}",
    )


# COMMAND ----------
# MAGIC %md ## Category 5 — Analysis Quality (Code-Based)

# COMMAND ----------


@scorer
def analysis_no_tables(trace):
    """Analysis text must NOT contain pipe/markdown tables (tables sent separately)."""
    from mlflow.entities import Feedback
    import re as _re
    spans = trace.search_spans(name="genie_analysis")
    if not spans or not spans[0].outputs:
        return Feedback(value="n/a", rationale="No genie_analysis span found")
    analysis = spans[0].outputs.get("nl_analysis") or ""
    if not analysis:
        return Feedback(value="n/a", rationale="No analysis text in span")
    # Check for pipe table patterns: | col1 | col2 |
    has_pipe_table = bool(_re.search(r"\|.*\|.*\|", analysis))
    return Feedback(
        value="yes" if not has_pipe_table else "no",
        rationale="Analysis is pure text" if not has_pipe_table else "Analysis contains pipe table (should be text only)",
    )


# COMMAND ----------

# Register all code-based scorers (cheap — no LLM)
for _fn, _name, _rate in [
    (genie_completed, "genie_completed", 0.5),
    (data_returned, "data_returned", 0.5),
    (table_data_quality, "table_data_quality", 0.5),
    (genie_poll_duration, "genie_poll_duration", 0.5),
    (sql_no_select_star, "sql_no_select_star", 0.3),
    (sql_has_where_clause, "sql_has_where_clause", 0.3),
    (sql_references_valid_table, "sql_references_valid_table", 0.3),
    (analysis_no_tables, "analysis_no_tables", 0.3),
]:
    _reg = _fn.register(name=_name)
    _reg.start(sampling_config=ScorerSamplingConfig(sample_rate=_rate))
    print(f"Registered: {_name} (sample_rate={_rate})")

# COMMAND ----------
# MAGIC %md ## Category 6 — Grounding / Faithfulness (LLM Judge)
# MAGIC
# MAGIC Uses `{{ trace }}` to compare genie_call outputs vs recommendation_gen outputs.
# MAGIC `model=` argument is REQUIRED when using `{{ trace }}` in make_judge.

# COMMAND ----------

from mlflow.genai.judges import make_judge
from typing import Literal

grounding_judge = make_judge(
    name="recommendation_grounding",
    instructions=(
        "Analyze the {{ trace }} to find:\n"
        "1. The data returned by the genie_call span "
        "(look at outputs.data_rows and outputs.nl_analysis)\n"
        "2. The final recommendation in the recommendation_gen span "
        "(look at outputs.recommendation)\n\n"
        "Evaluate whether the recommendation is faithfully grounded in "
        "the Genie data.\n\n"
        "Check:\n"
        "- Are all numbers and percentages in the recommendation traceable "
        "to the Genie output?\n"
        "- Does the recommendation avoid claiming facts not in the data?\n"
        "- If comparative claims are made, are they supported by the data?\n\n"
        "Rate the grounding."
    ),
    feedback_value_type=Literal["grounded", "partially_grounded", "hallucinated"],
    model="databricks:/databricks-gpt-5-2",
)

_gj = grounding_judge.register(name="recommendation_grounding")
_gj.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
print("Registered: recommendation_grounding (sample_rate=0.2, LLM judge)")

# COMMAND ----------
# MAGIC %md ## Category 7 — Performance

# COMMAND ----------


@scorer
def response_latency_within_sla(trace):
    """Total response time must be under 60 seconds (Genie polling is slower than AgentBricks)."""
    from mlflow.entities import Feedback
    duration_ms = trace.info.execution_duration or 0
    threshold_ms = 60_000
    return Feedback(
        value="yes" if duration_ms <= threshold_ms else "no",
        rationale=f"Response took {duration_ms}ms (SLA: {threshold_ms}ms)",
    )


@scorer(aggregations=["mean", "p50", "p90", "max"])
def response_latency_ms(trace):
    """Track latency distribution (ms) for dashboards."""
    from mlflow.entities import Feedback
    duration_ms = trace.info.execution_duration or 0
    return Feedback(
        value=float(duration_ms),
        rationale=f"Execution duration: {duration_ms}ms",
    )


for _fn, _name, _rate in [
    (response_latency_within_sla, "response_latency_within_sla", 0.3),
    (response_latency_ms, "response_latency_ms", 0.3),
]:
    _reg = _fn.register(name=_name)
    _reg.start(sampling_config=ScorerSamplingConfig(sample_rate=_rate))
    print(f"Registered: {_name} (sample_rate={_rate})")

# COMMAND ----------
# MAGIC %md ## Verify — List All Registered Scorers

# COMMAND ----------

from mlflow.genai.scorers import list_scorers

print("=== Registered Production Scorers ===")
for s in list_scorers():
    print(f"  {s._server_name}: sample_rate={s.sample_rate}")

# COMMAND ----------
# MAGIC %md ## Management — Stop / Delete Scorers (run manually as needed)

# COMMAND ----------

# --- Uncomment to stop or delete individual scorers ---
# from mlflow.genai.scorers import get_scorer, delete_scorer
#
# get_scorer(name="safety").stop()
# get_scorer(name="recommendation_grounding").stop()
#
# # To permanently remove:
# delete_scorer(name="safety")
# delete_scorer(name="recommendation_grounding")
#
# # Remove OLD agentbricks scorers if still registered:
# for old_name in ["sql_has_client_id_filter", "agentbricks_returned_data",
#                   "has_agentbricks_span", "actionable_recommendations",
#                   "tradeoff_surfacing", "channel_specificity"]:
#     try:
#         delete_scorer(name=old_name)
#         print(f"Deleted old scorer: {old_name}")
#     except Exception:
#         pass
