# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup Old AgentBricks-Era Scorers
# MAGIC
# MAGIC Run this notebook ONCE after deploying the Genie architecture.
# MAGIC Lists all registered scorers, deletes the old AgentBricks ones.
# MAGIC
# MAGIC **Must run in Databricks notebook** — `list_scorers()` / `delete_scorer()` require Databricks runtime.

# COMMAND ----------

import mlflow

EXPERIMENT_PATH = "/Users/harshith.r@diggibyte.com/netcore_insight_agent"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Experiment: {EXPERIMENT_PATH}")

# COMMAND ----------
# MAGIC %md ## Step 1 — List All Currently Registered Scorers

# COMMAND ----------

from mlflow.genai.scorers import list_scorers, get_scorer, delete_scorer

print("=== Currently Registered Scorers ===")
registered = list_scorers()
if not registered:
    print("  (none found)")
else:
    for s in registered:
        name = getattr(s, "_server_name", getattr(s, "name", "?"))
        rate = getattr(s, "sample_rate", "?")
        print(f"  {name} (sample_rate={rate})")

print(f"\nTotal: {len(registered)} scorers")

# COMMAND ----------
# MAGIC %md ## Step 2 — Delete Old AgentBricks-Era Scorers

# COMMAND ----------

OLD_SCORERS = [
    # Removed: Genie uses RLS, no CID in SQL
    "sql_has_client_id_filter",
    # Removed: migration checker no longer relevant
    "has_agentbricks_span",
    # Renamed: agentbricks_returned_data → data_returned
    "agentbricks_returned_data",
    # Removed: too vague guidelines
    "actionable_recommendations",
    "tradeoff_surfacing",
    "channel_specificity",
]

print("=== Deleting Old Scorers ===\n")
deleted = 0
for name in OLD_SCORERS:
    try:
        try:
            scorer = get_scorer(name=name)
            scorer.stop()
            print(f"  Stopped: {name}")
        except Exception:
            pass

        delete_scorer(name=name)
        print(f"  Deleted: {name}")
        deleted += 1
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "does not exist" in err.lower():
            print(f"  Skip:    {name} (not registered)")
        else:
            print(f"  Error:   {name} — {err[:100]}")

print(f"\nDeleted {deleted}/{len(OLD_SCORERS)} old scorers")

# COMMAND ----------
# MAGIC %md ## Step 3 — Verify Remaining Scorers

# COMMAND ----------

print("=== Remaining Scorers ===\n")
remaining = list_scorers()
if not remaining:
    print("  (none — ready to register new Genie scorers)")
else:
    for s in remaining:
        name = getattr(s, "_server_name", getattr(s, "name", "?"))
        rate = getattr(s, "sample_rate", "?")
        print(f"  {name} (sample_rate={rate})")

print(f"\nTotal remaining: {len(remaining)}")
print("\nNext: Run 01_register_scorers.py to register new Genie-era scorers.")
