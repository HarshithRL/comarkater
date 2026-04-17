# Databricks notebook source
# MAGIC %md
# MAGIC # Discover Lakebase Projects, Branches & Endpoints
# MAGIC Run this notebook to find the correct `LAKEBASE_PROJECT` and `LAKEBASE_BRANCH`
# MAGIC values for CoMarketer's LTM configuration.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. List All Lakebase Projects

# COMMAND ----------

print("=" * 60)
print("LAKEBASE PROJECTS")
print("=" * 60)
projects = list(w.postgres.list_projects())
if not projects:
    print("  No projects found! Create one in Lakebase Console.")
else:
    for p in projects:
        state = getattr(p, "current_state", "UNKNOWN")
        display = getattr(p, "display_name", p.name)
        print(f"  Project: {p.name}")
        print(f"  Display: {display}")
        print(f"  State:   {state}")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. List Branches for Each Project

# COMMAND ----------

print("=" * 60)
print("BRANCHES PER PROJECT")
print("=" * 60)
for p in projects:
    print(f"\n  Project: {p.name}")
    print(f"  {'─' * 40}")
    try:
        branches = list(w.postgres.list_branches(parent=f"projects/{p.name}"))
        if not branches:
            print("    No branches found.")
        for b in branches:
            state = getattr(b, "current_state", "UNKNOWN")
            print(f"    Branch: {b.name}")
            print(f"    State:  {state}")
            print()
    except Exception as e:
        print(f"    ERROR listing branches: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. List Endpoints for Each Branch

# COMMAND ----------

print("=" * 60)
print("ENDPOINTS PER BRANCH")
print("=" * 60)
for p in projects:
    try:
        branches = list(w.postgres.list_branches(parent=f"projects/{p.name}"))
    except Exception:
        continue
    for b in branches:
        parent = f"projects/{p.name}/branches/{b.name}"
        print(f"\n  Parent: {parent}")
        print(f"  {'─' * 40}")
        try:
            endpoints = list(w.postgres.list_endpoints(parent=parent))
            if not endpoints:
                print("    No endpoints found (compute may be scaled to zero).")
            for ep in endpoints:
                ep_status = getattr(ep, "status", None)
                ep_type = getattr(ep_status, "endpoint_type", "UNKNOWN") if ep_status else "UNKNOWN"
                hosts = getattr(ep_status, "hosts", None) if ep_status else None
                host = getattr(hosts, "host", "N/A") if hosts else "N/A"
                print(f"    Endpoint: {ep.name}")
                print(f"    Type:     {ep_type}")
                print(f"    Host:     {host}")
                print()
        except Exception as e:
            print(f"    ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Recommended Config Values
# MAGIC Copy the correct project name and branch name from above into your env vars:
# MAGIC ```
# MAGIC LAKEBASE_PROJECT=<project_name_from_above>
# MAGIC LAKEBASE_BRANCH=<branch_name_from_above>
# MAGIC ```
# MAGIC
# MAGIC Or update `core/config.py` defaults directly.
