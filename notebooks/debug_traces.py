"""Debug script — check traces, table, and app logs."""

from databricks.sdk import WorkspaceClient
import mlflow

HOST = "https://dbc-540c0e05-7c19.cloud.databricks.com"
w = WorkspaceClient(host=HOST, auth_type="external-browser")
print(f"Authenticated as: {w.current_user.me().user_name}\n")

EXPERIMENT_ID = "3310956511672050"
WAREHOUSE_ID = "64769e0e91e002b8"
TABLE = "channel.gold_channel.comarketer_traces"
SP_CLIENT_ID = "42aeaf49-13e8-4f57-97fe-c60f22b2576f"

# 1. MLflow Traces
print("=" * 50)
print("1. MLflow Traces")
print("=" * 50)
mlflow.set_tracking_uri("databricks")
try:
    traces = mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], max_results=5)
    print(f"   Traces: {len(traces)}")
    runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], max_results=5)
    print(f"   Runs: {len(runs)}")
except Exception as e:
    print(f"   Error: {e}")

# 2. Delta Table — try to CREATE it ourselves
print("\n" + "=" * 50)
print("2. Create Delta Table (as user, not SP)")
print("=" * 50)
try:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        trace_id        STRING      NOT NULL,
        request_id      STRING      NOT NULL,
        timestamp       STRING      NOT NULL,
        client_id       STRING      NOT NULL,
        sp_id           STRING      NOT NULL,
        user_query      STRING,
        intent          STRING,
        agent_route     STRING,
        response_text   STRING,
        llm_call_count  INT,
        latency_ms      INT,
        status          STRING      NOT NULL,
        error_message   STRING,
        conversation_id STRING,
        conversation    STRING,
        mlflow_trace_id STRING,
        environment     STRING
    ) USING DELTA
    COMMENT 'CoMarketer agent trace logs'
    """
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=create_sql,
        wait_timeout="30s",
    )
    print(f"   Result: {result.status.state}")
    if result.status.error:
        print(f"   Error: {result.status.error.message}")
    else:
        print(f"   Table created/verified: {TABLE}")
except Exception as e:
    print(f"   Error: {e}")

# 3. Grant SP access to the table
print("\n" + "=" * 50)
print("3. Grant SP Access to Table")
print("=" * 50)
try:
    grant_sql = f"GRANT ALL PRIVILEGES ON TABLE {TABLE} TO `{SP_CLIENT_ID}`"
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=grant_sql,
        wait_timeout="30s",
    )
    print(f"   Grant result: {result.status.state}")
    if result.status.error:
        print(f"   Error: {result.status.error.message}")
    else:
        print(f"   Granted ALL PRIVILEGES on {TABLE} to SP")
except Exception as e:
    print(f"   Error: {e}")

# 4. Grant SP USE SCHEMA
print("\n" + "=" * 50)
print("4. Grant SP Schema Access")
print("=" * 50)
try:
    for grant in [
        f"GRANT USE CATALOG ON CATALOG channel TO `{SP_CLIENT_ID}`",
        f"GRANT USE SCHEMA ON SCHEMA channel.gold_channel TO `{SP_CLIENT_ID}`",
    ]:
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=grant,
            wait_timeout="30s",
        )
        print(f"   {grant.split('GRANT ')[1].split(' TO')[0]}: {result.status.state}")
        if result.status.error:
            print(f"   Error: {result.status.error.message}")
except Exception as e:
    print(f"   Error: {e}")

# 5. Grant SP access to SQL warehouse
print("\n" + "=" * 50)
print("5. Verify Table Exists")
print("=" * 50)
try:
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=f"SELECT COUNT(*) FROM {TABLE}",
        wait_timeout="30s",
    )
    if result.result and result.result.data_array:
        print(f"   Row count: {result.result.data_array[0][0]}")
    print(f"   Status: {result.status.state}")
except Exception as e:
    print(f"   Error: {e}")

# 6. App logs via correct URL
print("\n" + "=" * 50)
print("6. App Logs")
print("=" * 50)
try:
    resp = w.api_client.do(
        method="GET",
        url=f"{HOST}/api/2.0/apps/comarketer/logs",
    )
    if isinstance(resp, dict):
        logs = resp.get("logs", resp.get("log_lines", str(resp)[:1000]))
        if isinstance(logs, list):
            for line in logs[-20:]:
                print(f"   {line}")
        else:
            print(f"   {str(logs)[:1000]}")
    else:
        print(f"   {str(resp)[:1000]}")
except Exception as e:
    print(f"   Logs API error: {e}")

print("\n" + "=" * 50)
print("DONE — Now redeploy and test again")
print("=" * 50)
