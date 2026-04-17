"""Grant the app's Service Principal access to the SQL Warehouse."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import PermissionLevel

HOST = "https://dbc-540c0e05-7c19.cloud.databricks.com"
w = WorkspaceClient(host=HOST, auth_type="external-browser")
print(f"Authenticated as: {w.current_user.me().user_name}\n")

WAREHOUSE_ID = "64769e0e91e002b8"
SP_ID = 70759304728358  # service_principal_id

# 1. Grant SP CAN_USE on SQL Warehouse
print("1. Granting SP access to SQL Warehouse...")
try:
    w.warehouses.set_permissions(
        warehouse_id=WAREHOUSE_ID,
        access_control_list=[
            {
                "service_principal_name": "app-2mg93g comarketer",
                "permission_level": "CAN_USE",
            }
        ],
    )
    print("   Granted CAN_USE on warehouse")
except Exception as e:
    print(f"   Error: {e}")
    # Try alternative approach
    try:
        from databricks.sdk.service.sql import SetWorkspaceWarehouseConfigRequest
        perms = w.warehouses.get_permissions(warehouse_id=WAREHOUSE_ID)
        print(f"   Current permissions: {perms}")
    except Exception as e2:
        print(f"   Also failed: {e2}")

# 2. Verify current warehouse permissions
print("\n2. Current warehouse permissions:")
try:
    perms = w.warehouses.get_permissions(warehouse_id=WAREHOUSE_ID)
    for acl in perms.access_control_list or []:
        name = getattr(acl, 'user_name', None) or getattr(acl, 'group_name', None) or getattr(acl, 'service_principal_name', None) or 'unknown'
        for p in (acl.all_permissions or []):
            print(f"   {name}: {p.permission_level}")
except Exception as e:
    print(f"   Error: {e}")

# 3. Test writing to the table directly (as user, to verify schema)
print("\n3. Test INSERT (as user)...")
try:
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement="""
        INSERT INTO channel.gold_channel.comarketer_traces VALUES (
            'test-trace-001', 'test-req-001', '2026-03-11T21:00:00Z',
            '99999', 'test-sp', 'test query', 'greeting', 'greeting',
            'test response', 0, 50, 'success', '', 'test-conv', '{}', '', 'development'
        )
        """,
        wait_timeout="30s",
    )
    print(f"   Insert result: {result.status.state}")
    if result.status.error:
        print(f"   Error: {result.status.error.message}")
except Exception as e:
    print(f"   Error: {e}")

# 4. Verify the row
print("\n4. Check table contents...")
try:
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement="SELECT trace_id, client_id, status FROM channel.gold_channel.comarketer_traces",
        wait_timeout="30s",
    )
    if result.result and result.result.data_array:
        for row in result.result.data_array:
            print(f"   {row}")
    else:
        print("   No rows")
    print(f"   Status: {result.status.state}")
except Exception as e:
    print(f"   Error: {e}")

print("\nDone!")
