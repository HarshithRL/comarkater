"""Grant the app SP permission to query the AgentBricks serving endpoint."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointPermissionLevel

w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
print(f"Authenticated as: {w.current_user.me().user_name}")

ENDPOINT_NAME = "mas-0c49109d-endpoint"
APP_SP_NAME = "app-2mg93g comarketer"
APP_SP_ID = 70759304728358

# Get the endpoint to confirm it exists
endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint: {endpoint.name} (id={endpoint.id})")

# Get current permissions
perms = w.serving_endpoints.get_permissions(ENDPOINT_NAME)
print(f"\nCurrent permissions:")
for acl in (perms.access_control_list or []):
    principal = acl.user_name or acl.group_name or acl.service_principal_name or "unknown"
    for p in (acl.all_permissions or []):
        print(f"  {principal}: {p.permission_level}")

# Grant CAN_QUERY to app SP
from databricks.sdk.service.iam import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel

print(f"\nGranting CAN_QUERY to {APP_SP_NAME} (ID: {APP_SP_ID})...")
w.serving_endpoints.update_permissions(
    serving_endpoint_id=ENDPOINT_NAME,
    access_control_list=[
        ServingEndpointAccessControlRequest(
            service_principal_name=APP_SP_NAME,
            permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
        )
    ],
)

# Verify
perms = w.serving_endpoints.get_permissions(ENDPOINT_NAME)
print(f"\nUpdated permissions:")
for acl in (perms.access_control_list or []):
    principal = acl.user_name or acl.group_name or acl.service_principal_name or "unknown"
    for p in (acl.all_permissions or []):
        print(f"  {principal}: {p.permission_level}")

print("\nDone! App SP can now query the endpoint.")
