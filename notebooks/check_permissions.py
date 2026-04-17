"""Check permissions on the AgentBricks serving endpoint."""

from databricks.sdk import WorkspaceClient

w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)

ENDPOINT_NAME = "mas-0c49109d-endpoint"

# Get endpoint to find its ID
endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint: {endpoint.name} (id={endpoint.id})")

# Use the ID for permissions API
perms = w.serving_endpoints.get_permissions(endpoint.id)
print(f"\nPermissions on {ENDPOINT_NAME}:")
for acl in (perms.access_control_list or []):
    principal = acl.user_name or acl.group_name or acl.service_principal_name or "unknown"
    for p in (acl.all_permissions or []):
        print(f"  {principal}: {p.permission_level}")
