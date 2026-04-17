"""Test the new structured output from CoMarketer.

Verifies that custom_outputs now contains ab_data with:
  - executive_summary, table_markdown, sql_query, insights, follow_up_suggestions
  - response_items array with typed items

Usage:
    cd comarketer
    python notebooks/test_structured_output.py
"""

import json
import time
from databricks.sdk import WorkspaceClient

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"

w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
print(f"Authenticated as: {w.current_user.me().user_name}\n")


def call(payload: dict) -> dict:
    """POST to /invocations."""
    resp = w.api_client.do(method="POST", url=f"{APP_URL}/invocations", body=payload)
    return resp if isinstance(resp, dict) else json.loads(resp)


# ─────────────────────────────────────────────────────────────
#  TEST: Data query that should produce table + summary + SQL
# ─────────────────────────────────────────────────────────────

request = {
    "input": [{"role": "user", "content": "Show me top 5 email campaigns by open rate last month"}],
    "custom_inputs": {"sp_id": "sp-igp-001"},
}

print("Sending data query to IGP...")
t0 = time.time()
response = call(request)
latency = time.time() - t0
print(f"Latency: {latency:.1f}s\n")

custom = response.get("custom_outputs", {})

# ── Metadata ──
print("=" * 70)
print("  METADATA")
print("=" * 70)
print(f"  intent:       {custom.get('intent')}")
print(f"  agent_route:  {custom.get('agent_route')}")
print(f"  client_name:  {custom.get('client_name')}")
print(f"  request_id:   {custom.get('request_id')}")

# ── Structured AB Data ──
ab_data = custom.get("ab_data", {})
print(f"\n{'=' * 70}")
print("  AB_DATA (structured)")
print(f"{'=' * 70}")
print(f"  executive_summary:  {len(ab_data.get('executive_summary', ''))} chars")
if ab_data.get("executive_summary"):
    print(f"    preview: {ab_data['executive_summary'][:200]}...")
print(f"  table_markdown:     {len(ab_data.get('table_markdown', ''))} chars")
if ab_data.get("table_markdown"):
    print(f"    preview:\n{ab_data['table_markdown'][:500]}")
print(f"  sql_query:          {len(ab_data.get('sql_query', ''))} chars")
if ab_data.get("sql_query"):
    print(f"    preview: {ab_data['sql_query'][:200]}...")
print(f"  query_description:  {ab_data.get('query_description', '')[:200]}")
print(f"  insights:           {len(ab_data.get('insights', ''))} chars")
if ab_data.get("insights"):
    try:
        ins = json.loads(ab_data["insights"])
        print(f"    columns: {ins.get('columns', [])}")
        rows = ins.get("rows", [])
        print(f"    rows: {len(rows)}")
        if rows:
            print(f"    first row preview: {str(rows[0])[:300]}")
    except Exception:
        print(f"    raw preview: {ab_data['insights'][:200]}")
print(f"  follow_ups:         {len(ab_data.get('follow_up_suggestions', []))} items")
for i, q in enumerate(ab_data.get("follow_up_suggestions", [])[:5]):
    print(f"    [{i}] {q}")

# ── Response Items ──
items = custom.get("response_items", [])
print(f"\n{'=' * 70}")
print(f"  RESPONSE_ITEMS ({len(items)} items)")
print(f"{'=' * 70}")
for i, item in enumerate(items):
    itype = item.get("type", "?")
    value = item.get("value", "")
    print(f"  [{i}] type={itype} | len={len(value)} chars")

# ── Backward compat: flat text still works ──
output = response.get("output", [{}])
text = output[0].get("content", [{}])[0].get("text", "") if output else ""
print(f"\n{'=' * 70}")
print(f"  BACKWARD COMPAT: output[0].text = {len(text)} chars")
print(f"{'=' * 70}")
print(f"  preview: {text[:200]}...")

# ── Validation ──
print(f"\n{'=' * 70}")
print("  VALIDATION")
print(f"{'=' * 70}")
checks = {
    "intent == data_query": custom.get("intent") == "data_query",
    "agent_route == agentbricks": custom.get("agent_route") == "agentbricks",
    "has ab_data": bool(ab_data),
    "has executive_summary": bool(ab_data.get("executive_summary")),
    "has table_markdown": bool(ab_data.get("table_markdown")),
    "has sql_query": bool(ab_data.get("sql_query")),
    "has insights": bool(ab_data.get("insights")),
    "has response_items": len(items) > 0,
    "response_items has text": any(i.get("type") == "text" for i in items),
    "response_items has table": any(i.get("type") == "table" for i in items),
    "response_items has sql": any(i.get("type") == "sql" for i in items),
    "flat text still works": len(text) > 50,
}

all_pass = True
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {check}")

print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print(f"{'=' * 70}")
