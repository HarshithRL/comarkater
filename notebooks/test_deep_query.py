"""Deep query test — same complex question across all SPs.

Sends the same analytical question to each client SP and prints
the full response + custom_outputs for trace analysis.
"""

import json
import time
from databricks.sdk import WorkspaceClient

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"

w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
print(f"Authenticated as: {w.current_user.me().user_name}")
print(f"App URL: {APP_URL}\n")

QUESTION = (
    "Which campaigns had the highest CTR but lowest conversions, "
    "and what could be the possible reasons? "
    "for last 3 months and for email channel"
)

CLIENTS = [
    {"name": "IGP", "sp_id": "sp-igp-001", "expected_client_id": "72994"},
    {"name": "Pepe Jeans", "sp_id": "sp-pepe-002", "expected_client_id": "81001"},
    {"name": "Crocs", "sp_id": "sp-crocs-003", "expected_client_id": "65432"},
    {"name": "Demo (default)", "sp_id": "default", "expected_client_id": "00000"},
]


def send_query(client: dict) -> dict:
    """Send the question and return full response data."""
    payload = {
        "input": [{"role": "user", "content": QUESTION}],
        "custom_inputs": {"sp_id": client["sp_id"]},
    }

    print(f"{'=' * 70}")
    print(f"  CLIENT: {client['name']} (sp_id={client['sp_id']})")
    print(f"  QUESTION: {QUESTION}")
    print(f"{'=' * 70}")

    t0 = time.time()
    resp = w.api_client.do(
        method="POST",
        url=f"{APP_URL}/invocations",
        body=payload,
    )
    latency = time.time() - t0

    data = resp if isinstance(resp, dict) else json.loads(resp)

    # Extract response text
    text = data.get("output", [{}])[0].get("content", [{}])[0].get("text", "NO TEXT")
    custom = data.get("custom_outputs", {})

    # Print trace info
    print(f"\n  --- TRACE INFO ---")
    print(f"  Request ID:   {custom.get('request_id', 'N/A')}")
    print(f"  Trace ID:     {custom.get('trace_id', 'N/A')}")
    print(f"  Client ID:    {custom.get('client_id', 'N/A')}")
    print(f"  Client Name:  {custom.get('client_name', 'N/A')}")
    print(f"  Intent:       {custom.get('intent', 'N/A')}")
    print(f"  Agent Route:  {custom.get('agent_route', 'N/A')}")
    print(f"  Latency:      {latency:.1f}s")

    # Print full response
    print(f"\n  --- FULL RESPONSE ---")
    for line in text.split("\n"):
        print(f"  {line}")

    # Validation
    print(f"\n  --- VALIDATION ---")
    checks = []

    # Check client_id
    cid_ok = custom.get("client_id") == client["expected_client_id"]
    checks.append(cid_ok)
    print(f"  Client ID match:  {'PASS' if cid_ok else 'FAIL'} (got={custom.get('client_id')}, expected={client['expected_client_id']})")

    # Check intent
    intent_ok = custom.get("intent") == "data_query"
    checks.append(intent_ok)
    print(f"  Intent=data_query: {'PASS' if intent_ok else 'FAIL'} (got={custom.get('intent')})")

    # Check route
    route_ok = custom.get("agent_route") == "agentbricks"
    checks.append(route_ok)
    print(f"  Route=agentbricks: {'PASS' if route_ok else 'FAIL'} (got={custom.get('agent_route')})")

    # Check response has content (not just an error)
    has_content = len(text) > 100 and "no SP token" not in text.lower()
    checks.append(has_content)
    print(f"  Has real content:  {'PASS' if has_content else 'FAIL'} (len={len(text)})")

    status = "PASSED" if all(checks) else "FAILED"
    print(f"\n  >>> {status} ({sum(checks)}/{len(checks)} checks)\n")

    return {"client": client["name"], "status": status, "latency": latency, "checks": checks}


if __name__ == "__main__":
    results = []
    for client in CLIENTS:
        try:
            result = send_query(client)
            results.append(result)
        except Exception as e:
            print(f"  >>> ERROR: {e}\n")
            results.append({"client": client["name"], "status": "ERROR", "latency": 0, "checks": []})

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['client']:20s} | {r['status']:8s} | {r['latency']:.1f}s")
    print("=" * 70)
