"""Phase A — Raw output test: same query across 3 clients.

Shows full raw response + inference time for each client.
Skips greeting tests. Skips Demo (403 — no endpoint permission).

Usage:
    cd comarketer
    python notebooks/test_phase_a.py
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


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

QUERY = "Show me top 5 email campaigns by open rate last month"

CLIENTS = [
    {"sp_id": "sp-igp-001", "name": "IGP"},
    {"sp_id": "sp-pepe-002", "name": "Pepe Jeans"},
    {"sp_id": "sp-crocs-003", "name": "Crocs"},
]

# ═══════════════════════════════════════════════════════════════
#  RUN SAME QUERY FOR ALL 3 CLIENTS
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print(f"  QUERY: {QUERY}")
print("=" * 80)

summary_rows = []

for client in CLIENTS:
    print(f"\n{'─' * 80}")
    print(f"  CLIENT: {client['name']} (sp_id={client['sp_id']})")
    print(f"{'─' * 80}")

    request = {
        "input": [{"role": "user", "content": QUERY}],
        "custom_inputs": {
            "sp_id": client["sp_id"],
            "user_name": "Analyst",
            "user_id": f"analyst-{client['name'].lower().replace(' ', '-')}",
            "conversation_id": f"conv-test-{client['sp_id']}",
            "task_type": "analytics",
        },
    }

    print(f"\n  >> Request payload:")
    print(json.dumps(request, indent=2))

    t0 = time.time()
    try:
        resp = call(request)
        latency = time.time() - t0

        print(f"\n  << Raw response ({latency:.1f}s):")
        print(json.dumps(resp, indent=2, default=str))

        # Extract key metrics
        custom = resp.get("custom_outputs", {})
        output = resp.get("output", [{}])
        text = output[0].get("content", [{}])[0].get("text", "") if output else ""
        items = custom.get("response_items", [])

        print(f"\n  ── Summary ──")
        print(f"  Inference time : {latency:.1f}s")
        print(f"  Intent         : {custom.get('intent', 'N/A')}")
        print(f"  Agent route    : {custom.get('agent_route', 'N/A')}")
        print(f"  Trace ID       : {custom.get('trace_id', 'N/A')}")
        print(f"  Request ID     : {custom.get('request_id', 'N/A')}")
        print(f"  Response length: {len(text)} chars")
        print(f"  Response items : {len(items)}")
        if items:
            for i, item in enumerate(items):
                itype = item.get("type", "?")
                val = str(item.get("value", ""))
                print(f"    [{i}] type={itype} | {len(val)} chars | preview: {val[:120]}...")
        if custom.get("error"):
            print(f"  ERROR          : {custom['error']}")

        summary_rows.append({
            "name": client["name"],
            "time": f"{latency:.1f}s",
            "items": len(items),
            "length": len(text),
            "error": custom.get("error", ""),
        })

    except Exception as e:
        latency = time.time() - t0
        print(f"\n  !! Exception after {latency:.1f}s: {e}")
        summary_rows.append({
            "name": client["name"],
            "time": f"{latency:.1f}s",
            "items": 0,
            "length": 0,
            "error": str(e)[:60],
        })

# ═══════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 80}")
print("  COMPARISON")
print(f"{'=' * 80}")
print(f"  {'Client':<15} {'Time':>8} {'Items':>6} {'Response Len':>14} {'Error'}")
print(f"  {'─' * 15} {'─' * 8} {'─' * 6} {'─' * 14} {'─' * 20}")
for row in summary_rows:
    print(f"  {row['name']:<15} {row['time']:>8} {row['items']:>6} {row['length']:>14} {row['error']}")
print(f"{'=' * 80}")
