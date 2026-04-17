"""Demo: How to call the CoMarketer endpoint and what you get back.

Run this file to see the exact request/response for both scenarios:
  1. Greeting  — "Hello!" → canned response, zero LLM cost
  2. Data query — "What was my email open rate last week?" → AgentBricks analysis

Usage:
    python demo_request_response.py
"""

import json
import time
from databricks.sdk import WorkspaceClient

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"

# Authenticate via browser OAuth (opens browser on first run)
w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
print(f"Authenticated as: {w.current_user.me().user_name}\n")


def call_endpoint(payload: dict) -> dict:
    """POST to /invocations and return parsed response."""
    resp = w.api_client.do(method="POST", url=f"{APP_URL}/invocations", body=payload)
    return resp if isinstance(resp, dict) else json.loads(resp)


def print_request_response(label: str, request: dict, response: dict):
    """Pretty-print the request and response."""
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)

    print("\n>>> REQUEST (what you send):\n")
    print(json.dumps(request, indent=2))

    print("\n>>> RESPONSE (what you get back):\n")
    print(json.dumps(response, indent=2))

    # Pull out the key fields
    output = response.get("output", [{}])
    text = output[0].get("content", [{}])[0].get("text", "") if output else ""
    custom = response.get("custom_outputs", {})

    print("\n>>> KEY FIELDS:")
    print(f"  intent:       {custom.get('intent')}")
    print(f"  agent_route:  {custom.get('agent_route')}")
    print(f"  client_id:    {custom.get('client_id')}")
    print(f"  client_name:  {custom.get('client_name')}")
    print(f"  request_id:   {custom.get('request_id')}")
    print(f"  trace_id:     {custom.get('trace_id')}")
    print(f"  response len: {len(text)} chars")
    print(f"  response preview: {text[:200]}...")
    print()


# ─────────────────────────────────────────────────────────────
#  TEST 1: Greeting
# ─────────────────────────────────────────────────────────────

greeting_request = {
    "input": [
        {"role": "user", "content": "Hello!"}
    ],
    "custom_inputs": {
        "sp_id": "sp-igp-001"       # which client (maps to CLIENT_REGISTRY)
    },
}

t0 = time.time()
greeting_response = call_endpoint(greeting_request)
greeting_latency = time.time() - t0

print_request_response("TEST 1: GREETING", greeting_request, greeting_response)
print(f"  latency: {greeting_latency:.1f}s\n")


# ─────────────────────────────────────────────────────────────
#  TEST 2: Data query
# ─────────────────────────────────────────────────────────────

data_query_request = {
    "input": [
        {"role": "user", "content": "What was my email open rate last week?"}
    ],
    "custom_inputs": {
        "sp_id": "sp-igp-001"       # IGP client
    },
}

t0 = time.time()
data_query_response = call_endpoint(data_query_request)
data_query_latency = time.time() - t0

print_request_response("TEST 2: DATA QUERY", data_query_request, data_query_response)
print(f"  latency: {data_query_latency:.1f}s\n")


# ─────────────────────────────────────────────────────────────
#  TEST 3: Data query — different client
# ─────────────────────────────────────────────────────────────

pepe_request = {
    "input": [
        {"role": "user", "content": "Show me last month's SMS campaign performance"}
    ],
    "custom_inputs": {
        "sp_id": "sp-pepe-002"      # Pepe Jeans client
    },
}

t0 = time.time()
pepe_response = call_endpoint(pepe_request)
pepe_latency = time.time() - t0

print_request_response("TEST 3: DATA QUERY (Pepe Jeans)", pepe_request, pepe_response)
print(f"  latency: {pepe_latency:.1f}s\n")


# ─────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────

print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Greeting (IGP):       {greeting_latency:.1f}s  | intent={greeting_response.get('custom_outputs', {}).get('intent')}")
print(f"  Data query (IGP):     {data_query_latency:.1f}s  | intent={data_query_response.get('custom_outputs', {}).get('intent')}")
print(f"  Data query (Pepe):    {pepe_latency:.1f}s  | intent={pepe_response.get('custom_outputs', {}).get('intent')}")
print("=" * 70)
