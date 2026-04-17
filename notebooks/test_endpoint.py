"""Test the deployed CoMarketer endpoint using Databricks SDK OAuth (U2M).

Stage 1 tests:
- Greeting routing (regex match → greeting node, zero LLM cost)
- Data query routing (non-greeting → supervisor rewrite → agentbricks)
- Per-client SP auth
- Health endpoint
"""

import json
from databricks.sdk import WorkspaceClient

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"

# Use OAuth U2M — this will open a browser for login on first run
w = WorkspaceClient(
    host="https://dbc-540c0e05-7c19.cloud.databricks.com",
    auth_type="external-browser",
)
print(f"Authenticated as: {w.current_user.me().user_name}")
print(f"App URL: {APP_URL}\n")


def test_invocation(
    test_name: str,
    payload: dict,
    expected_text: str,
    expected_client_id: str,
    expected_intent: str = None,
    expected_route: str = None,
) -> bool:
    """Send a request and validate the response."""
    print(f"--- {test_name} ---")
    try:
        resp = w.api_client.do(
            method="POST",
            url=f"{APP_URL}/invocations",
            body=payload,
        )

        data = resp if isinstance(resp, dict) else json.loads(resp)
        text = data["output"][0]["content"][0]["text"]
        custom = data.get("custom_outputs", {})

        print(f"  Client: {custom.get('client_name')} (ID: {custom.get('client_id')})")
        print(f"  Intent: {custom.get('intent', 'N/A')}")
        print(f"  Route: {custom.get('agent_route', 'N/A')}")
        print(f"  Response: {text[:150]}...")

        # Validate
        if expected_text:
            assert expected_text in text, f"Expected '{expected_text}' in response, got: {text[:100]}"
        assert custom.get("client_id") == expected_client_id, f"Expected client_id={expected_client_id}, got {custom.get('client_id')}"

        if expected_intent:
            assert custom.get("intent") == expected_intent, f"Expected intent={expected_intent}, got {custom.get('intent')}"
        if expected_route:
            assert custom.get("agent_route") == expected_route, f"Expected route={expected_route}, got {custom.get('agent_route')}"

        print(f"  PASSED\n")
        return True

    except Exception as e:
        print(f"  FAILED: {e}\n")
        return False


def test_health() -> bool:
    """Test the /ui/health endpoint."""
    print("--- Test: Health Endpoint ---")
    try:
        resp = w.api_client.do(method="GET", url=f"{APP_URL}/ui/health")
        data = resp if isinstance(resp, dict) else json.loads(resp)
        assert data["status"] == "healthy"
        assert data["agent"] == "comarketer"
        print(f"  Health: {data}")
        print(f"  PASSED\n")
        return True
    except Exception as e:
        print(f"  FAILED: {e}\n")
        return False


if __name__ == "__main__":
    results = []

    # ── Greeting tests (regex → greeting node, zero LLM cost) ──

    # Test 1: Pure greeting — IGP client
    results.append(test_invocation(
        "Test 1: Greeting (IGP)",
        {"input": [{"role": "user", "content": "Hello!"}], "custom_inputs": {"sp_id": "sp-igp-001"}},
        expected_text="CoMarketer",
        expected_client_id="72994",
        expected_intent="greeting",
        expected_route="greeting",
    ))

    # Test 2: Pure greeting — Demo fallback (unknown SP)
    results.append(test_invocation(
        "Test 2: Greeting (Demo fallback)",
        {"input": [{"role": "user", "content": "Hi"}], "custom_inputs": {"sp_id": "unknown-sp-xyz"}},
        expected_text="CoMarketer",
        expected_client_id="00000",
        expected_intent="greeting",
        expected_route="greeting",
    ))

    # ── Data query tests (supervisor → agentbricks) ──

    # Test 3: Data query — IGP (needs SP token)
    results.append(test_invocation(
        "Test 3: Data Query (IGP)",
        {"input": [{"role": "user", "content": "What was my email open rate last week?"}], "custom_inputs": {"sp_id": "sp-igp-001"}},
        expected_text="",  # Accept any response (success or auth error)
        expected_client_id="72994",
        expected_intent="data_query",
        expected_route="agentbricks",
    ))

    # Test 4: Data query — Pepe Jeans
    results.append(test_invocation(
        "Test 4: Data Query (Pepe Jeans)",
        {"input": [{"role": "user", "content": "Show me last week's email stats"}], "custom_inputs": {"sp_id": "sp-pepe-002"}},
        expected_text="",  # Accept any response
        expected_client_id="81001",
        expected_intent="data_query",
        expected_route="agentbricks",
    ))

    # ── Health ──
    results.append(test_health())

    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print(f"  Results: {passed}/{total} passed")
    print("=" * 50)
