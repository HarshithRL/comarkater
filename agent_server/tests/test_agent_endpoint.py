"""Simple agent endpoint test — send a query, print the raw response.

Usage:
    python agent_server/tests/test_agent_endpoint.py \\
        --endpoint https://<app-url>.databricksapps.com \\
        --token "$TOKEN" \\
        --query "how is my email CTR last week"

Or via env vars:
    export COMARKETER_ENDPOINT=https://<app-url>.databricksapps.com
    export COMARKETER_TOKEN=<bearer-token>
    python agent_server/tests/test_agent_endpoint.py --query "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests


def hit(endpoint: str, token: str | None, query: str) -> dict:
    url = endpoint.rstrip("/") + "/invocations"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "input": [{"role": "user", "content": query}],
        "stream": False,
        "custom_inputs": {
            "sp_id": "sp-igp-001",
            "user_name": "IGP Test",
        },
    }
    r = requests.post(url, json=payload, headers=headers, timeout=300)
    print(f"HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError:
        return {"raw_text": r.text}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=os.environ.get("COMARKETER_ENDPOINT"))
    p.add_argument("--token", default=os.environ.get("COMARKETER_TOKEN") or os.environ.get("DATABRICKS_TOKEN"))
    p.add_argument("--query", required=True)
    args = p.parse_args()

    if not args.endpoint:
        print("ERROR: --endpoint or COMARKETER_ENDPOINT is required", file=sys.stderr)
        sys.exit(2)

    response = hit(args.endpoint, args.token, args.query)
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
