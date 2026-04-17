"""Quick test — greeting streaming only. Validates 2D array + custom_outputs format."""

import json
import time
import requests as _requests
from databricks.sdk import WorkspaceClient

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"
DATABRICKS_HOST = "https://dbc-540c0e05-7c19.cloud.databricks.com"

w = WorkspaceClient(host=DATABRICKS_HOST, auth_type="external-browser")
print(f"Authenticated as: {w.current_user.me().user_name}\n")

payload = {
    "input": [{"role": "user", "content": "Hello!"}],
    "custom_inputs": {
        "sp_id": "sp-igp-001",
        "user_name": "TestUser",
        "user_id": "test@company.com",
        "conversation_id": "greeting-test-001",
        "task_type": "general",
        "thread_id": "thread-greeting-001",
    },
    "stream": True,
}

headers = {"Content-Type": "application/json"}
headers.update(w.config.authenticate())

print("=" * 70)
print("  GREETING STREAM TEST")
print("=" * 70)

t0 = time.time()
resp = _requests.post(
    f"{APP_URL}/invocations",
    json=payload,
    headers=headers,
    stream=True,
    timeout=30,
)
print(f"\n  HTTP {resp.status_code} | Content-Type: {resp.headers.get('content-type')}")

if not resp.ok:
    print(f"  ERROR: {resp.text[:500]}")
    exit(1)

events = []
for line in resp.iter_lines():
    if not line:
        continue
    line = line.decode("utf-8") if isinstance(line, bytes) else line
    if line.startswith("data:"):
        data = line[5:].strip()
        if data and data != "[DONE]":
            now = time.time()
            try:
                event = json.loads(data)
                event["_elapsed"] = round(now - t0, 2)
                events.append(event)
            except json.JSONDecodeError:
                print(f"  RAW: {data[:200]}")

latency = round(time.time() - t0, 2)
print(f"  Events: {len(events)} | Latency: {latency}s\n")

PASS = "PASS"
FAIL = "FAIL"
checks = []


def check(label, ok, detail=""):
    status = f"  {'✅' if ok else '❌'} {PASS if ok else FAIL}  {label}"
    if detail:
        status += f"\n          {detail}"
    print(status)
    checks.append(ok)


for i, ev in enumerate(events):
    print(f"\n  ── Event [{i}] (at {ev.get('_elapsed', '?')}s) ──")

    # 1. Check SSE wrapper custom_outputs
    co = ev.get("custom_outputs", {})
    print(f"  SSE custom_outputs: {json.dumps(co, indent=4)}")
    check("SSE has agent_id", "agent_id" in co, f"got: {co.get('agent_id')}")
    check("SSE agent_id = ZECDLGGP3J", co.get("agent_id") == "ZECDLGGP3J", f"got: {co.get('agent_id')}")
    check("SSE type = observation", co.get("type") == "observation", f"got: {co.get('type')}")

    # 2. Parse text payload
    item = ev.get("item", {})
    content = (item.get("content") or [{}])[0] if item else {}
    text = content.get("text", "")

    print(f"\n  Text payload (first 300 chars):")
    print(f"  {repr(text[:300])}")

    # 3. Check JSON structure
    try:
        parsed = json.loads(text)
        check("Text is valid JSON", True)

        # 4. Check items is 2D array
        items_raw = parsed.get("items", [])
        check("Has 'items' key", "items" in parsed)
        check("items is array", isinstance(items_raw, list), f"type: {type(items_raw).__name__}")
        if items_raw:
            check("items[0] is array (2D)", isinstance(items_raw[0], list), f"type: {type(items_raw[0]).__name__}")
            if isinstance(items_raw[0], list) and items_raw[0]:
                first_item = items_raw[0][0]
                check("items[0][0] has 'type'", "type" in first_item, f"keys: {list(first_item.keys())}")
                check("items[0][0] has 'id'", "id" in first_item)
                check("items[0][0] has 'value'", "value" in first_item)
                check("items[0][0].type = 'text'", first_item.get("type") == "text", f"got: {first_item.get('type')}")
                print(f"\n  Item value preview: {repr(str(first_item.get('value', ''))[:150])}")

        # 5. Check custom_outputs INSIDE payload
        inner_co = parsed.get("custom_outputs", {})
        print(f"\n  Inner custom_outputs: {json.dumps(inner_co, indent=4)}")
        check("Payload has 'custom_outputs' key", "custom_outputs" in parsed)
        check("Inner agent_id = ZECDLGGP3J", inner_co.get("agent_id") == "ZECDLGGP3J", f"got: {inner_co.get('agent_id')}")
        check("Inner type = observation", inner_co.get("type") == "observation", f"got: {inner_co.get('type')}")
        check("Inner has user_name", "user_name" in inner_co, f"val: {inner_co.get('user_name')}")
        check("Inner has conversation_id", "conversation_id" in inner_co, f"val: {inner_co.get('conversation_id')}")
        check("Inner has thread_id", "thread_id" in inner_co, f"val: {inner_co.get('thread_id')}")

    except json.JSONDecodeError:
        check("Text is valid JSON", False, f"raw: {text[:100]}")

print(f"\n{'=' * 70}")
passed = sum(checks)
total = len(checks)
print(f"  {passed}/{total} checks passed | Latency: {latency}s")
if passed == total:
    print("  ALL CHECKS PASSED")
else:
    print(f"  {total - passed} CHECKS FAILED")
print(f"{'=' * 70}\n")
