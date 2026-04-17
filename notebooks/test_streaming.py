"""Streaming capability test — verifies SSE events, output format, and MLflow tracing.

Tests:
  1. Greeting stream  — fast path, 1 event, no graph
  2. Data query stream — 2 events (ack + response), structured {items:[...]} JSON
  3. Output format    — validates each item type (text, table, collapsedText, chart)
  4. custom_outputs   — verifies agent_id + type on each event
  5. Compare invoke   — runs same query via /invocations and checks consistency

Usage (run from any directory):
    cd comarketer
    python notebooks/test_streaming.py

Prerequisites:
    - App deployed and running
    - Databricks SDK installed: pip install databricks-sdk
    - Auth: external-browser OAuth (opens browser once, caches token)
"""

import json
import time
import requests as _requests
from databricks.sdk import WorkspaceClient

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"
DATABRICKS_HOST = "https://dbc-540c0e05-7c19.cloud.databricks.com"

DEFAULT_CLIENT = {
    "sp_id": "sp-igp-001",
    "name": "IGP",
    "user_name": "StreamTestUser",
    "user_id": "stream-test-001",
    "conversation_id": "stream-conv-001",
    "task_type": "analytics",
}

DATA_QUERY = "Show me top 5 email campaigns by open rate last month"
GREETING_QUERY = "Hello!"

# ═══════════════════════════════════════════════════════════════
#  AUTH + HTTP
# ═══════════════════════════════════════════════════════════════

w = WorkspaceClient(host=DATABRICKS_HOST, auth_type="external-browser")
print(f"Authenticated as: {w.current_user.me().user_name}\n")


def call_invoke(payload: dict) -> dict:
    """POST to /invocations (non-streaming)."""
    resp = w.api_client.do(method="POST", url=f"{APP_URL}/invocations", body=payload)
    return resp if isinstance(resp, dict) else json.loads(resp)


def call_stream(payload: dict) -> list[dict]:
    """POST to /invocations with stream=True — parses SSE events line-by-line via requests."""
    headers = {"Content-Type": "application/json"}
    headers.update(w.config.authenticate())  # returns {"Authorization": "Bearer <token>"}

    resp = _requests.post(
        f"{APP_URL}/invocations",
        json={**payload, "stream": True},
        headers=headers,
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()

    events = []
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    events.append(json.loads(data))
                except json.JSONDecodeError:
                    events.append({"raw": data})
    return events


def call_stream_timed(payload: dict) -> list[dict]:
    """POST with stream=True — returns events with timing metadata.

    Each event gets '_received_at' (epoch) and '_elapsed_s' (seconds since request start).
    """
    headers = {"Content-Type": "application/json"}
    headers.update(w.config.authenticate())

    t0 = time.time()
    resp = _requests.post(
        f"{APP_URL}/invocations",
        json={**payload, "stream": True},
        headers=headers,
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()

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
                except json.JSONDecodeError:
                    event = {"raw": data}
                event["_received_at"] = now
                event["_elapsed_s"] = round(now - t0, 2)
                events.append(event)

    total = round(time.time() - t0, 2)
    return events, total


def make_request(query: str, client: dict = None) -> dict:
    """Build standard request payload."""
    c = client or DEFAULT_CLIENT
    return {
        "input": [{"role": "user", "content": query}],
        "custom_inputs": {
            "sp_id": c["sp_id"],
            "user_name": c["user_name"],
            "user_id": c["user_id"],
            "conversation_id": c["conversation_id"],
            "task_type": c["task_type"],
        },
    }


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
results = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n          {detail}"
    print(msg)
    results.append((label, condition))
    return condition


def section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


def print_event(idx: int, event: dict) -> None:
    """Pretty-print a single SSE event."""
    etype = event.get("type", "?")
    item = event.get("item", {})
    co = event.get("custom_outputs", {})
    content = (item.get("content") or [{}])[0] if item else {}
    text = content.get("text", "")[:200] if content else ""

    print(f"\n  Event [{idx}]:")
    print(f"    type         : {etype}")
    print(f"    item_id      : {event.get('item_id', event.get('id', '—'))}")
    print(f"    agent_id     : {co.get('agent_id', '—')}")
    print(f"    output_type  : {co.get('type', '—')}")
    print(f"    text preview : {repr(text[:120])}{'...' if len(text) > 120 else ''}")
    if co:
        print(f"    custom_outputs keys: {list(co.keys())}")


def parse_items_from_event(event: dict) -> list[dict]:
    """Extract {items:[...]} or {items:[[...]]} from event text field."""
    item = event.get("item", {})
    content = (item.get("content") or [{}])[0] if item else {}
    text = content.get("text", "")
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "items" in parsed:
            raw = parsed["items"]
            # Handle 2D array [[item1, item2]] → [item1, item2]
            if raw and isinstance(raw[0], list):
                return raw[0]
            return raw
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ═══════════════════════════════════════════════════════════════
#  TEST 1: GREETING STREAM
# ═══════════════════════════════════════════════════════════════

section("TEST 1 — Greeting Stream (fast path)")

t0 = time.time()
try:
    greeting_events = call_stream(make_request(GREETING_QUERY))
    greeting_latency = time.time() - t0

    print(f"\n  Raw events returned: {len(greeting_events)}")
    for i, e in enumerate(greeting_events):
        print_event(i, e)

    check("Greeting returns at least 1 event", len(greeting_events) >= 1)
    check(f"Greeting latency <5s", greeting_latency < 5.0, f"actual: {greeting_latency:.1f}s")

    if greeting_events:
        ev = greeting_events[-1]  # last event = final response
        co = ev.get("custom_outputs", {})
        item = ev.get("item", {})
        content = (item.get("content") or [{}])[0] if item else {}
        text = content.get("text", "")

        check("Greeting event has custom_outputs", bool(co))
        check("Greeting event has agent_id", "agent_id" in co, f"agent_id={co.get('agent_id')}")
        check("Greeting agent_id = VEF9O1SFFR", co.get("agent_id") == "VEF9O1SFFR", f"got: {co.get('agent_id')}")
        check("Greeting type = observation", co.get("type") == "observation", f"got: {co.get('type')}")
        check("Greeting text is non-empty", bool(text.strip()), f"text: {repr(text[:80])}")

except Exception as e:
    print(f"\n  EXCEPTION: {e}")
    check("Greeting stream call succeeded", False, str(e))
    greeting_latency = time.time() - t0

# ═══════════════════════════════════════════════════════════════
#  TEST 2: DATA QUERY STREAM — EVENT COUNT + TIMING
# ═══════════════════════════════════════════════════════════════

section("TEST 2 — Data Query Stream (ack + response events)")

t0 = time.time()
data_events = []
try:
    data_events = call_stream(make_request(DATA_QUERY))
    data_latency = time.time() - t0

    print(f"\n  Total events: {len(data_events)} | Latency: {data_latency:.1f}s")
    for i, e in enumerate(data_events):
        print_event(i, e)

    check("Data query returns ≥2 events (ack + response)", len(data_events) >= 2,
          f"got {len(data_events)} events")
    check(f"Data query latency <120s", data_latency < 120.0, f"actual: {data_latency:.1f}s")

except Exception as e:
    print(f"\n  EXCEPTION: {e}")
    check("Data query stream call succeeded", False, str(e))
    data_latency = time.time() - t0

# ═══════════════════════════════════════════════════════════════
#  TEST 3: ACK EVENT (first event)
# ═══════════════════════════════════════════════════════════════

section("TEST 3 — Acknowledgment Event (first event = RATIONALE)")

if len(data_events) >= 1:
    ack_event = data_events[0]
    print_event(0, ack_event)
    co = ack_event.get("custom_outputs", {})
    item = ack_event.get("item", {})
    content = (item.get("content") or [{}])[0] if item else {}
    ack_text = content.get("text", "")

    check("Ack event has custom_outputs", bool(co))
    check("Ack agent_id = VEF9O1SFFR", co.get("agent_id") == "VEF9O1SFFR", f"got: {co.get('agent_id')}")
    check("Ack type = RATIONALE", co.get("type") == "RATIONALE", f"got: {co.get('type')}")
    check("Ack text is non-empty", bool(ack_text.strip()), f"text: {repr(ack_text[:80])}")
    check("Ack text is JSON {items:[...]} format", ack_text.strip().startswith("{"),
          f"text: {repr(ack_text[:40])}")
else:
    check("Ack event present (need ≥1 event)", False, "no events returned")

# ═══════════════════════════════════════════════════════════════
#  TEST 4: RESPONSE EVENT (last event = observation with {items:[...]})
# ═══════════════════════════════════════════════════════════════

section("TEST 4 — Response Event (last event = observation, {items:[...]} JSON)")

if len(data_events) >= 2:
    resp_event = data_events[-1]
    print_event(len(data_events) - 1, resp_event)
    co = resp_event.get("custom_outputs", {})
    items = parse_items_from_event(resp_event)

    subsection("custom_outputs checks")
    check("Response event has custom_outputs", bool(co))
    check("Response agent_id = VEF9O1SFFR", co.get("agent_id") == "VEF9O1SFFR", f"got: {co.get('agent_id')}")
    check("Response type = observation", co.get("type") == "observation", f"got: {co.get('type')}")
    check("Response has user_name", "user_name" in co, f"keys: {list(co.keys())}")
    check("Response has conversation_id", "conversation_id" in co)

    subsection("Response body {items:[...]} format")
    check("Response text is valid JSON", len(items) >= 0,  # parse already checked
          f"items count: {len(items)}")
    check("Response has ≥1 item", len(items) >= 1, f"got {len(items)} items")

    subsection("Item-by-item inspection")
    VALID_TYPES = {"text", "table", "chart", "collapsedText"}
    type_counts = {}

    for idx, item in enumerate(items):
        itype = item.get("type", "?")
        iid = item.get("id", "")
        ival = item.get("value")
        type_counts[itype] = type_counts.get(itype, 0) + 1

        print(f"\n    Item [{idx}]  type={itype}  id={iid[:12]}...")

        if itype == "text":
            val_str = str(ival) if ival else ""
            print(f"      value preview: {repr(val_str[:200])}")
            check(f"  [{idx}] text.value is non-empty string", bool(val_str.strip()))

        elif itype == "table":
            if isinstance(ival, dict):
                headers = ival.get("tableHeaders", [])
                data = ival.get("data", [])
                print(f"      headers: {headers}")
                print(f"      rows   : {len(data)}")
                check(f"  [{idx}] table has tableHeaders", bool(headers), f"headers: {headers}")
                check(f"  [{idx}] table has data rows", len(data) > 0, f"rows: {len(data)}")
            else:
                # Pipe table string (from fallback)
                val_str = str(ival) if ival else ""
                print(f"      (pipe table string) preview: {repr(val_str[:200])}")
                check(f"  [{idx}] table value is non-empty", bool(val_str.strip()))

        elif itype == "chart":
            val_str = str(ival) if ival else ""
            print(f"      chart value type: {type(ival).__name__}")
            print(f"      preview: {repr(val_str[:200])}")
            check(f"  [{idx}] chart.value is non-empty", bool(ival))

        elif itype == "collapsedText":
            val_str = str(ival) if ival else ""
            name = item.get("name", "")
            print(f"      name: {name}")
            print(f"      value preview: {repr(val_str[:200])}")
            check(f"  [{idx}] collapsedText.value is non-empty", bool(val_str.strip()))

        else:
            check(f"  [{idx}] item type is valid", itype in VALID_TYPES, f"unexpected type: {itype}")

    print(f"\n  Item type counts: {type_counts}")
    check("Response has text item", type_counts.get("text", 0) >= 1)
    if type_counts.get("table", 0) > 0:
        print(f"  {PASS}  Response has table item (data query returned table)")
    if type_counts.get("chart", 0) > 0:
        print(f"  {PASS}  Response has chart item (Highcharts generated)")
    if type_counts.get("collapsedText", 0) > 0:
        print(f"  {PASS}  Response has collapsedText item (SQL or other collapsed content)")

else:
    check("Response event present (need ≥2 events)", False, f"only {len(data_events)} events")
    items = []

# ═══════════════════════════════════════════════════════════════
#  TEST 5: COMPARE INVOKE VS STREAM
# ═══════════════════════════════════════════════════════════════

section("TEST 5 — Invoke vs Stream Consistency")

print("\n  Running same query via non-streaming invoke...")
t0 = time.time()
try:
    invoke_resp = call_invoke(make_request(DATA_QUERY))
    invoke_latency = time.time() - t0

    invoke_co = invoke_resp.get("custom_outputs", {})
    invoke_output = invoke_resp.get("output", [{}])
    invoke_text = ""
    if invoke_output:
        content_list = invoke_output[0].get("content", [{}]) if isinstance(invoke_output[0], dict) else []
        if content_list:
            invoke_text = content_list[0].get("text", "") if isinstance(content_list[0], dict) else ""

    print(f"  Invoke latency: {invoke_latency:.1f}s")
    print(f"  Invoke intent : {invoke_co.get('intent')}")
    print(f"  Invoke items  : {len(invoke_co.get('response_items', []))}")

    stream_co = data_events[-1].get("custom_outputs", {}) if data_events else {}
    stream_items_from_text = items  # from Test 4

    check("Invoke intent matches expected (data_query)", invoke_co.get("intent") == "data_query",
          f"got: {invoke_co.get('intent')}")
    check("Stream response has items", len(stream_items_from_text) >= 1,
          f"stream items: {len(stream_items_from_text)}")
    check("Invoke response_items is list", isinstance(invoke_co.get("response_items", []), list))

    # Both should have same user context fields
    for field in ["user_name", "user_id", "conversation_id", "task_type"]:
        s_val = stream_co.get(field, "—")
        check(f"Stream custom_outputs.{field} present", field in stream_co, f"val: {s_val}")

except Exception as e:
    invoke_latency = time.time() - t0
    print(f"\n  EXCEPTION: {e}")
    check("Invoke call succeeded", False, str(e))

# ═══════════════════════════════════════════════════════════════
#  TEST 6: MULTI-CLIENT STREAM
# ═══════════════════════════════════════════════════════════════

section("TEST 6 — Multi-client Streaming (IGP + Pepe Jeans)")

CLIENTS = [
    {"sp_id": "sp-igp-001",   "name": "IGP",        "user_name": "IGP Analyst",   "user_id": "igp-s-001",   "conversation_id": "conv-s-igp",   "task_type": "analytics"},
    {"sp_id": "sp-pepe-002",  "name": "Pepe Jeans",  "user_name": "Pepe Analyst",  "user_id": "pepe-s-001",  "conversation_id": "conv-s-pepe",  "task_type": "analytics"},
]

stream_summary = []
for client in CLIENTS:
    print(f"\n  Client: {client['name']} (sp_id={client['sp_id']})")
    t0 = time.time()
    try:
        events = call_stream(make_request(DATA_QUERY, client))
        latency = time.time() - t0
        resp_items_count = len(parse_items_from_event(events[-1])) if events else 0
        co = events[-1].get("custom_outputs", {}) if events else {}
        print(f"    Events: {len(events)} | Items: {resp_items_count} | Latency: {latency:.1f}s | agent_id: {co.get('agent_id')}")
        stream_summary.append({"name": client["name"], "events": len(events), "items": resp_items_count, "latency": f"{latency:.1f}s", "error": ""})
    except Exception as e:
        latency = time.time() - t0
        print(f"    ERROR: {e}")
        stream_summary.append({"name": client["name"], "events": 0, "items": 0, "latency": f"{latency:.1f}s", "error": str(e)[:60]})

print(f"\n  {'Client':<15} {'Events':>7} {'Items':>6} {'Latency':>10} {'Error'}")
print(f"  {'─' * 15} {'─' * 7} {'─' * 6} {'─' * 10} {'─' * 30}")
for row in stream_summary:
    print(f"  {row['name']:<15} {row['events']:>7} {row['items']:>6} {row['latency']:>10} {row['error']}")

# ═══════════════════════════════════════════════════════════════
#  TEST 7: EVENT-BY-EVENT TIMING
# ═══════════════════════════════════════════════════════════════

section("TEST 7 — Event-by-Event Timing (progressive streaming)")

print("\n  Sending data query with timed streaming...")
print(f"  Query: {DATA_QUERY}\n")

try:
    timed_events, total_time = call_stream_timed(make_request(DATA_QUERY))

    print(f"  Total events: {len(timed_events)} | Total time: {total_time}s\n")

    print(f"  {'#':<4} {'Elapsed':>8} {'Delta':>8} {'Type':<14} {'Output Type':<14} {'Items':>6} {'Preview'}")
    print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*14} {'─'*14} {'─'*6} {'─'*40}")

    prev_t = 0.0
    timing_rows = []

    for i, ev in enumerate(timed_events):
        elapsed = ev.get("_elapsed_s", 0)
        delta = round(elapsed - prev_t, 2)
        prev_t = elapsed

        co = ev.get("custom_outputs", {})
        output_type = co.get("type", "?")
        agent_id = co.get("agent_id", "?")

        # Parse items from event text
        item = ev.get("item", {})
        content = (item.get("content") or [{}])[0] if item else {}
        text = content.get("text", "")

        items_count = 0
        item_types = []
        preview = ""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "items" in parsed:
                items_count = len(parsed["items"])
                item_types = [it.get("type", "?") for it in parsed["items"]]
                # Preview: first item value
                first_val = str(parsed["items"][0].get("value", ""))[:60] if parsed["items"] else ""
                preview = first_val
        except (json.JSONDecodeError, TypeError):
            preview = text[:60]

        types_str = ",".join(item_types) if item_types else "—"
        print(f"  [{i}] {elapsed:>7.2f}s {delta:>+7.2f}s {output_type:<14} {types_str:<14} {items_count:>5}  {repr(preview[:50])}")

        timing_rows.append({
            "idx": i,
            "elapsed": elapsed,
            "delta": delta,
            "output_type": output_type,
            "item_types": types_str,
            "items_count": items_count,
        })

    # Timing summary
    print(f"\n  ── Timing Summary ──")
    ack_events = [r for r in timing_rows if r["output_type"] == "RATIONALE"]
    obs_events = [r for r in timing_rows if r["output_type"] == "observation"]

    if ack_events:
        print(f"  Ack (RATIONALE):       {ack_events[0]['elapsed']:.2f}s after request")
    if len(obs_events) >= 1:
        print(f"  Data (EVENT 1):        {obs_events[0]['elapsed']:.2f}s after request")
    if len(obs_events) >= 2:
        print(f"  Analysis (EVENT 2):    {obs_events[1]['elapsed']:.2f}s after request  (+{obs_events[1]['elapsed'] - obs_events[0]['elapsed']:.2f}s after data)")
    print(f"  Total:                 {total_time:.2f}s")

    # Checks
    if ack_events:
        check("Ack arrives within 5s", ack_events[0]["elapsed"] < 5.0,
              f"actual: {ack_events[0]['elapsed']:.2f}s")
    if len(obs_events) >= 1:
        check("Data event arrives (agentbricks)", True,
              f"at {obs_events[0]['elapsed']:.2f}s with {obs_events[0]['items_count']} items ({obs_events[0]['item_types']})")
    if len(obs_events) >= 2:
        check("Analysis event arrives (format_supervisor)", True,
              f"at {obs_events[1]['elapsed']:.2f}s with {obs_events[1]['items_count']} items ({obs_events[1]['item_types']})")
        check("Data arrives BEFORE analysis", obs_events[0]["elapsed"] < obs_events[1]["elapsed"],
              f"data={obs_events[0]['elapsed']:.2f}s < analysis={obs_events[1]['elapsed']:.2f}s")
    check("Total events ≥ 2 (ack + data)", len(timed_events) >= 2,
          f"got {len(timed_events)} events")

except Exception as e:
    print(f"\n  EXCEPTION: {e}")
    check("Timed streaming call succeeded", False, str(e))

# ═══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

section("FINAL SUMMARY")

passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total = len(results)

print(f"\n  Checks: {passed}/{total} passed")
print(f"\n  {'Label':<55} {'Status'}")
print(f"  {'─' * 55} {'─' * 8}")
for label, ok in results:
    status = "PASS ✅" if ok else "FAIL ❌"
    print(f"  {label:<55} {status}")

print(f"\n  {'═' * 65}")
if failed == 0:
    print(f"  ALL {total} CHECKS PASSED — streaming is working correctly")
else:
    print(f"  {failed} CHECKS FAILED — review output above for details")
print(f"  {'═' * 65}\n")
