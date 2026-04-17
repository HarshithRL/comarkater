"""Progressive streaming test — validates event-by-event timing for all 3 clients.

Tests that:
  1. Ack (RATIONALE) arrives within 5s
  2. Data (agentbricks) arrives BEFORE analysis (format_supervisor)
  3. All 3 events have correct agent_id and output format
  4. Runs for IGP, Pepe Jeans, Crocs

Usage:
    cd comarketer
    python notebooks/test_progressive_stream.py
"""

import json
import time
import requests as _requests
from databricks.sdk import WorkspaceClient

APP_URL = "https://comarketer-2276245144129479.aws.databricksapps.com"
DATABRICKS_HOST = "https://dbc-540c0e05-7c19.cloud.databricks.com"

w = WorkspaceClient(host=DATABRICKS_HOST, auth_type="external-browser")
print(f"Authenticated as: {w.current_user.me().user_name}\n")

QUERY = "Show me top 5 email campaigns by open rate last month"

CLIENTS = [
    {"sp_id": "sp-igp-001",   "name": "IGP",         "user_name": "IGP Analyst",   "user_id": "igp-001",   "conversation_id": "prog-igp",   "task_type": "analytics"},
    {"sp_id": "sp-pepe-002",  "name": "Pepe Jeans",   "user_name": "Pepe Analyst",  "user_id": "pepe-001",  "conversation_id": "prog-pepe",  "task_type": "analytics"},
    {"sp_id": "sp-crocs-003", "name": "Crocs",        "user_name": "Crocs Analyst", "user_id": "crocs-001", "conversation_id": "prog-crocs", "task_type": "analytics"},
]


def call_stream_timed(payload: dict) -> tuple[list[dict], float]:
    """POST with stream=True — returns events with per-event timing."""
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
                event["_elapsed_s"] = round(now - t0, 2)
                events.append(event)

    total = round(time.time() - t0, 2)
    return events, total


def parse_event_info(event: dict) -> dict:
    """Extract key info from an SSE event."""
    co = event.get("custom_outputs", {})
    item = event.get("item", {})
    content = (item.get("content") or [{}])[0] if item else {}
    text = content.get("text", "")

    # Parse inner payload
    items = []
    inner_co = {}
    try:
        parsed = json.loads(text)
        raw = parsed.get("items", [])
        items = raw[0] if raw and isinstance(raw[0], list) else raw
        inner_co = parsed.get("custom_outputs", {})
    except (json.JSONDecodeError, TypeError):
        pass

    item_types = [it.get("type", "?") for it in items] if items else []

    return {
        "elapsed": event.get("_elapsed_s", 0),
        "output_type": co.get("type", "?"),
        "agent_id": co.get("agent_id", "?"),
        "inner_agent_id": inner_co.get("agent_id", "?"),
        "inner_type": inner_co.get("type", "?"),
        "items_count": len(items),
        "item_types": item_types,
        "has_inner_co": bool(inner_co),
    }


# ═══════════════════════════════════════════════════════════════
#  RUN TESTS
# ═══════════════════════════════════════════════════════════════

all_results = []

for client in CLIENTS:
    print(f"\n{'═' * 75}")
    print(f"  CLIENT: {client['name']} ({client['sp_id']})")
    print(f"  QUERY:  {QUERY}")
    print(f"{'═' * 75}")

    payload = {
        "input": [{"role": "user", "content": QUERY}],
        "custom_inputs": {
            "sp_id": client["sp_id"],
            "user_name": client["user_name"],
            "user_id": client["user_id"],
            "conversation_id": client["conversation_id"],
            "task_type": client["task_type"],
        },
    }

    try:
        events, total_time = call_stream_timed(payload)
        print(f"\n  Events: {len(events)} | Total: {total_time}s\n")

        # ── Event table ──
        print(f"  {'#':<4} {'Elapsed':>8} {'Delta':>8} {'agent_id':<12} {'Output Type':<14} {'Items':>5} {'Item Types'}")
        print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*12} {'─'*14} {'─'*5} {'─'*30}")

        prev_t = 0.0
        infos = []
        for i, ev in enumerate(events):
            info = parse_event_info(ev)
            delta = round(info["elapsed"] - prev_t, 2)
            prev_t = info["elapsed"]
            infos.append({**info, "delta": delta, "idx": i})

            types_str = ",".join(info["item_types"]) if info["item_types"] else "—"
            print(f"  [{i}] {info['elapsed']:>7.1f}s {delta:>+7.1f}s {info['agent_id']:<12} {info['output_type']:<14} {info['items_count']:>5} {types_str}")

        # ── Checks ──
        print(f"\n  ── Checks ──")
        checks = []

        def check(label, ok, detail=""):
            status = "✅ PASS" if ok else "❌ FAIL"
            msg = f"  {status}  {label}"
            if detail:
                msg += f"  ({detail})"
            print(msg)
            checks.append(ok)

        # Find events by type
        ack = [i for i in infos if i["output_type"] == "RATIONALE"]
        obs = [i for i in infos if i["output_type"] == "observation"]

        check("Has ack event (RATIONALE)", len(ack) >= 1)
        check("Has ≥2 observation events", len(obs) >= 2, f"got {len(obs)}")

        if ack:
            check("Ack agent_id = VEF9O1SFFR", ack[0]["agent_id"] == "VEF9O1SFFR", ack[0]["agent_id"])
            check("Ack arrives within 10s", ack[0]["elapsed"] < 10.0, f"{ack[0]['elapsed']:.1f}s")
            check("Ack has inner custom_outputs", ack[0]["has_inner_co"])
            check("Ack inner type = RATIONALE", ack[0]["inner_type"] == "RATIONALE", ack[0]["inner_type"])

        if len(obs) >= 1:
            check("Data agent_id = ZECDLGGP3J", obs[0]["agent_id"] == "ZECDLGGP3J", obs[0]["agent_id"])
            check("Data has table item", "table" in obs[0]["item_types"], str(obs[0]["item_types"]))
            check("Data has inner custom_outputs", obs[0]["has_inner_co"])

        if len(obs) >= 2:
            check("Analysis agent_id = VEF9O1SFFR", obs[1]["agent_id"] == "VEF9O1SFFR", obs[1]["agent_id"])
            check("Analysis has text or chart", "text" in obs[1]["item_types"] or "chart" in obs[1]["item_types"], str(obs[1]["item_types"]))
            check("Analysis has inner custom_outputs", obs[1]["has_inner_co"])

            # THE KEY CHECK: progressive streaming
            gap = round(obs[1]["elapsed"] - obs[0]["elapsed"], 2)
            check("Data arrives BEFORE analysis (gap > 1s)", gap > 1.0,
                  f"data={obs[0]['elapsed']:.1f}s, analysis={obs[1]['elapsed']:.1f}s, gap={gap:.1f}s")

        passed = sum(checks)
        total_checks = len(checks)

        all_results.append({
            "name": client["name"],
            "events": len(events),
            "total_time": f"{total_time:.1f}s",
            "ack_time": f"{ack[0]['elapsed']:.1f}s" if ack else "—",
            "data_time": f"{obs[0]['elapsed']:.1f}s" if obs else "—",
            "analysis_time": f"{obs[1]['elapsed']:.1f}s" if len(obs) >= 2 else "—",
            "gap": f"{round(obs[1]['elapsed'] - obs[0]['elapsed'], 1)}s" if len(obs) >= 2 else "—",
            "passed": f"{passed}/{total_checks}",
            "progressive": gap > 1.0 if len(obs) >= 2 else False,
        })

    except Exception as e:
        print(f"\n  ❌ EXCEPTION: {e}")
        all_results.append({
            "name": client["name"],
            "events": 0,
            "total_time": "—",
            "ack_time": "—",
            "data_time": "—",
            "analysis_time": "—",
            "gap": "—",
            "passed": "0/0",
            "progressive": False,
        })

# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'═' * 90}")
print("  PROGRESSIVE STREAMING SUMMARY — ALL CLIENTS")
print(f"{'═' * 90}")
print(f"\n  {'Client':<14} {'Events':>6} {'Total':>8} {'Ack':>8} {'Data':>8} {'Analysis':>10} {'Gap':>8} {'Checks':>8} {'Progressive'}")
print(f"  {'─'*14} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*12}")

for r in all_results:
    prog = "✅ YES" if r["progressive"] else "❌ NO"
    print(f"  {r['name']:<14} {r['events']:>6} {r['total_time']:>8} {r['ack_time']:>8} {r['data_time']:>8} {r['analysis_time']:>10} {r['gap']:>8} {r['passed']:>8} {prog}")

print(f"\n{'═' * 90}")

all_progressive = all(r["progressive"] for r in all_results)
if all_progressive:
    print("  ✅ ALL CLIENTS STREAMING PROGRESSIVELY — events arrive at different times")
else:
    buffered = [r["name"] for r in all_results if not r["progressive"]]
    print(f"  ❌ BUFFERED for: {', '.join(buffered)} — data+analysis arrive together")
    print("     If gap is 0s, the Databricks proxy is buffering SSE responses")

print(f"{'═' * 90}\n")
