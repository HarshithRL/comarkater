"""Graph caching latency test — measures the performance gain from compile-once.

Tests:
  1. Compilation cost      — time to compile the graph (one-time)
  2. Cache hit latency     — time to get cached graph (should be ~0ms)
  3. Greeting e2e          — full invoke through cached graph (no LLM)
  4. Greeting streaming    — stream mode through cached graph (no LLM)
  5. Multi-request reuse   — N requests on same graph, verify identical object
  6. Before vs After       — simulate old per-request compilation vs new cached
  7. Data query path       — full pipeline with real SP token (if available)
  8. Thread safety         — concurrent access from multiple threads

Usage:
    cd comarketer
    python notebooks/test_graph_caching.py
"""

import logging
import os
import sys
import time
import threading
import statistics

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
AGENT_DIR = os.path.join(PROJECT_DIR, "agent_server")
sys.path.insert(0, AGENT_DIR)

from core.graph import get_compiled_graph, _graph_lock, StateGraph, AgentState, START, END
from core.graph import route_after_supervisor, supervisor_node, greeting_node
from core.graph import agentbricks_node, format_supervisor_node
from core.config import settings
import core.graph as graph_module

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

PASS = "PASS"
FAIL = "FAIL"
results = []


def report(test_num: int, name: str, status: str, detail: str = ""):
    tag = f"[{status}]"
    line = f"  Test {test_num}: {tag:6s} {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    results.append((test_num, name, status))


def make_greeting_state(request_id="test"):
    return {
        "messages": [],
        "client_id": "test-client",
        "client_name": "TestCorp",
        "sp_id": "sp-test",
        "request_id": request_id,
        "conversation_id": f"conv-{request_id}",
        "original_question": "Hello!",
        "rewritten_question": "",
        "intent": "",
        "response_text": "",
        "response_items": [],
        "ab_summary": "",
        "ab_sql": "",
        "ab_table": "",
        "ab_insights": "",
        "ab_query_description": "",
        "follow_up_suggestions": [],
        "llm_call_count": 0,
        "ab_trace_id": "",
        "error": None,
    }


def make_config(sp_token="fake-token", thread_id="conv-test"):
    return {
        "configurable": {
            "thread_id": thread_id,
            "sp_token": sp_token,
            "sp_identity": "sp-test",
        },
    }


def make_data_query_state(question, request_id="data-test"):
    state = make_greeting_state(request_id)
    state["original_question"] = question
    return state


def compile_fresh_graph():
    """Build a graph from scratch (simulates the OLD per-request pattern)."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("agentbricks", agentbricks_node)
    graph.add_node("format_supervisor", format_supervisor_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"greeting": "greeting", "agentbricks": "agentbricks"},
    )
    graph.add_edge("greeting", END)
    graph.add_edge("agentbricks", "format_supervisor")
    graph.add_edge("format_supervisor", END)
    return graph.compile()


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════

def test_1_compilation_cost():
    """Measure one-time graph compilation time."""
    # Reset cache to force recompilation
    graph_module._compiled_graph = None

    t0 = time.perf_counter()
    g = get_compiled_graph()
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000

    ok = g is not None and compile_ms < 5000
    report(1, "Compilation cost (one-time)", PASS if ok else FAIL,
           f"{compile_ms:.1f}ms")
    return compile_ms


def test_2_cache_hit_latency():
    """Measure cached graph retrieval (should be near-zero)."""
    # Ensure graph is already compiled
    get_compiled_graph()

    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        g = get_compiled_graph()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1_000_000)  # microseconds

    avg_us = statistics.mean(times)
    p99_us = sorted(times)[int(len(times) * 0.99)]

    ok = avg_us < 100  # should be sub-100 microseconds
    report(2, "Cache hit latency (1000 calls)", PASS if ok else FAIL,
           f"avg={avg_us:.1f}us  p99={p99_us:.1f}us")
    return avg_us


def test_3_greeting_e2e():
    """End-to-end greeting through cached graph."""
    graph = get_compiled_graph()
    state = make_greeting_state("greet-001")
    config = make_config()

    t0 = time.perf_counter()
    result = graph.invoke(state, config=config)
    t1 = time.perf_counter()
    invoke_ms = (t1 - t0) * 1000

    checks = [
        result["intent"] == "greeting",
        "CoMarketer" in result.get("response_text", ""),
        len(result.get("response_items", [])) == 1,
        result.get("llm_call_count", -1) == 0,
    ]
    ok = all(checks)
    report(3, "Greeting end-to-end invoke", PASS if ok else FAIL,
           f"{invoke_ms:.1f}ms | intent={result['intent']} | items={len(result.get('response_items', []))}")
    return invoke_ms


def test_4_greeting_stream():
    """Streaming mode greeting through cached graph."""
    graph = get_compiled_graph()
    state = make_greeting_state("greet-002")
    config = make_config()

    t0 = time.perf_counter()
    events = list(graph.stream(state, config=config, stream_mode="updates"))
    t1 = time.perf_counter()
    stream_ms = (t1 - t0) * 1000

    node_names = [name for evt in events for name in evt.keys()]
    ok = node_names == ["supervisor", "greeting"] and stream_ms < 1000
    report(4, "Greeting streaming", PASS if ok else FAIL,
           f"{stream_ms:.1f}ms | nodes={node_names}")
    return stream_ms


def test_5_multi_request_reuse():
    """Verify N requests all get the exact same graph object."""
    N = 50
    graphs = [get_compiled_graph() for _ in range(N)]
    ids = set(id(g) for g in graphs)

    ok = len(ids) == 1
    report(5, f"Multi-request reuse ({N} calls)", PASS if ok else FAIL,
           f"unique graph objects={len(ids)}")


def test_6_before_vs_after():
    """Compare old per-request compilation vs new cached approach.

    Simulates 20 requests:
      - OLD: compile_fresh_graph() + invoke for each request
      - NEW: get_compiled_graph() (cached) + invoke for each request
    """
    N = 20
    graph_cached = get_compiled_graph()
    state = make_greeting_state("perf-test")
    config = make_config()

    # ── OLD: fresh compilation per request ──
    old_times = []
    for i in range(N):
        t0 = time.perf_counter()
        g = compile_fresh_graph()
        _ = g.invoke(make_greeting_state(f"old-{i}"), config=config)
        t1 = time.perf_counter()
        old_times.append((t1 - t0) * 1000)

    # ── NEW: cached graph ──
    new_times = []
    for i in range(N):
        t0 = time.perf_counter()
        g = get_compiled_graph()
        _ = g.invoke(make_greeting_state(f"new-{i}"), config=config)
        t1 = time.perf_counter()
        new_times.append((t1 - t0) * 1000)

    old_avg = statistics.mean(old_times)
    new_avg = statistics.mean(new_times)
    old_total = sum(old_times)
    new_total = sum(new_times)
    speedup = old_avg / new_avg if new_avg > 0 else float("inf")
    saved_ms = old_total - new_total

    ok = new_avg < old_avg
    report(6, f"Before vs After ({N} requests)", PASS if ok else FAIL,
           f"OLD avg={old_avg:.1f}ms  NEW avg={new_avg:.1f}ms  speedup={speedup:.1f}x  total saved={saved_ms:.0f}ms")

    # Print breakdown
    print(f"\n         {'':>8} {'Avg':>8} {'Min':>8} {'Max':>8} {'P50':>8} {'P99':>8} {'Total':>9}")
    for label, times in [("OLD", old_times), ("NEW", new_times)]:
        s = sorted(times)
        print(f"         {label:>8} {statistics.mean(s):>7.1f}ms {min(s):>7.1f}ms {max(s):>7.1f}ms "
              f"{s[len(s)//2]:>7.1f}ms {s[int(len(s)*0.99)]:>7.1f}ms {sum(s):>8.0f}ms")
    print()

    return old_avg, new_avg


def test_7_data_query_graceful_error():
    """Data query path with fake token — verifies config reaches all nodes."""
    graph = get_compiled_graph()
    state = make_data_query_state("Show top 5 email campaigns by open rate", "data-001")
    config = make_config(sp_token="fake-token-for-test")

    # Suppress expected 401 tracebacks from fake token
    logging.getLogger("agents.agentbricks").setLevel(logging.CRITICAL)
    logging.getLogger("agents.supervisor").setLevel(logging.CRITICAL)
    logging.getLogger("core.graph").setLevel(logging.CRITICAL)

    t0 = time.perf_counter()
    result = graph.invoke(state, config=config)
    t1 = time.perf_counter()

    # Restore log levels
    logging.getLogger("agents.agentbricks").setLevel(logging.INFO)
    logging.getLogger("agents.supervisor").setLevel(logging.INFO)
    logging.getLogger("core.graph").setLevel(logging.INFO)
    invoke_ms = (t1 - t0) * 1000

    checks = [
        result["intent"] == "data_query",
        result.get("error") is not None,  # expected: fake token fails
        result.get("rewritten_question", "") != "",  # fallback to original
    ]
    ok = all(checks)
    report(7, "Data query graceful error (fake token)", PASS if ok else FAIL,
           f"{invoke_ms:.0f}ms | intent={result['intent']} | error={'yes' if result.get('error') else 'no'}")
    return invoke_ms


def test_8_thread_safety():
    """Concurrent access from multiple threads."""
    # Reset cache
    graph_module._compiled_graph = None

    results_list = []
    errors = []

    def worker(idx):
        try:
            g = get_compiled_graph()
            state = make_greeting_state(f"thread-{idx}")
            config = make_config(thread_id=f"thread-conv-{idx}")
            r = g.invoke(state, config=config)
            results_list.append((idx, id(g), r["intent"]))
        except Exception as e:
            errors.append((idx, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000

    graph_ids = set(gid for _, gid, _ in results_list)
    all_greeting = all(intent == "greeting" for _, _, intent in results_list)

    ok = len(errors) == 0 and len(graph_ids) == 1 and all_greeting and len(results_list) == 10
    detail = f"{wall_ms:.0f}ms wall | threads=10 | errors={len(errors)} | unique_graphs={len(graph_ids)}"
    if errors:
        detail += f" | first_error={errors[0][1][:80]}"
    report(8, "Thread safety (10 concurrent)", PASS if ok else FAIL, detail)


# ═══════════════════════════════════════════════════════════════
#  OPTIONAL: Live data query with real SP token
# ═══════════════════════════════════════════════════════════════

def test_9_live_data_query():
    """Full data query with real SP token — requires Databricks secrets access.

    Skipped if running outside Databricks or without SP credentials.
    """
    try:
        from core.auth import resolve_client, token_provider
    except Exception:
        report(9, "Live data query (real SP)", "SKIP", "auth module not available")
        return

    # Try to get a real token for IGP
    try:
        from core.auth import SecretsLoader
        loader = SecretsLoader()
        sp_client_id = loader.get("igp-sp-client-id")
        sp_client_secret = loader.get("igp-sp-client-secret")
        sp_token = token_provider.get_token(sp_client_id, sp_client_secret)
        if not sp_token:
            raise ValueError("empty token")
    except Exception as e:
        report(9, "Live data query (real SP)", "SKIP", f"no SP token: {e}")
        return

    graph = get_compiled_graph()
    state = make_data_query_state("Show top 5 email campaigns by open rate last month", "live-001")
    state["client_id"] = "72994"
    state["client_name"] = "IGP Team"
    state["sp_id"] = "sp-igp-001"
    config = make_config(sp_token=sp_token, thread_id="live-conv-001")

    t0 = time.perf_counter()
    result = graph.invoke(state, config=config)
    t1 = time.perf_counter()
    invoke_ms = (t1 - t0) * 1000

    has_summary = bool(result.get("ab_summary"))
    has_error = bool(result.get("error"))
    ok = result["intent"] == "data_query" and (has_summary or has_error)

    report(9, "Live data query (real SP)", PASS if ok else FAIL,
           f"{invoke_ms:.0f}ms | summary={len(result.get('ab_summary',''))}ch | "
           f"table={bool(result.get('ab_table'))} | sql={bool(result.get('ab_sql'))} | "
           f"error={has_error}")
    return invoke_ms


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Graph Caching Latency Test")
    print("  Measures: compile-once savings, cache hit, e2e greeting, threading")
    print("=" * 70)
    print()

    compile_ms = test_1_compilation_cost()
    cache_us = test_2_cache_hit_latency()
    greeting_ms = test_3_greeting_e2e()
    stream_ms = test_4_greeting_stream()
    test_5_multi_request_reuse()
    old_avg, new_avg = test_6_before_vs_after()
    test_7_data_query_graceful_error()
    test_8_thread_safety()
    test_9_live_data_query()

    # ── Summary ──
    passed = sum(1 for _, _, s in results if s == PASS)
    failed = sum(1 for _, _, s in results if s == FAIL)
    skipped = sum(1 for _, _, s in results if s == "SKIP")
    total = len(results)

    print("=" * 70)
    print(f"  Results: {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")
    print()
    print(f"  Key metrics:")
    print(f"    Graph compilation (one-time):  {compile_ms:>8.1f} ms")
    print(f"    Cache hit (avg of 1000):       {cache_us:>8.1f} us  ({cache_us/1000:.3f} ms)")
    print(f"    Greeting invoke e2e:           {greeting_ms:>8.1f} ms")
    print(f"    Greeting stream e2e:           {stream_ms:>8.1f} ms")
    print(f"    Per-request (OLD avg):         {old_avg:>8.1f} ms")
    print(f"    Per-request (NEW avg):         {new_avg:>8.1f} ms")
    print(f"    Speedup:                       {old_avg/new_avg:>8.1f}x")
    print(f"    Saved per request:             {old_avg - new_avg:>8.1f} ms")
    print("=" * 70)

    if failed > 0:
        print("\n  FAILED tests:")
        for num, name, status in results:
            if status == FAIL:
                print(f"    Test {num}: {name}")
        sys.exit(1)
