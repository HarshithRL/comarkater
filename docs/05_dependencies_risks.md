# 05 — Dependencies & Risks

*See `01_structure.md` for file inventory, `04_code_audit.md` for quality defects referenced here.*

## External Dependencies

From `pyproject.toml`:

| Package | Version | Role |
|---------|---------|------|
| `mlflow[databricks]` | ≥3.1.3 | Tracing, ResponsesAgent runtime, feedback assessments, experiment. |
| `langgraph` | ≥0.3.0 | Graph, Send API, CheckpointSaver. |
| `langchain-core` | ≥0.3.0 | Messages, runnables, config wiring. |
| `databricks-langchain[memory]` | ≥0.5.0 | `DatabricksStore` (LTM), `ChatDatabricks` (migration target). |
| `databricks-sdk` | ≥0.50.0 | Workspace client for secrets + SQL warehouse. |
| `flask` | ≥3.0.0 | UI blueprint on `/ui`. |
| `asgiref` | ≥3.7.0 | `WsgiToAsgi` adapter to mount Flask under AgentServer (FastAPI). |
| `httpx` | ≥0.27.0 | Genie REST + AgentBricks HTTP. |
| `pydantic` | ≥2.0.0 | Settings, typed models. |

Transitive concerns:
- `langchain_openai.ChatOpenAI` is currently used against the Databricks AI Gateway (`base_url` override). Migration to `databricks-langchain.ChatDatabricks` would drop the OpenAI SDK as a live dep.

## External Systems

| System | Criticality | Single point of failure? |
|--------|:----------:|:------:|
| Databricks Genie space `01f1369ea0a11cd9a386669de0083fd8` | Critical — all data queries | Yes |
| AI Gateway `2276245144129479.ai-gateway...` | Critical — all LLM calls | Yes |
| Lakebase `agentmemory` (Postgres) | High — STM checkpoints + LTM | Yes (STM loss = no conversation memory) |
| SQL Warehouse `64769e0e91e002b8` | Medium — trace-table writes only | No (degrades gracefully via `_sql_warned`) |
| Unity Catalog `channel.gold_channel.campaign_details` | Critical — single source of truth, RLS-enforced | Yes |
| MLflow experiment | Medium — tracing + feedback | No (feedback fails 500, runtime unaffected) |

## Internal Coupling

### God-object: `agent_server/agent.py` (1,441 LOC)

Imports into nearly every other module. Houses the pre-graph and post-graph logic that touches memory, auth, streaming, graph compilation, and error recovery. Changes here are high-blast-radius.

### Hot Spot: `core/state.py`

Imported by virtually every node. Any schema change must be reflected in `init_agent_state()` and every consumer. Flat-TypedDict choice makes refactors ergonomic but offers no compile-time safety.

### Hot Spot: `core/graph.py`

Module-level `_compiled_graph` cache means:
- First request pays compile cost.
- Graph topology changes require process restart (acceptable for a Databricks App, breaking for hot-reload dev).
- `_graph_lock` guards compile — contention only on cold start.

### Hot Spot: `agents/agentbricks.py` + `agents/genie_*.py`

Data-access surface split across 5 files with overlapping helpers (see `04_code_audit.md §5`). Any Genie API contract change touches multiple files.

## Bottlenecks

| Location | Type | Impact |
|----------|------|--------|
| `_fan_out_to_workers` | Hard-coded `MAX_WORKERS = 6` | Genie rate-limits at ~5–6 concurrent conversations; exceeding this causes 429s. Legacy path only. |
| `format_supervisor_node` | Single LLM call blocks finalization | ~1–3 s latency floor on every simple-path request. |
| `genie_data_node` (simple) / per-step `insight_agent` (complex) | Synchronous Genie REST inside LangGraph nodes | Sequential steps compound latency linearly; no intra-step parallelism on the complex path. |
| `asyncio.run()` inside sync LangGraph nodes `[CLAUDE.md]` | Documented design decision | Blocks the ASGI worker for the entire async call. Under concurrency, workers saturate before external rate limits kick in. |
| SSE long-lived connections | One FastAPI worker per active stream | Without sized worker pool, a burst of streaming users starves the pool. |

## Scalability Risks

1. **LTM semantic search latency** — every request loads LTM context before graph invocation; `databricks-gte-large-en` embedding call + Postgres vector search scales with user memory size.
2. **Checkpoint bloat** — `format_supervisor_node` explicitly clears large fields before write, but `plan`, `step_results`, and `worker_results` are unbounded reducer-accumulated lists.
3. **SSE proxy timeout** — addressed by `SSENoBufferMiddleware`, but the fix is fragile (must stay first in the ASGI chain; easy to break by adding `BaseHTTPMiddleware`).
4. **Graph recompile on cold start** — combined with Lakebase connection init, cold-start TTFB likely dominated by infra warmup.

## Security Surface

- **OBO auth** (`core/auth.py`) — secrets fetched from scope `agent-sp-credentials`, token minted per request (or per SP-pair). `REQUIRE_PER_REQUEST_SP=false` in `databricks.yml` → default SP fallback when no per-request SP supplied.
- **RLS** — `[CLAUDE.md]` guarantee: enforced at Unity Catalog layer, agent NEVER adds `WHERE cid` manually. No code path found that injects a `cid` WHERE-clause into generated SQL — consistent with the guarantee.
- **UI whitelist** — `ALLOWED_SP_IDS` in `ui/routes.py` limits who can drive `/ui/chat` to three known clients + `default`.
- **Parameterized SQL** `[CLAUDE.md]` rule — trace-table writer in `core/tracing.py` needs verification it parameterizes inputs (not re-read here).
- **Prompt injection** — `supervisor_prompt.py` + `genie_analysis_node` feed user text directly; no explicit guardrail or output validator beyond `RESPONSE_FORMAT_SCHEMA`.

## Compliance / Invariants `[CLAUDE.md]`

| Invariant | Enforcement |
|-----------|------|
| No currency symbols in output | Prompt-level only (no post-validator). |
| Never display "genie"/"Databricks" to users | Prompt-level only. |
| Numbers must trace back to data | Currently only LLM-guarded — `stats_engine` would enforce structurally (see `04_code_audit.md §2`). |
| One file = one responsibility, ≤200 LOC | Six files currently violate (see `04_code_audit.md §1`). |
| Every module has tests | Single test file present. |
