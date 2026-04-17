# 06 — Unknowns & Assumptions

Items requiring human confirmation or deeper investigation before relying on them.

## Confirmed by This Analysis (previously unknown in the plan)

- `feedback.py` destination → **resolved**: writes to **MLflow trace assessments** via `mlflow.log_feedback()` (not Delta / not Lakebase).
- Whether `genie_*.py` files are duplicates → **resolved**: they are distinct (client vs. simple-path node vs. analysis LLM vs. legacy worker vs. AgentBricks alternative). Documented in `03` + `04`.
- Stats engine / anomaly scanner / filter persistence / analytical cache existence → **confirmed absent** via grep (`04_code_audit.md §2`).
- Exact `AgentState` field list → **resolved**: documented in `03_agent_architecture.md`.

## Still Unknown — Require Validation

### A. Registry fan-out

Four registry files live in the repo:
- `agents/registry.py`
- `agents/capability_registry.py`
- `agents/agent_model.py`
- `capabilities/registry.py`

**Question**: Are these one logical concept split across layers, or overlapping/competing abstractions? `04_code_audit.md §5` flags this as potential duplication — verifying requires reading each file's public API and grepping import sites.

### B. `bundles/netcore-insight-agent/`

A separate Databricks asset bundle lives under `bundles/`. The main `databricks.yml` at root references none of it. Possibilities:
1. **Legacy** — pre-unification bundle kept for reference.
2. **Active** — a separately-deployed insight-agent service (not the embedded subgraph).
3. **Scaffold** — future standalone deployment target.

**Impact**: If #2, there may be runtime surface not covered by this analysis.

### C. Intent taxonomy reconciliation

Code uses 4 intent values. CLAUDE.md specifies 5 different values (uppercase enum style). **Neither is a subset of the other.**

**Question**: Which is canonical? Is the code drift-forward (CLAUDE.md stale) or drift-backward (code stale)?

### D. Streaming phase coverage

CLAUDE.md: 8 SSE phases. Code demonstrably emits `node_started`/`progress` events and the `format_supervisor` lifecycle. Phases `thought`, `plan`, `data_table`, `chart`, `analysis`, `recommendations`, `complete` — each needs a grep to confirm emission points.

### E. `AgentBricks` path activation

`agents/agentbricks.py::agentbricks_node` is defined but not registered in `core/graph.py`. **Question**: Is it dead code awaiting a graph-wiring cutover, or a parallel path invoked from elsewhere (e.g. directly from `agent.py`)?

### F. Notebook role

17 scripts in `notebooks/`. Used as diagnostics? Integration tests? User-facing runbooks? Whether any are load-bearing for deployment verification is unclear.

### G. `core/tracing.py` SQL parameterization

CLAUDE.md: "ALL SQL must be parameterized". Trace-table writer was not fully read in this pass — need a targeted inspection to confirm it uses placeholders, not f-strings.

### H. Lakebase STM checkpointer wiring

`core/graph.py::get_compiled_graph(checkpointer=None)` accepts a checkpointer but the caller site in `agent.py` (line unknown) must be verified to actually pass a `CheckpointSaver`. Without it, STM persistence is silently disabled.

### I. Supervisor prompt vs. code intent drift

`prompts/supervisor_prompt.py` (1,062 LOC) is the source of truth for classification behavior. Its intent set and the 4-value enum in `core/graph.py` / `agents/supervisor.py` must match — not verified at character level.

### J. LangGraph version match

`pyproject.toml` pins `langgraph≥0.3.0`. The `Command` + `Send` + `get_stream_writer` APIs used assume a specific minor. If deployed against a newer langgraph that changed these APIs, runtime breaks. **Question**: Is there a max-pin policy or CI check?

## Assumptions Made in Docs

All documentation assumes:

1. `genie_data_node` (REST) is the currently-live Genie path. If production has been flipped to `agentbricks_node`, `02_workflow.md` and `03_agent_architecture.md` are outdated.
2. `databricks-gpt-5-2` is the actual model answering requests today (pulled from `databricks.yml`). If the App overrides env vars post-deploy, the real model may differ.
3. The main Databricks App bundle (`databricks.yml` at repo root) is the production deployment target; `bundles/netcore-insight-agent/` is not.
4. `supervisor_prompt.py` has not been regenerated since the 4-value intent enum was last edited (i.e., their drift, if any, is unintentional).
5. The `capabilities/` subgraph is invoked via `capabilities/registry.py` keyed on `current_capability`, mapping to `insight_agent`. Other capabilities may exist but were not enumerated.

## Suggested Validation Order

1. Resolve §C (intent taxonomy) — one grep + one CLAUDE.md reconciliation.
2. Resolve §E (`agentbricks_node` wiring) — decides whether the audit in `04 §4` is urgent.
3. Resolve §H (STM checkpointer) — silent data loss if misconfigured.
4. Resolve §A (registry files) — affects refactor plan.
5. Remaining items are confidence-improvers, not correctness-blockers.
