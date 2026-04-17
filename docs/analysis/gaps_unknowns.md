# 7. Gaps, Missing Pieces, and Unknowns

Facts against `CLAUDE.md` intent. Classified as **MISSING**, **PARTIAL**, or **UNKNOWN (needs verification)**.

---

## 7.1 MISSING — claimed in spec, absent in code

| Component | Claim | Evidence of absence |
|-----------|-------|---------------------|
| **Stats engine (11 derived metrics)** | "ALL derived metrics (CTR, CVR, etc.) computed in Python" | No module computes these; derived metrics arrive pre-computed from Genie SQL. |
| **Anomaly scanner node** | Listed as dedicated insight-agent node | No file, no graph node. |
| **Analytical cache** | "Analytical Cache (Delta, daily refresh)" — 4th memory layer | No cache module, no scheduled job, no Delta table wiring. |
| **Filter persistence** | "Filter state persisted across requests" | No filter store beyond in-state fields. |
| **Error recovery chain** | "Error recovery chain" | Only `genie_retry_count`; no retry/backoff/fallback ladder. |
| **Genie managed MCP** | "Migrating from REST API to managed MCP" | Only REST (`genie_client.py`) implemented. |
| **5-intent taxonomy** (`DATA_ANALYSIS`, `DOMAIN_KNOWLEDGE`, `AMBIGUOUS`, `CONTINUATION`, `OUT_OF_SCOPE`) | Spec lists 5 | Code has `SIMPLE_QUERY`, `COMPLEX_QUERY`, `CLARIFICATION`, implicit `OUT_OF_SCOPE` — different taxonomy |
| **Haiku/Sonnet split** | Haiku (supervisor) + Sonnet (reasoning) | Both paths use `databricks-gpt-5-2` via `ChatOpenAI` on AI Gateway. |
| **Distinct streaming phases "plan" and "complete"** | 8 phases | 6 explicit phases; "plan" is folded into insight node, "complete" is implicit (only TRACE_DONE). |
| **Insight Agent as LangGraph subgraph with 6 named nodes** | query_planner → genie_execution → validator → stats_engine → response_builder → anomaly_scanner | Collapsed into 1 LangGraph node (`campaign_insight_agent`) whose 5 phases are method calls, not nodes. Stats engine + anomaly scanner are absent entirely. |
| **Per-node model routing (Haiku vs Sonnet)** | Supervisor tasks Haiku; reasoning Sonnet | No per-node model selection code. |
| **Session (Lakebase checkpoint) + Client (Delta) + Analytical (Delta) as separate layers** | 4 memory layers | LTM uses Lakebase vector store, not Delta; no analytical cache at all. |

## 7.2 PARTIAL

| Component | Status |
|-----------|--------|
| **Intent classification (4 of 5)** | Code has 4 coarse intents; no `AMBIGUOUS`/`CONTINUATION`/`DATA_ANALYSIS` naming. |
| **Streaming (6 of 8 phases)** | `thought`, `progress`, `data_table`, `chart`, `analysis`, `recommendations` — explicit. `plan` and `complete` — implicit. |
| **LTM 3-layer memory** | Working + STM (Lakebase) + LTM (Lakebase vector) — 3 of claimed 4 layers. |
| **Tracing** | `@mlflow.trace` + `autolog` + custom spans present. Custom Delta trace table initialized but UNKNOWN whether consistently written to on every request. |
| **Table compression / truncation** | `parsers/table_truncator.py` exists; integration points not fully audited. |
| **Reflector** | Present and invoked, but impact on final output vs. straight Interpreter/Recommender path UNKNOWN. |

## 7.3 UNKNOWN — needs verification before trusting

| Topic | Open question |
|-------|---------------|
| Bundle duplication | Are both `databricks.yml` and `bundles/netcore-insight-agent/databricks.yml` deployed or is one stale? |
| `supervisor/domain_context.py` | Is it imported/used anywhere at runtime? |
| `parsers/filters.py` | Exact semantics (filter parsing vs. filter storage) — not confirmed. |
| UI templates | `index.html` and any JS/CSS assets not fully enumerated. Chart rendering library on client side UNKNOWN. |
| Trace table schema | `core/tracing.py` creates a Delta trace table, but UNKNOWN whether writes happen per request or only at init. |
| Tool registry | `agent_server/tools/` has Genie wrappers — whether any code path calls them (vs. direct `genie_client`) is UNKNOWN. |
| `notebooks/test_graph_caching.py` claim of graph compilation caching | Caching strategy in runtime (singleton per process?) not explicitly verified. |
| SSE `complete` phase | No code emits an explicit "complete" event type; whether this counts as a gap for clients depends on contract with front-end. |

## 7.4 Doc drift

- `docs/` directory contains 6 markdown files that appear to be older architectural scaffolding. `CLAUDE.md` is authoritative. Recommend deleting or archiving.
- `workflow.md` describes "Phase 3" lifecycle only; other phases referenced in CLAUDE.md not documented.
