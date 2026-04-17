# 2. System Architecture

## 2.1 Architecture type

**Hierarchical multi-agent system on LangGraph** with a single supervisor node routing to a small set of leaf nodes, one of which (`campaign_insight_agent`) is a 5-phase async subgraph embedded as one LangGraph node (its internal phases are **not** LangGraph nodes — they are sequential async method calls inside `CampaignInsightAgent.run`).

- Central orchestrator: `core/graph.py` compiles a `StateGraph(AgentState)` with a `CheckpointSaver` (Lakebase; `InMemorySaver` fallback).
- Control transfer: supervisor uses `Command(goto=..., update=...)` (LangGraph routing primitive), not conditional edges.
- Streaming: SSE via MLflow `ResponsesAgent` `predict_stream` + Flask/ASGI middleware that disables buffering.

---

## 2.2 Compiled graph topology

```
START
  │
  ▼
supervisor_classify  (agents/supervisor.py:supervisor_node)
  │ Command(goto=...)
  ├──► greeting            ──► END
  ├──► clarification       ──► END
  ├──► out_of_scope        ──► END
  └──► campaign_insight_agent   (async, 5 internal phases)
            │
            ▼
       supervisor_synthesize  (core/graph.py — wraps supervisor/synthesizer.py)
            │
            ▼
           END
```

Legacy nodes (`planner_node`, `genie_data_node`, `genie_analysis_node`, `genie_worker_node`, `synthesizer_node`, `insight_agent_node`) are importable but **not added to the compiled graph**.

---

## 2.3 Core components

### Supervisor layer
- **supervisor_classify** — Layer 1 regex greeting detection; Layer 2 LLM classification + query rewrite.
- **supervisor_synthesize** — Consumes `subagent_output`, invokes supervisor LLM with `RESPONSE_FORMAT_SCHEMA` (strict JSON) to produce `response_items` (`text | table | chart | collapsedText`).

### CampaignInsightAgent subgraph (Phase 1–5, all inside one LangGraph node)
1. **AdaptivePlanner** — intent → dimension-constrained `PlanStep`s.
2. **ReActExecutor** — reason → Genie call → observe → evaluate, ≤3 iters, 120s total.
3. **Interpreter** — deterministic + LLM pattern classification over `DisplayTable`.
4. **Recommender** — 2–3 action items per pattern (Apply / Avoid / Explore).
5. **OutputBuilder** — emits `SubagentOutput`.

Helpers: `DimensionClassifier`, `DimensionValidator`, `TableAnalyzer`, `TableBuilder`, `ChartBuilder`, `ToolHandler`, `Reflector`.

### Tooling layer
- **Genie REST client** (`agents/genie_client.py`) — sync httpx; status polling.
- **MCP** (CLAUDE.md says "migrating") — **NOT implemented**.

### Memory layer
| Layer | Storage | Class |
|-------|---------|-------|
| Working | in-memory `AgentState` dict | — |
| STM (session) | Lakebase | `databricks_langchain.CheckpointSaver` |
| LTM (client) | Lakebase vector store | `databricks_langchain.DatabricksStore` via `LTMManager` |
| Analytical cache | — | **MISSING** |

### Prompt layer
`prompts/` + `agents/campaign_insight/prompts/`. Supervisor prompt is 1,063 LOC and governs routing, response structure, Highcharts 11.x, termination rules.

### Parsing/validation
`parsers/` (table truncation, filter parsing, client formatting). `core/models.py` provides the strict JSON schema enforced on supervisor output.

### UI layer
Flask app mounted on `/ui` via `WsgiToAsgi`. Functional chat with SSE.

---

## 2.4 Component dependency graph

```
start_server.py
  └─► agent.py (CoMarketerAgent)
        ├─► core/config.py
        ├─► core/auth.py           ──► databricks.sdk.WorkspaceClient
        ├─► core/graph.py
        │     ├─► agents/supervisor.py        ──► prompts/supervisor_prompt.py
        │     ├─► agents/greeting.py
        │     ├─► agents/clarification.py
        │     ├─► agents/campaign_insight/agent.py
        │     │     ├─► adaptive_planner / executor / interpreter / recommender / output_builder
        │     │     ├─► tool_handler ──► agents/genie_client.py (httpx)
        │     │     ├─► chart_builder / table_builder / table_analyzer
        │     │     └─► domain_knowledge/*.yaml
        │     └─► supervisor/synthesizer.py   ──► prompts/supervisor_prompt.py
        ├─► memory/ltm_manager.py   ──► databricks_langchain.DatabricksStore
        ├─► memory/stm_trimmer.py, extractors, context_formatter
        ├─► parsers/*
        ├─► core/tracing.py         ──► mlflow
        └─► ui/routes.py (Flask) ──► agent.predict / predict_stream
```

---

## 2.5 Key architectural choices (fact + source)

| Choice | Location | Note |
|--------|----------|------|
| `Command(goto=...)` routing | `agents/supervisor.py` | Instead of conditional edges |
| Async subgraph inside sync LangGraph node | `agent.py:~1457`, `core/graph.py:~113` | Uses `asyncio.run()` / `loop.run_in_executor`; documented design choice |
| Single source table + RLS | `CLAUDE.md`; genie client never appends WHERE | Unity Catalog enforces cid |
| Per-client OBO SP auth | `core/auth.py` | Secrets in `agent-sp-credentials` scope |
| Strict JSON schema on supervisor output | `core/models.py:RESPONSE_FORMAT_SCHEMA` | `strict: True` |
| Single Genie space shared by clients | `GENIE_SPACE_ID` env var | RLS ensures isolation |
| Highcharts 11.x (not 12.x) | `prompts/supervisor_prompt.py` visualization section | Explicit v11 rules |

---

## 2.6 Mismatches with `CLAUDE.md`

| CLAUDE.md | Reality |
|-----------|---------|
| Haiku (supervisor) + Sonnet (reasoning) | Both use `databricks-gpt-5-2` via AI Gateway (`ChatOpenAI`) |
| "Hierarchical supervisor → Insight Agent subgraph" with 5 supervisor nodes + 6 insight nodes | Supervisor is 1 node + 3 terminal leaves + 1 synthesizer; insight subgraph collapses its 5 phases into 1 LangGraph node |
| Genie managed MCP | REST API only |
| 4-layer memory | 3 layers (analytical cache missing) |
| Stats engine, anomaly scanner | Not present |
