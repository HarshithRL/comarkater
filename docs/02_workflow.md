# 02 — Workflow (End-to-End Execution)

*For file locations see `01_structure.md`. For agent responsibilities see `03_agent_architecture.md`.*

## Request Surfaces

Two entry paths hit the same `CoMarketerAgent`:

1. **Native AgentServer** — `POST /invocations` / `/stream` (from MLflow ResponsesAgent protocol).
2. **Flask UI** — `POST /ui/chat` builds a `ResponsesAgentRequest` and calls `agent.predict()` or `agent.predict_stream()` directly.

Both traverse the same `SSENoBufferMiddleware` (raw ASGI) that injects `x-accel-buffering: no` + `cache-control: no-cache, no-transform` on any `text/event-stream` response to prevent the Databricks App proxy from buffering.

## End-to-End Sequence

```
HTTP request
  → SSENoBufferMiddleware
  → AgentServer → CoMarketerAgent.predict[_stream]
  → [pre-graph]
      • resolve cid/sp_id/user_id from custom_inputs + auth
      • load LTM (memory/ltm_manager) + serialize STM (memory/stm_trimmer)
      • optionally emit acknowledgment event
  → get_compiled_graph().invoke(state, config={"configurable": {"sp_token": ...}})
      ├─ supervisor_node          (intent classify → Command(goto=..))
      ├─ Simple path:
      │    genie_data_node → genie_analysis_node → format_supervisor_node → END
      └─ Complex path:
           planner_node → insight_agent_node → increment_step
                              ↑            │ (loops until steps exhausted)
                              └────────────┘
                           → synthesizer_node → format_supervisor_node → END
  → [post-graph]
      • extract supervisor_json.items (text / table / chart)
      • persist STM checkpoint, write LTM updates
      • emit SSE phases and final ResponsesAgentResponse
  → client
```

## Control-Flow Details

| Stage | What happens |
|-------|--------------|
| **Intent routing** | `supervisor_node` returns `Command(update=..., goto=X)` where X ∈ {`greeting`, `clarification`, `genie_data`, `planner`}. Intent values observed in code: `greeting`, `data_query`, `complex_query`, `clarification` (4 values — CLAUDE.md mentions 5). |
| **Simple path** | `genie_data` (REST call + table shaping) → `genie_analysis` (LLM analysis) → `format_supervisor` (structured items + chart LLM). |
| **Complex path** | `planner` decomposes into `plan[]` steps → `insight_agent` executes one step (ReAct subgraph in `capabilities/insight_agent.py`) → `increment_step` advances index → `should_continue_plan` conditional edge loops back or falls through to `synthesizer` → `format_supervisor`. |
| **Legacy path** | `genie_worker` + `_fan_out_to_workers` Send-API parallel fanout remain wired (`genie_worker → synthesizer`) but are not on the default routes. |

## Data Flow

1. **Input**: `AgentState` seeded from `ResponsesAgentRequest.custom_inputs` (`sp_id`, `user_id`, `conversation_id`, `task_type`, `user_name`).
2. **Auth**: `core/auth.py` resolves service-principal credentials from the Databricks secrets scope `agent-sp-credentials`, mints an OBO token, and injects it into `config["configurable"]["sp_token"]` — nodes read it from config, never from state (keeps checkpoints clean).
3. **Genie**: `genie_data_node` calls the Genie REST API (`agents/genie_client.py`) and populates `genie_columns`, `genie_data_array`, `genie_tables` (UI-ready 2D dicts), `genie_table` (LLM-readable text), `genie_sql`, `genie_trace_id`.
4. **Analysis**: `genie_analysis_node` adds `genie_summary` + `genie_insights` via Sonnet-class LLM.
5. **Formatting**: `format_supervisor_node` builds `response_items` as `[{type, id, name, value, hidden}]` where:
   - Table items are constructed programmatically from `genie_tables` (no LLM).
   - Text + chart items come from a single LLM call constrained by `RESPONSE_FORMAT_SCHEMA` (JSON schema, strict).
   - Large transient fields (`genie_data_array`, `genie_columns`, `genie_table`, `genie_text_content`) are explicitly cleared before checkpoint write to keep STM small.
6. **Output**: `response_items` flow back into `ResponsesAgentResponse.output`; MLflow trace ID attached via `custom_outputs.mlflow_trace_id`.

## Streaming (SSE) Phases

Emitted via `langgraph.config.get_stream_writer()` inside nodes; relayed as `data:` frames. Phases present in code: `thought`, `plan`, `progress` (via `node_started` events), `data_table`, `chart`, `analysis`, `recommendations`, `complete` — CLAUDE.md lists 8, code confirms ≥6 actively emitted (2 partial per audit).

## External Integrations

| System | Used for | Access point |
|--------|----------|--------------|
| Databricks Genie (REST) | Campaign SQL + data returns | `agents/genie_client.py` → `agents/genie_agent.py` |
| Databricks AgentBricks (alt) | MCP-style access path (migration target) | `agents/agentbricks.py` |
| AI Gateway (`ChatOpenAI`) | All LLM calls currently routed here | `settings.AI_GATEWAY_URL` + `LLM_ENDPOINT_NAME=databricks-gpt-5-2` |
| Lakebase Postgres (`agentmemory`) | STM checkpoint + LTM store | `memory/ltm_manager.py`, LangGraph `CheckpointSaver` |
| MLflow 3 | Tracing + feedback assessments + experiment | `core/tracing.py`, `feedback.py` |
| Unity Catalog (`channel.gold_channel.campaign_details`) | Single-source analytics table (RLS on `cid`) | Accessed indirectly via Genie space `GENIE_SPACE_ID` |
| SQL Warehouse `64769e0e91e002b8` | Trace table writes | `core/tracing.py` |
