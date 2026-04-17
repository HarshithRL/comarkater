# 3. Execution Flow

Trace from HTTP request to SSE response, grounded in `agent_server/agent.py`, `core/graph.py`, `start_server.py`, and `ui/routes.py`.

---

## 3.1 Entry points

| Entry | File | Notes |
|-------|------|-------|
| CLI boot | `agent_server/start_server.py:main` | Sets `MLFLOW_TRACKING_URI=databricks`, `mlflow.langchain.autolog()`, constructs `AgentServer`, mounts Flask UI via `WsgiToAsgi`, installs no-buffer SSE ASGI middleware |
| MLflow `/invocations` | `@invoke` on `agent.predict` | Non-streaming |
| MLflow `/stream` | `@stream` on `agent.predict_stream` | SSE, yields `ResponsesAgentStreamEvent` |
| Flask `POST /chat` | `ui/routes.py` | Thin wrapper: builds `ResponsesAgentRequest`, returns `Response(generator, content_type="text/event-stream")` or JSON |
| Notebooks | `notebooks/test_*.py` | Dev-only |

---

## 3.2 Request lifecycle (`predict_stream`)

| # | Step | Location | Output |
|---|------|----------|--------|
| 1 | Extract `CustomInputs` (sp_id, user_name, thread_id, conversation_id, task_type) | `agent.py:_extract_custom_inputs` | Pydantic model |
| 2 | Emit MLflow trace tags (client_id, user_name, request_id) | `agent.py:~313–337` | trace metadata |
| 3 | Greeting short-circuit: regex `GREETING_PATTERNS` / `GREETING_WITH_NAME` | `agent.py:~363–420` | If match → optional LTM personalization → early return (no graph) |
| 4 | LTM read (gated on client_scope + query keywords) | `agent.py:~422–452` under span `ltm_gated_read` | `ltm_context` markdown |
| 5 | STM read via `CheckpointSaver.get` + `get_trim_removals` | `agent.py:~469–497` | `conversation_history` string |
| 6 | Build per-request LLM: `ChatOpenAI(base_url=AI_GATEWAY_URL, model=LLM_ENDPOINT_NAME, temperature=0.0)` with SP token | `agent.py:_build_llm` | Caller LLM |
| 7 | Acknowledgment generation (Haiku-style "working on it" text) | `agent.py:~520–548` + `create_acknowledgment` | streamed as `thought` phase |
| 8 | `graph.stream(state, config, stream_mode="updates")` iteration | `agent.py:~552–1380` | Per-node state updates |
| 9 | Map node updates → `ResponsesAgentStreamEvent` (`text`, `table`, `chart`, `progress`) | `agent.py` large if/elif per node name | SSE chunks yielded |
| 10 | LTM write (extract entities + insights from final response, `LTMManager.save_episode`) | `agent.py:~1381–1403` | Episode persisted |
| 11 | TRACE_DONE event with `mlflow.get_last_active_trace_id()` | `agent.py:~1405–1421` | Feedback linkage |

`predict` (non-streaming) reuses steps 1–10 but accumulates `response_items` into a single `ResponsesAgentResponse`.

---

## 3.3 Graph execution — DATA_ANALYSIS example

User: "Show me email performance last month."

```
supervisor_classify
  L1: not a greeting
  L2: LLM → {intent: "simple_query", rewritten_question: "..."}
  → Command(goto="campaign_insight_agent", update={...})

campaign_insight_agent (one LangGraph node; internally async)
  Phase 1 AdaptivePlanner  → 1 step (channel=email, period=last_month, metrics=[...])
  Phase 2 ReActExecutor
    - ToolHandler.ask_genie → genie_client.start_conversation
    - poll check_status: FILTERING_CONTEXT → ASKING_AI → PENDING_WAREHOUSE → EXECUTING_QUERY → COMPLETED
    - fetch_result → GenieResponse(columns, data_array, sql, statement_id)
    - DimensionClassifier tags fields
    - TableAnalyzer computes summary
  Phase 3 Interpreter  → patterns (e.g. open_rate=18.5%)
  Phase 4 Recommender  → 2–3 recs (Apply/Avoid/Explore)
  Phase 5 OutputBuilder → SubagentOutput
  → state update: {subagent_output: ...}

supervisor_synthesize
  - Reads subagent_output
  - Calls supervisor LLM with strict JSON schema
  - Builds response_items: [text("What Did We Find?"), table, chart, text("Recommendations")]
  → state update: {response_items: [...], supervisor_json: {"items": [...]}}

END
```

---

## 3.4 SSE phase mapping (actual vs CLAUDE.md 8-phase spec)

| Spec phase | Real stream event | Source |
|-----------|-------------------|--------|
| thought | ack text as RATIONALE item | step 7 |
| plan | **implicit** — not emitted as a distinct phase | — |
| progress | Genie status events | ReActExecutor polling |
| data_table | `table` item | supervisor_synthesize |
| chart | `chart` item | supervisor_synthesize |
| analysis | `text` items (findings) | supervisor_synthesize |
| recommendations | `text` items (recs) | supervisor_synthesize |
| complete | **implicit** — TRACE_DONE event emitted instead | step 11 |

6 explicit + 2 implicit of the claimed 8.

---

## 3.5 State transitions (selected fields)

| Node | Writes |
|------|--------|
| supervisor_classify | `intent`, `rewritten_question`, `original_question` |
| campaign_insight_agent | `subagent_output`, `genie_*`, `step_results`, `llm_call_count`, `genie_retry_count` |
| supervisor_synthesize | `response_items`, `supervisor_json`, `response_text` |
| greeting / clarification / out_of_scope | `response_items` (terminal) |

---

## 3.6 Error paths

- Genie FAILED → ReActExecutor returns error step; user receives "Please rephrase and try again" (no node-level escalation).
- Lakebase unreachable → STM falls back to `InMemorySaver`; LTM read/write is try/except with logged error.
- MLflow experiment setup fails → logged warning, traces default to the default experiment.
- No global retry/backoff chain beyond `genie_retry_count` counter.

---

## 3.7 Concurrency model

- MLflow `ResponsesAgent.predict_stream` is sync generator.
- `CampaignInsightAgent.run` is async.
- Bridge: `agent.py` uses `loop.run_in_executor(None, run_sync)` (≈ line 1468) so the async subgraph runs on a dedicated thread, avoiding `ContextVar` issues with MLflow tracing (documented in in-file comment ~1467).
