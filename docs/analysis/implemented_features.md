# 5. Implemented Features

What **actually** works in the current codebase, with source-file evidence.

---

## 5.1 Routing and orchestration

- ✅ Supervisor classification + rewrite — `agents/supervisor.py`.
- ✅ Greeting regex pre-filter with optional LTM personalization — `agent.py:~363`.
- ✅ Clarification node for meta-questions — `agents/clarification.py`.
- ✅ Out-of-scope canned response.
- ✅ LangGraph `Command`-based routing (no conditional edges).
- ✅ Synthesizer producing strict-JSON response (`core/models.py:RESPONSE_FORMAT_SCHEMA`).

## 5.2 Campaign analytics pipeline

- ✅ Adaptive planning (dimension-constrained steps) — `adaptive_planner.py`.
- ✅ ReAct executor with Genie polling (≤3 iterations, 120s budget) — `executor.py`.
- ✅ Genie REST client (`start_conversation` → `check_status` → `fetch_result`) — `genie_client.py`.
- ✅ Dimension classification + rule-based validation — `dimension_classifier.py`, `dimension_validator.py`.
- ✅ Table analysis (shape, numeric stats, light anomalies) — `table_analyzer.py`.
- ✅ Pattern interpretation (deterministic + LLM) — `interpreter.py`.
- ✅ Recommendations (Apply / Avoid / Explore) — `recommender.py`.
- ✅ Reflection / self-correction — `reflector.py`.
- ✅ Highcharts 11.x chart JSON generation with compatibility rules — `chart_builder.py`, supervisor prompt visualization section.
- ✅ `SubagentOutput` envelope and strict-schema response items — `contracts.py`, `core/models.py`.

## 5.3 Memory

- ✅ STM via Lakebase `CheckpointSaver` (thread-scoped; `InMemorySaver` fallback) — `agent.py:_get_stm_checkpointer`.
- ✅ STM trimming — `memory/stm_trimmer.py:get_trim_removals`.
- ✅ LTM client profile + episode storage via `DatabricksStore` (vector, 1024-dim `databricks-gte-large-en`) — `memory/ltm_manager.py`.
- ✅ Query-gated LTM read (only when keywords suggest need) — `agent.py:~422`.
- ✅ Entity/metric/insight extraction and episode save after response — `memory/extractors.py`, `agent.py:~1381`.
- ✅ Markdown formatting of LTM context for prompt injection — `memory/context_formatter.py`.

## 5.4 Auth & multi-tenant

- ✅ Per-client Service Principal OBO token flow — `core/auth.py:SecureTokenProvider`.
- ✅ Secret caching via `SecretsLoader` against `agent-sp-credentials` scope.
- ✅ Client registry (IGP, Pepe, Crocs, Demo fallback).
- ✅ RLS delegated to Unity Catalog — agent never injects `WHERE cid` filters.

## 5.5 Observability

- ✅ MLflow 3 auto-logging (`mlflow.langchain.autolog()`).
- ✅ Root `@mlflow.trace` span on `predict`.
- ✅ Custom spans (e.g., `ltm_gated_read`).
- ✅ Trace tagging with `client_id`, `user_name`, `request_id`.
- ✅ Delta trace table initialized at startup — `core/tracing.py`.
- ✅ Trace ID returned as final SSE event for feedback linkage.

## 5.6 Streaming

- ✅ SSE via MLflow `ResponsesAgent.predict_stream`.
- ✅ ASGI no-buffer middleware handling Databricks App proxy — `start_server.py`.
- ✅ Phases emitted: thought (ack), progress, data_table, chart, analysis, recommendations (6 explicit + 2 implicit).
- ✅ Acknowledgment message before long-running analysis — `acknowledgment_prompt.py`.

## 5.7 UI

- ✅ Flask app mounted on `/ui` via `WsgiToAsgi`.
- ✅ Functional chat — `POST /chat` (invoke or stream modes).
- ✅ Feedback endpoint (thumbs up/down) — `POST /feedback`.
- ✅ Health + debug endpoints.
- ✅ Client picker (IGP / Pepe / Crocs).

## 5.8 Domain knowledge

- ✅ YAML-driven intent taxonomy, metrics, constraints, channel/audience/content vocab — `agents/campaign_insight/domain_knowledge/*.yaml`.
- ✅ Loaded once into `InsightAgentDomainKnowledge` and injected into ReAct / Interpreter prompts.
- ✅ Minimum volume + rate thresholds enforced via validators.

## 5.9 Deployment

- ✅ Asset Bundle (`databricks.yml`) with secrets, MLflow experiment, SQL warehouse bindings.
- ✅ Reproducible boot via `python start_server.py`.
- ✅ `databricks bundle deploy` + `databricks apps deploy` documented in `CLAUDE.md`.

## 5.10 Testing

- ✅ 2 pytest smoke tests — `tests/test_local.py` (greeting, unknown SP fallback).
- ✅ 17 integration / ops notebooks.
