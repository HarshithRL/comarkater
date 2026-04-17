# 1. Repository Map

Evidence-based mapping of the CoMarketer repo at `D:\netcore_co\comarketer`. Every entry is derived from file-reads; items marked UNKNOWN could not be verified.

---

## 1.1 Top-level directories

| Path | Purpose |
|------|---------|
| `agent_server/` | Main application — LangGraph agents, graph, memory, UI, tests (~13k LOC) |
| `bundles/netcore-insight-agent/` | Databricks Asset Bundle config (terraform state + secondary `databricks.yml`); points to `agent_server` as source |
| `docs/` | Markdown scaffolding (6 files, not authoritative — `CLAUDE.md` is) |
| `notebooks/` | 17 test/debug notebooks (deployment verification, streaming smoke tests, permission grants) |
| `.claude/` | Claude Code config / memory |
| Root | `CLAUDE.md` (spec), `databricks.yml` (deployment), `pyproject.toml`, `uv.lock`, `workflow.md`, `setup_claude_code.sh` |

---

## 1.2 `agent_server/` — per-subdirectory breakdown

### Top-level files

| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| `agent.py` | ~1,494 | `CoMarketerAgent(ResponsesAgent)` — predict/predict_stream, LTM/STM glue, graph invoke, SSE event mapping | ✅ Active; **VIOLATES 200-line rule** |
| `start_server.py` | ~83 | Entry point; MLflow autolog, `AgentServer` boot, Flask UI mount, ASGI no-buffer SSE middleware | ✅ Active |

### `agents/` — agent nodes

| File | Purpose | Wired? |
|------|---------|--------|
| `supervisor.py` | Supervisor node: regex greeting → LLM classify → `Command(goto=...)` (4 intents) | ✅ |
| `greeting.py` | Returns greeting string (no LLM) | ✅ |
| `clarification.py` | Answers meta-questions from context | ✅ |
| `planner.py` | Legacy complex-query decomposer | ❌ NOT WIRED |
| `genie_agent.py` | Legacy data-fetch node | ❌ NOT WIRED |
| `genie_analysis.py` | Legacy analyzer | ❌ NOT WIRED |
| `genie_worker.py` | Legacy parallel-fetch worker | ❌ NOT WIRED |
| `synthesizer.py` | Legacy multi-worker synthesizer | ❌ NOT WIRED |
| `genie_client.py` | Sync `httpx` Genie REST client (`start_conversation`, `check_status`, `fetch_result`) | ✅ (used by CampaignInsightAgent) |
| `registry.py` | Legacy capability registry | ❌ |

### `agents/campaign_insight/` — CampaignInsightAgent subgraph (active)

| File | Role |
|------|------|
| `agent.py` | `CampaignInsightAgent.run()` — async orchestrator of 5 phases |
| `adaptive_planner.py` | Phase 1: decompose intent → dimension-constrained steps |
| `executor.py` | Phase 2: ReAct loop (reason → Genie → observe → evaluate, ≤3 iters, 120s) |
| `interpreter.py` | Phase 3: deterministic + LLM pattern classification on table data |
| `recommender.py` | Phase 4: 2–3 recommendations per pattern category |
| `output_builder.py` | Phase 5: assemble `SubagentOutput` envelope |
| `reflector.py` | Self-correction on interpretations/recommendations |
| `dimension_classifier.py` | Tag fields as TIME/CHANNEL/METRIC/AUDIENCE/CONTENT |
| `dimension_validator.py` | Rule-based cross-check vs intent |
| `table_analyzer.py` | Pure-Python stats summary of the returned table |
| `table_builder.py` | Format 2D data into `DisplayTable` |
| `chart_builder.py` | Highcharts 11.x JSON generator (LLM-assisted) |
| `tool_handler.py` | Genie call translator |
| `contracts.py` | Dataclasses/enums: `DimensionRole`, `StepStatus`, `GenieResponse`, `DisplayTable`, `PlanStep`, `SubagentOutput`, … |
| `domain_knowledge.py` | YAML loader for domain files |
| `domain_knowledge/*.yaml` | `intent.yaml`, `metrics.yaml`, `constraints.yaml`, `domain_context.yaml`, `audience_knowledge.yaml`, `content_knowledge.yaml`, `interpretation.yaml`, `recommendations.yaml`, `domain_context_data_source_update.yaml` |
| `prompts/interpretation_prompt.py` | Prompts for interpreter + recommender |
| `prompts/reflection_prompt.py` | Self-correction prompts |

### `core/`

| File | Purpose |
|------|---------|
| `state.py` | `AgentState` TypedDict (messages, routing, genie_*, ltm_context, conversation_history, plan, step_results, subagent_input/output, genie_retry_count, error) |
| `graph.py` | Compiled `StateGraph`: `supervisor_classify → {greeting|clarification|out_of_scope|campaign_insight_agent → supervisor_synthesize} → END` |
| `config.py` | Settings (env-var loader): `LLM_ENDPOINT_NAME`, `AI_GATEWAY_URL`, `GENIE_SPACE_ID`, `LAKEBASE_*`, `EMBEDDING_*`, `SECRETS_SCOPE` |
| `auth.py` | `SecretsLoader` (in-memory cache), `SecureTokenProvider` (OAuth2 client_credentials for SP OBO) |
| `tracing.py` | Custom MLflow spans, `trace_logger.init_table()` Delta-backed trace sink |
| `models.py` | Pydantic `CustomInputs`, `RESPONSE_FORMAT_SCHEMA` (JSON schema, strict=True) |

### `prompts/`

| File | LOC | Injected into |
|------|-----|----------------|
| `supervisor_prompt.py` | 1,063 | Supervisor classify + synthesize LLM |
| `insight_agent_prompt.py` | ~80+ | ReActExecutor (`INSIGHT_REACT_PROMPT`) |
| `acknowledgment_prompt.py` | small | `create_acknowledgment()` |
| `genie_agent_prompt.py` | — | Legacy; unused |
| `planner_prompt.py` | — | Legacy; unused |
| `synthesizer_prompt.py` | — | Legacy; unused |

### `parsers/`

| File | Purpose |
|------|---------|
| `table_truncator.py` | Shrink large tables before LLM injection |
| `subagent_parser.py` | Parse `SubagentOutput` → response items |
| `filters.py` | (UNKNOWN specifics — appears to parse filter strings) |
| `format_for_client.py` | Comma-grouping counts, rate→% formatting |
| (others) | 5 files total |

### `memory/`

| File | Purpose |
|------|---------|
| `ltm_manager.py` | `LTMManager` — `DatabricksStore` (vector) for client profiles + episodes; namespace `("client", "{scope}", "profile"|"episodes")`; uses `databricks-gte-large-en`, 1024 dims |
| `extractors.py` | `extract_entities_from_query`, `extract_metrics_from_query`, `extract_insights_from_response` |
| `context_formatter.py` | `format_ltm_context`, `format_greeting_context` — markdown injection |
| `stm_trimmer.py` | `get_trim_removals()` — bounded history |
| (5 files total) | |

### `ui/`

| File | Routes |
|------|--------|
| `routes.py` | `GET /` (index), `POST /chat` (invoke or SSE stream), `POST /feedback`, `GET /health`, `GET /debug` |
| `templates + static` | index.html + assets (UNKNOWN inventory — not fully enumerated) |

### `tests/`

| File | Coverage |
|------|----------|
| `test_local.py` | 2 smoke tests: greeting for known SP (IGP), unknown-SP fallback |

### Other subdirs

| Path | Purpose | Wired? |
|------|---------|--------|
| `tools/` | Single file — Genie query/search wrappers | partial |
| `supervisor/` (5 files: `intent_classifier.py`, `router.py`, `planner.py`, `synthesizer.py`, `domain_context.py`) | New classes not referenced by the compiled graph (synthesizer IS imported via `supervisor_synthesize_node`) | Synthesizer ✅; others ❌ STUBS |
| `capabilities/` | `insight_agent_node`, `capability_registry` | ❌ Legacy |

---

## 1.3 Root files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Architectural spec (authoritative) |
| `databricks.yml` | Bundle + App deployment: secrets, MLflow experiment, SQL warehouse, env vars |
| `pyproject.toml` | Deps: `mlflow[databricks]>=3.1.3`, `langgraph>=0.3`, `langchain-core>=0.3`, `databricks-langchain>=0.5`, `databricks-sdk>=0.50`, `flask`, `asgiref`, `httpx`, `pydantic>=2` |
| `workflow.md` | Phase 3 request lifecycle doc |
| `setup_claude_code.sh` | Dev env setup |
| `uv.lock` | UV lockfile |
| `bundles/netcore-insight-agent/` | Secondary bundle with terraform state; duplicates top-level `databricks.yml` intent (UNKNOWN whether both are active) |

---

## 1.4 `notebooks/` (17 files)

Deployment verification (`01_verify_deployment.py`, `check_permissions.py`, `fix_permissions.py`, `grant_permission.py`), debug (`debug_traces.py`, `inspect_agentbricks_raw.py`, `discover_lakebase.py`), integration smoke (`test_endpoint.py`, `test_streaming.py`, `test_greeting_stream.py`, `test_progressive_stream.py`, `test_structured_output.py`, `test_with_requests.py`, `test_phase_a.py`, `test_deep_query.py`, `test_graph_caching.py`, `demo_request_response.py`).

All are dev-facing; none imported by the runtime.
