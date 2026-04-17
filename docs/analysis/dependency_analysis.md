# 9. Dependency Analysis

External and internal dependencies of the CoMarketer codebase.

---

## 9.1 External package dependencies (`pyproject.toml`)

| Package | Min version | Used for |
|---------|-------------|----------|
| `mlflow[databricks]` | >= 3.1.3 | `ResponsesAgent`, `@mlflow.trace`, `langchain.autolog`, Delta trace table |
| `langgraph` | >= 0.3.0 | `StateGraph`, `CheckpointSaver`, `Command` routing |
| `langchain-core` | >= 0.3.0 | `BaseMessage`, LLM abstractions |
| `langchain-openai` | (implied) | `ChatOpenAI` against AI Gateway |
| `databricks-langchain` | >= 0.5.0 | `CheckpointSaver` (Lakebase-backed STM), `DatabricksStore` (Lakebase vector LTM) |
| `databricks-sdk` | >= 0.50.0 | `WorkspaceClient` for Secrets/resources |
| `flask` | >= 3.0 | UI + REST endpoints |
| `asgiref` | >= 3.7 | `WsgiToAsgi` to mount Flask under the ASGI SSE pipeline |
| `httpx` | >= 0.27 | Sync Genie REST client |
| `pydantic` | >= 2.0 | `CustomInputs`, response schema |

**Not pinned**: `mlflow.types.responses` is used extensively (`ResponsesAgent*`); breaking changes will cascade.

---

## 9.2 External service dependencies

| Service | Interface | Auth |
|---------|-----------|------|
| Databricks AI Gateway | HTTPS `ChatOpenAI(base_url=AI_GATEWAY_URL)` | per-client SP token |
| Genie (Databricks) | REST API via `httpx` | per-client SP token |
| Lakebase (`agentmemory`) | `CheckpointSaver`, `DatabricksStore` | Databricks SDK credentials |
| Embedding endpoint `databricks-gte-large-en` | Databricks Serving | Databricks SDK |
| SQL Warehouse `64769e0e91e002b8` | Used for Delta trace table | Databricks SDK |
| MLflow Tracking (`databricks`) | `MLFLOW_TRACKING_URI=databricks` | Databricks workspace |
| Databricks Secrets scope `agent-sp-credentials` | `WorkspaceClient.secrets` | workspace creds |
| Unity Catalog `channel.gold_channel.campaign_details` | Indirect via Genie | RLS on `cid` |

Failure of Lakebase degrades gracefully (InMemorySaver + try/except LTM). Failure of Genie or AI Gateway is terminal for any non-greeting request.

---

## 9.3 Internal module coupling (hotspots)

| Module | Inbound fan-in | Risk |
|--------|----------------|------|
| `agent.py` | all entry paths | **Tight coupling** — touches config, auth, memory, graph, MLflow, SSE mapping. Highest blast radius. |
| `core/state.py:AgentState` | every node | Adding a state field is cheap; removing one is expensive (checkpoint migration required). |
| `prompts/supervisor_prompt.py` | supervisor_classify + supervisor_synthesize + synthesizer | Single 1,063-line prompt governs two very different concerns (routing + response formatting). |
| `agents/campaign_insight/contracts.py` | all 12 insight sub-components | Dataclass changes cascade widely; no versioning. |
| `core/models.py:RESPONSE_FORMAT_SCHEMA` | UI + supervisor synthesize | Front-end contract — schema change breaks client. |
| `agents/genie_client.py` | ToolHandler, ReActExecutor, legacy nodes | Only external data access point. |

## 9.4 Circular / cross-layer imports

- None observed inbound to `core/` from `agents/` (good).
- `agents/campaign_insight/*` self-contained (good).
- `supervisor/synthesizer.py` is imported by `core/graph.py`; `supervisor/` folder otherwise unreferenced (see dead code).

## 9.5 Environment variables (runtime contract)

| Var | Purpose |
|-----|---------|
| `MLFLOW_TRACKING_URI` = `databricks` | MLflow sink |
| `MLFLOW_EXPERIMENT_ID` / `MLFLOW_EXPERIMENT_NAME` | Experiment binding |
| `AI_GATEWAY_URL` | LLM endpoint base |
| `LLM_ENDPOINT_NAME` = `databricks-gpt-5-2` | Model id |
| `GENIE_SPACE_ID` | Genie space |
| `SECRETS_SCOPE` = `agent-sp-credentials` | Secret scope |
| `LAKEBASE_INSTANCE_NAME` / `LAKEBASE_PROJECT` / `LAKEBASE_BRANCH` / `LAKEBASE_MODE` | Lakebase STM + LTM |
| `EMBEDDING_ENDPOINT` / `EMBEDDING_DIMS` | LTM vector config |
| `SQL_WAREHOUSE_ID` | Trace table warehouse |
| `COMARKETER_ENV` | Application env flag (currently `development` even in prod) |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | Implicit `WorkspaceClient` credentials |

## 9.6 Deployment flow (inferred)

1. `databricks bundle deploy --profile DEFAULT` → uploads code + config.
2. Databricks App runtime starts `python start_server.py`.
3. `start_server.py` sets MLflow tracking, calls `autolog()`, constructs `CoMarketerAgent`, mounts Flask UI, installs ASGI SSE middleware.
4. Graph is compiled once (singleton via `core/graph.py`).
5. SP secrets read lazily per request via `SecretsLoader`.
6. Lakebase instance `agentmemory` must be provisioned **manually** (Asset Bundles do not yet support it).

## 9.7 Decoupling recommendations

- Introduce a **Genie adapter interface** so `ToolHandler` can swap REST ↔ MCP without touching `executor.py`.
- Move response-format JSON schema and Highcharts rules out of `supervisor_prompt.py` into dedicated modules that both synthesizer and chart builder consume.
- Version `contracts.py` dataclasses (schema version field) so state evolution doesn't invalidate existing checkpoints.
- Extract per-request LLM factory from `agent.py:_build_llm` into `core/llm_factory.py` to prepare for Haiku/Sonnet routing per CLAUDE.md.
