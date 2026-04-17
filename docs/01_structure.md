# 01 — Repository Structure

## Folder Tree

```
comarketer/
├── agent_server/                 # Core application (deployed as Databricks App)
│   ├── agent.py                  # CoMarketerAgent wrapper (ResponsesAgent) — 1,441 LOC
│   ├── start_server.py           # Entry: AgentServer + SSE middleware + Flask /ui mount
│   ├── feedback.py               # Writes user 👍/👎 to MLflow trace assessments
│   ├── app.yaml                  # Databricks App runtime env (production overlay)
│   ├── requirements.txt
│   ├── agents/                   # 16 node modules (supervisor + workers + helpers)
│   ├── capabilities/             # insight_agent subgraph (ReAct) + registry
│   ├── core/                     # state, graph, config, auth, tracing, models, filtered_chat
│   ├── memory/                   # LTM manager, STM trimmer, context formatter, extractors
│   ├── parsers/                  # formatters, validators, subagent_parser, filters, table_truncator
│   ├── prompts/                  # 6 prompt templates (supervisor, planner, synthesizer, ...)
│   ├── tools/                    # tool registry (LangChain-compat)
│   ├── ui/                       # Flask routes + templates + static
│   └── tests/                    # Single placeholder pytest file
├── notebooks/                    # 17 ad-hoc diagnostic/evaluation scripts
├── bundles/netcore-insight-agent # Separate sub-bundle (Databricks asset bundle)
├── pyproject.toml                # Python 3.11+, deps listed below
├── databricks.yml                # Main App bundle (resources, env, secrets)
├── CLAUDE.md                     # Contributor guide + architecture invariants
├── workflow.md                   # Roadmap / phase tracker
└── uv.lock, .env, .env.example, .python-version, .gitignore
```

## Entry Points

| Path | Role |
|------|------|
| `agent_server/start_server.py` | Boots `AgentServer`, mounts `SSENoBufferMiddleware`, mounts Flask `/ui` via `WsgiToAsgi`, enables MLflow git-based version tracking. |
| `agent_server/agent.py` | Registers `CoMarketerAgent(ResponsesAgent)` — called by `@invoke` / `@stream` via AgentServer. |
| `agent_server/ui/routes.py` | Flask Blueprint: `/`, `/chat` (SSE+JSON), `/feedback`, `/health`, `/debug`. |

## Module One-Liners

| Module | Purpose |
|--------|---------|
| `agents/` | All LangGraph node implementations. One node per file. |
| `capabilities/` | ReAct-style capability subgraphs invoked inside `insight_agent_node`. |
| `core/` | Infrastructure: state schema, graph builder, settings, auth/OBO, MLflow tracing, LLM wrappers. |
| `memory/` | Lakebase LTM store + STM trimming + context formatting for prompt injection. |
| `parsers/` | Deterministic post-processors for LLM/Genie output (formatting, validation, truncation). |
| `prompts/` | System-prompt string templates, one per agent. |
| `tools/` | Tool definitions discoverable by ReAct capability agents. |
| `ui/` | Human-facing chat UI (Flask) mounted at `/ui` of the AgentServer. |
| `tests/` | pytest (currently minimal — 1 file). |
| `notebooks/` | Ad-hoc diagnostic / integration / evaluation scripts — **not** deployment artifacts. |
| `bundles/netcore-insight-agent/` | Secondary Databricks asset bundle (scope unclear — see `06_unknowns.md`). |

## Key Files — Size & Role

| File | LOC | Role |
|------|----:|------|
| `agent.py` | 1,441 | Orchestrator: request shaping, graph invocation, SSE event emission, memory I/O. |
| `prompts/supervisor_prompt.py` | 1,062 | Static supervisor system prompt (intent rules + formatting contract + examples). |
| `agents/agentbricks.py` | 479 | Alternative Genie access path via Databricks AgentBricks endpoint (MCP target — see `04_code_audit.md`). |
| `core/graph.py` | 374 | `StateGraph` builder (cached, thread-safe), conditional plan-loop edge, `format_supervisor_node`. |
| `memory/ltm_manager.py` | 338 | Long-term memory CRUD on Lakebase `DatabricksStore`. |
| `tools/registry.py` | 297 | Tool registry surfaced to ReAct capabilities. |
| `agents/synthesizer.py` | 292 | Merges sequential step_results (or legacy worker_results) via LLM into final summary. |
| `agents/genie_agent.py` | 286 | `genie_data_node` — the simple-path Genie call + table shaping. |
| `ui/routes.py` | 266 | Flask routes (chat, feedback, debug, health). |
| `agents/genie_worker.py` | 242 | Legacy parallel fan-out worker (kept for backward compat). |
| `core/tracing.py` | 204 | MLflow spans + Delta trace table writer. |
| `agents/supervisor.py` | 208 | Intent classifier + router emitting `Command(goto=...)`. |
| `core/auth.py` | 191 | Databricks secrets loader + OBO token provider. |
| `core/state.py` | 175 | `AgentState` / `CapabilityState` TypedDicts + initializers. |

## Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Deps: `mlflow[databricks]≥3.1.3`, `langgraph≥0.3`, `langchain-core≥0.3`, `databricks-langchain[memory]`, `databricks-sdk`, `flask≥3`, `asgiref`, `httpx`, `pydantic≥2`. Dev: `pytest`, `ruff`. |
| `databricks.yml` | Main App bundle: experiment, SQL warehouse `64769e0e91e002b8`, 8 SP-credential secret refs, 17 env vars incl. `GENIE_SPACE_ID`, `LAKEBASE_INSTANCE_NAME=agentmemory`, `LLM_ENDPOINT_NAME=databricks-gpt-5-2`. |
| `agent_server/app.yaml` | Production overlay (`AGENT_ENV=production`, MLflow experiment resource binding). |
| `.env.example` | Local dev env template. |
