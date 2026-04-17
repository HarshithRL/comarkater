# 03 — Agent Architecture

*See `01_structure.md` for files, `02_workflow.md` for flow.*

## Framework

- **LangGraph** `StateGraph` — single compiled graph, cached module-level (`core/graph.py`), guarded by `threading.Lock`. One compile for the process lifetime.
- **MLflow `ResponsesAgent`** — `agent.py` subclasses it to surface `predict` / `predict_stream`.
- **Command routing** — supervisor returns `Command(update=..., goto=...)` rather than conditional edges for the initial branch.
- **Send API fan-out** — implemented in `_fan_out_to_workers` (legacy path); retained but not on default routes.
- **Subgraph** — `capabilities/insight_agent.py` exposes `insight_agent_node` as a subgraph with overlapping keys (`messages`, `user_query`, `cid`, `current_task`) so the parent `AgentState` and child `CapabilityState` exchange data automatically.

## Graph Topology

```
              ┌───────────────────────────────────────────────────┐
              │                   AgentState                      │
              └───────────────────────────────────────────────────┘
                               ▲           ▲
                 read/write    │           │    read/write
                               │           │
START → supervisor ─Command─┬──► greeting ───────────────────────► END
                            ├──► clarification ──────────────────► END
                            ├──► genie_data ─► genie_analysis ─┐
                            │                                   ▼
                            │                          format_supervisor ► END
                            └──► planner ─► insight_agent ─► increment_step
                                              ▲                    │
                                              │  (loop)            │
                                              └────────────────────┤
                                                                   ▼
                                                       synthesizer ─► format_supervisor ► END

[legacy] genie_worker ─────────► synthesizer     (Send-API fan-out, not on default path)
```

## Agent / Node Inventory

| Node | Source | Responsibility |
|------|--------|----------------|
| `supervisor` | `agents/supervisor.py` | Intent classification (`greeting` / `data_query` / `complex_query` / `clarification`), question rewriting, Command routing. Uses Haiku per CLAUDE.md but code currently routes through `ChatOpenAI` + AI Gateway. |
| `greeting` | `agents/greeting.py` | Static friendly response (no LLM if possible). |
| `clarification` | `agents/clarification.py` | Asks user a disambiguation question. |
| `genie_data` | `agents/genie_agent.py` | Calls Genie REST, shapes `genie_columns/data_array/tables/table/sql`. |
| `genie_analysis` | `agents/genie_analysis.py` | LLM writes `genie_summary` + `genie_insights`. |
| `planner` | `agents/planner.py` | LLM decomposes complex query into ordered `plan[]` of sub-questions. |
| `insight_agent` | `capabilities/insight_agent.py` | ReAct subgraph — executes one plan step with tool access. |
| `increment_step` | `core/graph.py` | Advances `current_step_index`; prepares next `current_task`/`current_capability`. |
| `synthesizer` | `agents/synthesizer.py` | LLM merges `step_results` (or legacy `worker_results`) into final summary. |
| `format_supervisor` | `core/graph.py` | Builds final `response_items` (programmatic tables + LLM text/chart via strict JSON schema). |
| `genie_worker` *(legacy)* | `agents/genie_worker.py` | Parallel Send-API fan-out worker; still in graph but not on default path. |
| `acknowledgment`, `followup` | `agents/*.py` | Helper utilities invoked outside the graph (pre-graph ack, post-graph follow-up suggestions). |

## Tools

- `tools/registry.py` (297 LOC) exposes LangChain-compatible tools to the `insight_agent` ReAct loop.
- `capabilities/registry.py` maps capability names → subgraph instances (routing from planner steps).
- `agents/capability_registry.py` + `agents/registry.py` + `agents/agent_model.py` — capability/agent metadata (relationship to `capabilities/registry.py` is unclear, see `06_unknowns.md`).

## State

### `AgentState` (top-level, `core/state.py:18`)

Logical groups:

- **Input**: `messages`, `client_id`, `client_name`, `sp_id`, `request_id`
- **User context**: `user_name`, `user_id`, `thread_id`, `conversation_id`, `task_type`
- **Routing**: `intent`, `original_question`, `rewritten_question`
- **Output**: `response_text`, `response_items`, `acknowledgment_text`
- **Memory injection**: `ltm_context`, `conversation_history`
- **Genie data (phase 1)**: `genie_summary`, `genie_sql`, `genie_table`, `genie_tables`, `genie_insights`, `genie_query_description`, `genie_columns`, `genie_data_array`, `genie_text_content`, `follow_up_suggestions`
- **Supervisor output**: `supervisor_json`
- **Planning**: `plan`, `plan_count`, `worker_results` (reducer: `operator.add`)
- **Sequential capability loop**: `current_step_index`, `current_task`, `current_capability`, `step_results` (reducer: `operator.add`)
- **Metadata**: `llm_call_count`, `genie_trace_id`, `genie_retry_count`, `error`

### `CapabilityState` (subgraph, `core/state.py:85`)

`user_query`, `cid`, `current_task` (inputs) / `messages`, `iterations_used` (ReAct) / `result`, `tool_calls`, `confidence`, `error` (outputs).

## Memory Model

| Layer | Scope | Implementation |
|-------|-------|----------------|
| **Working** | Single request | `AgentState` in-memory only |
| **Session (STM)** | Single conversation | Lakebase `CheckpointSaver` keyed on `thread_id`; trimmed by `memory/stm_trimmer.py`; serialized into prompt via `memory/context_formatter.py` |
| **Client / User (LTM)** | Cross-session, per `user_id` | `memory/ltm_manager.py` over `databricks-langchain DatabricksStore`; semantic search via `databricks-gte-large-en` (1024 dims) |
| **Analytical cache** | Project-wide | **Not implemented** (flagged in `04_code_audit.md`) |

## Orchestration Pattern

Hierarchical supervisor with optional subgraph delegation:

- The **supervisor layer** (intent/router nodes) is a cheap classifier loop.
- The **capability layer** (insight agent ReAct) is an expensive reasoning loop executed per plan step.
- The **deterministic output layer** (`format_supervisor`) separates structured data (programmatic) from narrative/visual content (LLM-constrained by strict JSON schema) — this is the contract surface with the UI.
