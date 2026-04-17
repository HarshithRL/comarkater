# 04 — Code Quality Audit

*Findings are grounded in file reads / greps. Claims dependent on CLAUDE.md are marked `[CLAUDE.md]`. Unresolved items moved to `06_unknowns.md`.*

## 1. Oversized Files (vs. 200-line rule in CLAUDE.md)

| File | LOC | Verdict |
|------|----:|---------|
| `agent_server/agent.py` | 1,441 | **Severe** — acknowledged in CLAUDE.md ("needs splitting"). Single god-class holding request shaping, graph invocation, SSE emission, memory I/O, error handling. |
| `prompts/supervisor_prompt.py` | 1,062 | Acceptable — almost entirely static prompt text + few-shot examples. Natural candidate for YAML/MD externalization. |
| `agents/agentbricks.py` | 479 | Over-limit — contains the alternative AgentBricks Genie path + response parsers + retry logic. Could split response parsing into `parsers/`. |
| `core/graph.py` | 374 | Over-limit — `format_supervisor_node` (~145 LOC) embedded in the file that should only build/compile the graph. Extracting it to `agents/format_supervisor.py` would restore single-responsibility. |
| `memory/ltm_manager.py` | 338 | Over-limit — mixes CRUD, semantic search, formatting. |
| `tools/registry.py` | 297 | Over-limit — one file registering all tools. |

## 2. Missing Components `[CLAUDE.md]`

Grepped codebase: **no file** mentions `stats_engine`, `anomaly_scanner`, `analytical_cache`, or `filter_store`. Confirms the following from CLAUDE.md's "Missing" list:

- **Stats engine** (the 11 derived rates — CTR, CVR, Open Rate, Bounce Rate, Unsub Rate, Complaint Rate, Delivery Rate, CTOR, Rev/Delivered, Rev/Click, Conv from Click) — not implemented. Violates the "ALL derived metrics computed in Python, NEVER by LLM" rule — right now any such number would be synthesized by the LLM inside `genie_analysis_node` or `format_supervisor_node`.
- **Anomaly scanner** — not implemented.
- **Filter persistence** — no filter store, filters must be re-specified per turn.
- **Error recovery chain** — `genie_retry_count` exists in state but no node consumes it to trigger a recovery loop.
- **Analytical cache** — no Delta-backed cache layer; every request re-queries Genie.

## 3. Partial Components

| Component | Observation |
|-----------|-------------|
| Intent classification | Code defines 4 intent values (`greeting`, `data_query`, `complex_query`, `clarification`). CLAUDE.md specifies 5 (`DATA_ANALYSIS`, `DOMAIN_KNOWLEDGE`, `AMBIGUOUS`, `CONTINUATION`, `OUT_OF_SCOPE`). Taxonomy drift — neither set matches the other. |
| SSE streaming | `get_stream_writer` used in `format_supervisor_node` and implied in several agents. Of the 8 documented phases, `thought`, `plan`, `progress`, `data_table`, `chart`, `analysis` are plausibly covered; `recommendations` + `complete` emission paths not fully verified. |
| Tracing | MLflow spans present in `format_supervisor_node` and `core/tracing.py`, but Delta trace-table writer has a `_sql_warned` fallback flag indicating silent-degradation mode when the SQL warehouse is unreachable. |
| Data compression | `table_truncator.py` + `BUDGET_FORMAT_SUPERVISOR` present; `genie_data_array`/`genie_columns` cleared post-format. No summary/paging strategy for very large result sets. |

## 4. Migration Debt

| From | To | Status |
|------|----|--------|
| Genie REST (`agents/genie_client.py` → `genie_agent.py`) | Managed MCP (`agents/agentbricks.py`) | **Both present, both wired** — `genie_data_node` (REST) is on the default route; `agentbricks_node` exists but is not referenced in `core/graph.py`. Risk: stale drift. |
| `ChatOpenAI` + AI Gateway (`databricks-gpt-5-2`) | `ChatDatabricks` + `databricks-claude-haiku-4-5` / `databricks-claude-sonnet-4-5` `[CLAUDE.md]` | **Not started in code** — `core/graph.py:32` imports `ChatOpenAI`; `databricks.yml` sets `LLM_ENDPOINT_NAME=databricks-gpt-5-2`. |
| Supervisor few-shots embedded in `supervisor_prompt.py` | External examples file | Not started. |

## 5. Duplication / Overlap

| Cluster | Files | Status |
|---------|-------|--------|
| Genie access | `genie_client.py`, `genie_agent.py`, `genie_analysis.py`, `genie_worker.py`, `agentbricks.py` | **Not pure duplicates** — client vs. simple-path node vs. analysis LLM vs. legacy fan-out vs. MCP alternative. But `agentbricks.py`'s `_error_response` (line 464) mirrors `genie_agent.py`'s `_error_response` (line 76) — drift hazard. |
| Registries | `agents/registry.py`, `agents/capability_registry.py`, `agents/agent_model.py`, `capabilities/registry.py` | Four registry-ish files for what appears to be one concept. Boundary unclear (`06_unknowns.md`). |
| Error-response builders | `_error_response` in `genie_agent.py` + `agentbricks.py` | Identical intent, separate code paths. |

## 6. Test Coverage Gap

- `agent_server/tests/` contains one 78-line `test_local.py`.
- CLAUDE.md: "Every new module MUST have corresponding tests" — ~50 Python modules, 1 test file. **Non-compliance**.
- `notebooks/` holds 17 scripts used as ad-hoc integration/diagnostic tests — not executable as part of CI.

## 7. Style / Conventions

- `print()` in `start_server.py` for Flask mount + MLflow tracking fallbacks — violates "use `logging` module, never `print()`". Other files consistently use `logging`.
- Long transient fields cleared in `format_supervisor_node` (correct — keeps checkpoints small).
- `agent_server/ui/routes.py:40` hard-codes `ALLOWED_SP_IDS = {...}` — whitelist works for 3 clients but must move to config for scale.

## 8. Observability Smells

- `secrets_loader._init_failed`, `trace_logger._sql_warned`, `trace_logger._sql_client is None` — all silent-degradation flags surfaced only via `/ui/debug`. No metric / alert path.
