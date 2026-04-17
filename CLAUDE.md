# CoMarketer — AI Marketing Analytics Agent

## What This Is
Multi-agent system: CoMarketer (supervisor) routes queries to Insight Agent (analytics subgraph).
Built on LangGraph + Databricks. Deployed as Databricks App with SSE streaming.
Serves Netcore Cloud clients (IGP, Crocs, Pepe Jeans) with campaign analytics.

## Quick Commands
- Run locally: `cd agent_server && python start_server.py`
- Test: `pytest agent_server/tests/ -v`
- Lint: `ruff check agent_server/`
- Deploy: `databricks bundle deploy --profile DEFAULT`
- App deploy: `databricks apps deploy comarketer --source-code-path agent_server --profile DEFAULT`

## Architecture (MUST FOLLOW — from execution doc)
- **Hierarchical Supervisor**: CoMarketer (Haiku, fast/cheap) → Insight Agent subgraph (Sonnet, reasoning)
- **CoMarketer nodes**: load_context → intent_classifier → router → response_orchestrator → follow_up_handler
- **Insight Agent nodes**: query_planner → genie_execution → validator → stats_engine → response_builder → anomaly_scanner
- **Intents**: DATA_ANALYSIS, DOMAIN_KNOWLEDGE, AMBIGUOUS, CONTINUATION, OUT_OF_SCOPE
- **Data access**: Genie managed MCP (migrating from REST API)
- **Memory**: 4 layers — Working (in-memory) → Session (Lakebase checkpoint) → Client (Delta/Store) → Analytical Cache (Delta, daily refresh)
- **Streaming**: SSE with 8 phases — thought → plan → progress → data_table → chart → analysis → recommendations → complete

## Current Repo State (from audit 2026-03-30)
- **Working**: Graph with 9 nodes, Genie REST API, OBO auth, parallel fan-out (Send API), LTM (Lakebase), Highcharts, Databricks App deployment
- **Partial**: Intent classification (4 of 5), streaming (6 of 8 phases), data compression, tracing
- **Missing**: Stats engine (11 derived metrics), anomaly scanner, filter persistence, error recovery chain, analytical cache
- **Key file**: `agent.py` is 700 lines — needs splitting per 200-line rule

## Code Style
- Python 3.11+ with type hints on ALL function signatures
- Google-style docstrings on all public methods
- Pydantic models for all data validation
- One file = one responsibility, max 200 lines per file
- Import order: stdlib → third-party → local
- Use `logging` module, never `print()`
- Async: `asyncio.run()` inside sync LangGraph nodes (documented design decision)

## Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- State fields: `snake_case`
- Span names: `{agent_name}.{node_name}[.{sub_op}]`

## Critical Rules (NON-NEGOTIABLE)
- NEVER hallucinate Databricks/LangGraph APIs — use skills for reference
- NEVER hardcode credentials — use env vars or Databricks secrets scope
- ALL SQL must be parameterized (no string interpolation)
- ALL derived metrics (CTR, CVR, etc.) computed in Python, NEVER by LLM
- ALL numbers in responses must trace back to actual data — zero tolerance for invented metrics
- No currency symbols in output — use "units"
- Never display "genie", "Databricks", or internal system names to users
- RLS enforced at infrastructure level (Unity Catalog) — agent NEVER adds WHERE cid manually
- Every new module MUST have corresponding tests
- Format contract: commas for counts (2,38,100), 2 decimal + % for rates (3.20%), no currency symbols

## File Layout
```
agent_server/
├── agent.py              # CoMarketerAgent (ResponsesAgent wrapper)
├── start_server.py       # Entry: AgentServer + SSE middleware
├── agents/               # All agent nodes (one file per node)
├── core/                 # State, graph, config, auth, tracing
├── prompts/              # All prompt templates (one file per prompt)
├── parsers/              # Formatters, validators, truncators
├── memory/               # LTM, STM, filter store, cache
├── ui/                   # Flask routes + static assets
└── tests/                # pytest tests
```

## LLM Configuration
- Supervisor tasks (intent, ack, follow-up): `databricks-claude-haiku-4-5` via ChatDatabricks
- Reasoning tasks (query plan, response build, analysis): `databricks-claude-sonnet-4-5` via ChatDatabricks
- Current: `databricks-gpt-5-2` via ChatOpenAI + AI Gateway (being migrated)
- Temperature: 0.0 for all nodes
- Structured output: JSON schema with `strict: True` on format_supervisor

## Key Integrations
- **Genie**: Migrating from REST API (`genie_client.py`) to managed MCP (`{host}/api/2.0/mcp/genie/{space_id}`)
- **MLflow 3**: `mlflow.langchain.autolog()` + custom spans + Delta trace table
- **Lakebase**: CheckpointSaver (STM) + DatabricksStore (LTM)
- **Unity Catalog**: `channel.gold_channel.campaign_details` (single source table, RLS on cid)

## Domain Quick Reference
- Channels: Email, SMS, APN (mobile push), BPN (browser push), WhatsApp
- Measures: sent, delivered, opened, clicked, conversion, revenue, bounce, unsubscribe, complaint
- 11 derived rates: CTR, CVR, Open Rate, Bounce Rate, Unsub Rate, Complaint Rate, Delivery Rate, CTOR, Rev/Delivered, Rev/Click, Conv from Click
- 12 query verticals: A (simple lookup) through L (compound multi-vertical)
- See domain-marketing skill for full reference

## When Compacting
ALWAYS preserve: current implementation phase, list of modified files, test status, which components are done vs pending, any architecture decisions made in this session.
