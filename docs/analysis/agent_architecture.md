# 4. Agent Architecture

Inventory of every agent / node / sub-component in the repo, with role, LLM, prompt, and wiring status.

---

## 4.1 Agents in the compiled graph

| Agent / node | File | Kind | LLM | Prompt | Role |
|--------------|------|------|-----|--------|------|
| **supervisor_classify** | `agents/supervisor.py:supervisor_node` | Single node, returns `Command(goto=...)` | GPT-5-2 (via AI Gateway) | `prompts/supervisor_prompt.py` (CLASSIFY_AND_REWRITE section) | Intent classification + query rewrite; 4 intents |
| **greeting** | `agents/greeting.py` | Terminal leaf | none | none | Hardcoded/LTM-personalized greeting |
| **clarification** | `agents/clarification.py` | Terminal leaf | GPT-5-2 | simple inline | Meta-answer about the conversation |
| **out_of_scope** | `agents/` (node defined in graph) | Terminal leaf | none | canned | Rejects non-marketing questions |
| **campaign_insight_agent** | `agents/campaign_insight/agent.py:CampaignInsightAgent.run` | Async node, 5 internal phases | GPT-5-2 | multiple | Core analytics pipeline |
| **supervisor_synthesize** | `supervisor/synthesizer.py` wrapped by `core/graph.py` | Compose node | GPT-5-2 | `supervisor_prompt.py` (Response Structure section) + `RESPONSE_FORMAT_SCHEMA` | Emit strict-JSON `response_items` |

---

## 4.2 CampaignInsightAgent internal components

All live under `agents/campaign_insight/`. Not LangGraph nodes — sequential async calls inside `run()`.

| Component | File | LLM | Responsibility |
|-----------|------|-----|----------------|
| AdaptivePlanner | `adaptive_planner.py` | GPT-5-2 | Build `ExecutionPlan` of `PlanStep`s |
| ReActExecutor | `executor.py` | GPT-5-2 | Reason → Genie call → observe → evaluate (≤3 iters, 120s) |
| ToolHandler | `tool_handler.py` | none | Translate NL to Genie REST calls |
| DimensionClassifier | `dimension_classifier.py` | GPT-5-2 + rules | Tag columns as TIME / CHANNEL / METRIC / AUDIENCE / CONTENT |
| DimensionValidator | `dimension_validator.py` | none | Rule-based: verify classified dims match intent |
| TableAnalyzer | `table_analyzer.py` | none | Row/col counts, numeric stats, basic anomalies |
| TableBuilder | `table_builder.py` | none | 2D → `DisplayTable` for UI |
| Interpreter | `interpreter.py` | GPT-5-2 | Pattern classification on `DisplayTable` |
| Recommender | `recommender.py` | GPT-5-2 | 2–3 Apply/Avoid/Explore recommendations |
| Reflector | `reflector.py` | GPT-5-2 | Self-correct interpretations + recs |
| ChartBuilder | `chart_builder.py` | GPT-5-2 | Emit Highcharts 11.x JSON |
| OutputBuilder | `output_builder.py` | none | Assemble `SubagentOutput` |

---

## 4.3 Legacy agents (importable, not wired into compiled graph)

| File | Original role |
|------|---------------|
| `agents/planner.py` | Decompose complex query into sub-steps |
| `agents/genie_agent.py` | Fetch raw data via Genie |
| `agents/genie_analysis.py` | LLM analysis over genie result |
| `agents/genie_worker.py` | Parallel (`Send` API) per-sub-question workers |
| `agents/synthesizer.py` | Merge worker outputs |
| `capabilities/insight_agent.py` | `insight_agent_node` — superseded by CampaignInsightAgent |
| `supervisor/intent_classifier.py` | Duplicates greeting regex; unreferenced |
| `supervisor/router.py` | `SupervisorRouter`; unreferenced |
| `supervisor/planner.py` | `SupervisorPlanner`; unreferenced |

`supervisor/synthesizer.py` IS used (wrapped by `supervisor_synthesize_node` in `core/graph.py`).

---

## 4.4 Interaction model

- **Supervisor → leaf**: single hop via `Command(goto=...)`.
- **Supervisor → insight → synthesize**: 3-hop linear path.
- **Within insight agent**: strictly sequential phases; no parallelism among the 5 phases.
- **No ReAct across agents**: only inside ReActExecutor (per plan step).
- **Legacy parallel fan-out** (Send API) in `genie_worker.py` is not reachable from the runtime graph.

---

## 4.5 Intents (from `agents/supervisor.py:CLASSIFY_AND_REWRITE_PROMPT`, ~lines 42–68)

1. `SIMPLE_QUERY` — one data lookup.
2. `COMPLEX_QUERY` — multi-step / cross-channel.
3. `CLARIFICATION` — meta-question about prior turn.
4. `OUT_OF_SCOPE` — non-marketing.
5. (greeting handled pre-LLM by regex)

CLAUDE.md claims 5 intents including `DATA_ANALYSIS`, `DOMAIN_KNOWLEDGE`, `AMBIGUOUS`, `CONTINUATION`, `OUT_OF_SCOPE`. The code's taxonomy is different.

---

## 4.6 LLM configuration (actual)

```python
ChatOpenAI(
    model=settings.LLM_ENDPOINT_NAME,          # "databricks-gpt-5-2"
    base_url=settings.AI_GATEWAY_URL,          # Databricks AI Gateway /mlflow/v1
    temperature=0.0,
    api_key=<per-client SP token from OBO>
)
```

All agents share one model family — no Haiku/Sonnet split yet. `databricks-claude-haiku-4-5` / `databricks-claude-sonnet-4-5` referenced in `CLAUDE.md` but **not** in code/config.
