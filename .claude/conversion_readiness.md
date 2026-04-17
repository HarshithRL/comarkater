# Campaign Insight Agent — Subgraph Conversion Readiness

**Scope:** Map the 5 phases currently orchestrated by `agents/campaign_insight/agent.py` to concrete LangGraph patterns from `langgraph-skill` and `databricks-langchain-skill`.
**No code changes.** This document drives the Phase C work in `migration_plan.md`.

---

## 1. Today's orchestrator (what `agent.py` actually does)

`CampaignInsightAgent.run(state, config, writer)` runs a **linear, mostly-sync** 5-phase sequence inside a single LangGraph node. State is local Python variables, not a graph state. Streaming is via a `writer` callback passed down.

| Phase | Calls inside agent.py | Sync/async | Inputs read | Outputs produced | SSE emitted |
|---|---|---|---|---|---|
| **1 — Adaptive Plan** | `dim_classifier.classify()` → `dimension_validator.validate()` → `adaptive_planner.plan()` | sync | `query`, `intent`, `supervisor_plan`, `feature_flags` | `dim_config: DimensionClassification`, `plan` with `.steps` | `plan_ready` |
| **2 — ReAct Execute** | `await executor.execute_plan(plan, stream_callback=…)` | **async** | `plan`, `stream_callback` | `step_results` (dict/list of `StepResult`) | per-step progress via callback |
| **3 — Interpret + Recommend** | `interpreter.interpret()` → `recommender.recommend()` | sync | `step_results`, `intent` | `interpretation`, `recommendations` | — |
| **4 — Reflect** | `reflector.verify()` | sync | `interpretation`, `recommendations`, `step_results` | `verified`; `final_interp = verified.interpretation or interpretation`; `final_recs = verified.recommendations or recommendations` | — |
| **5 — Build Output** | `chart_builder.build_chart()` → `_collect_caveats()` → `output_builder.build()` | sync | `intent`, `step_results`, `dim_config`, `verified`, `final_interp`, `final_recs`, `metadata` | `SubagentOutput` returned as `{"subagent_output": output}` | — |

**Branching:** none. All phases always execute. Phase 4 only chooses between `verified.*` and the original.
**Error path:** one try/except around all 5 phases → `SubagentOutput(status=AgentStatus.ERROR, …)`.
**Config dependency:** `config["configurable"]["sp_token"]` is threaded to `ChatOpenAI` and `ToolHandler`.

---

## 2. Target topology

A **compiled `StateGraph(SubagentState)`** with 8 first-class nodes (Phase 1 splits into 3, Phase 3 splits into 2). Parent graph adds the compiled subgraph as a node via `parent.add_node("campaign_insight_agent", compiled_subgraph)`.

```
subgraph START
  └─ dimension_classifier         (Phase 1A, LLM)
  └─ dimension_validator          (Phase 1B, deterministic)
  └─ adaptive_planner             (Phase 1C, LLM; emits plan_ready)
        │ conditional_edges
        ├─ Send(fan-out) → executor  (if independent steps)
        └─ sequential → executor     (otherwise)
  └─ executor                     (Phase 2, ReAct loop; emits step_progress / data_table)
        │ conditional_edges
        ├─ executor (next step)
        └─ interpreter (done)
  └─ interpreter                  (Phase 3a; emits analysis)
        │ conditional_edges
        ├─ recommender             (intent ∈ {diagnostic, strategic_recommendation})
        └─ reflector               (skip recommender)
  └─ recommender                  (Phase 3b; emits recommendations)
  └─ reflector                    (Phase 4)
  └─ output_builder               (Phase 5; emits chart, complete)
subgraph END
```

---

## 3. Skill patterns applied per phase

### 3.1 Shared scaffolding (applies to all nodes)

**State schema — `agents/campaign_insight/state.py` (NEW):**
```python
from typing import Annotated, Optional, TypedDict
import operator

class SubagentState(TypedDict):
    # input contract (read-only)
    request_id: str
    query: str
    intent: dict
    supervisor_plan: list
    feature_flags: dict

    # phase 1 outputs
    dim_config: Optional[dict]
    plan: Optional[dict]

    # phase 2 accumulation — reducer so Send fan-out doesn't collide
    step_results: Annotated[list, operator.add]
    step_index: int

    # phase 3/4/5
    interpretation: Optional[dict]
    recommendations: Optional[list]
    reflection: Optional[dict]
    caveats: list
    chart: Optional[dict]
    output: Optional[dict]
```
**Skill rule:** keys **without** `Annotated[..., reducer]` are overwritten on each node return; concurrent writes to the same non-reduced key = runtime error. Only `step_results` needs a reducer because Send fan-out can produce parallel writes.

**Runtime context — avoid putting `sp_token`, client_id in state:**
```python
@dataclass
class SubagentContext:
    sp_token: str
    client_id: str
    genie_space_id: str
    model_name: str
```
Accessed inside nodes as `runtime: Runtime[SubagentContext]` per langgraph-skill.

**Parent integration:** `compiled_subgraph = build_campaign_insight_graph().compile()` then `parent.add_node("campaign_insight_agent", compiled_subgraph)`. Per skill, **default `.compile()` (checkpointer=None)** is correct — the subgraph inherits the parent's checkpointer per invocation and does NOT accumulate state across parent invocations. `.compile(checkpointer=True)` is wrong here (would collide across parallel calls).

**Async rule:** executor is async → entire subgraph is invoked via `ainvoke` / `astream`. Skill: *"AsyncCheckpointSaver only works with ainvoke/astream/abatch; sync invoke silently misbehaves."* Parent must call `astream`.

**Tracing:** `mlflow.langchain.autolog()` once at serving entry. Each node becomes its own span automatically.

---

### 3.2 Phase 1A → `dimension_classifier` node

**What it does today:** LLM call that returns a `DimensionClassification`.
**Pattern needed:** plain node returning `dict` — no routing decision.

```python
def dimension_classifier_node(state: SubagentState, runtime) -> dict:
    result = DimensionClassifier(...).classify(state["query"], state["intent"])
    return {"dim_config": result.model_dump()}
```
**Skill element:** `with_structured_output(DimensionClassification)` on a `ChatDatabricks` instance for deterministic schema — already used internally.
**Edge:** static `add_edge("dimension_classifier", "dimension_validator")`.

---

### 3.3 Phase 1B → `dimension_validator` node

**What it does today:** deterministic Python validation, feature-flag gating, budget clamping.
**Pattern needed:** plain node, no LLM.

```python
def dimension_validator_node(state: SubagentState, runtime) -> dict:
    validated = DimensionValidator().validate(
        state["dim_config"], state["query"], state["feature_flags"]
    )
    return {"dim_config": validated.model_dump()}
```
**Edge:** static `add_edge("dimension_validator", "adaptive_planner")`.

---

### 3.4 Phase 1C → `adaptive_planner` node

**What it does today:** builds the plan, then `_safe_stream(writer, {"event_type": "plan_ready", …})`.
**Pattern needed:** node emits a custom stream event.

The langgraph-skill shows streaming via `astream()` modes; to push arbitrary in-node events use LangGraph's `get_stream_writer()` (skill references it for custom streaming). Replace `_safe_stream(writer, …)` with:
```python
from langgraph.config import get_stream_writer

def adaptive_planner_node(state: SubagentState, runtime) -> dict:
    plan = AdaptivePlanner().plan(
        state["query"], state["intent"], state["dim_config"], state["supervisor_plan"]
    )
    get_stream_writer()({"event_type": "plan_ready",
                         "steps": [s.purpose for s in plan.steps]})
    return {"plan": plan.model_dump(), "step_index": 0, "step_results": []}
```
**Edge (Send fan-out):** this is the first place Send API applies.
```python
def plan_dispatch(state) -> list[Send] | str:
    plan = state["plan"]
    if plan.get("parallelizable") and len(plan["independent_steps"]) > 1:
        return [Send("executor", {"step": s, "step_index": i})
                for i, s in enumerate(plan["independent_steps"])]
    return "executor"

builder.add_conditional_edges("adaptive_planner", plan_dispatch, ["executor"])
```
**Skill rule:** *annotate Command/conditional returns with `Literal[...]` so the graph renderer draws edges*. For pure `add_conditional_edges`, pass the explicit target list as third arg (shown above).

**Send payload rule:** each `Send("executor", {...})` delivers its own partial state slice — the receiving node must merge via the `operator.add` reducer on `step_results`. This is why `step_results: Annotated[list, operator.add]` is mandatory.

---

### 3.5 Phase 2 → `executor` node (loop)

**What it does today:** `await executor.execute_plan(plan, stream_callback=…)` — the entire per-step ReAct loop runs inside one async call.
**Pattern needed:** convert inner loop to a **graph-level loop** — executor node processes one step per invocation, conditional edge loops it back to itself until `step_index == len(plan.steps)`, then routes to `interpreter`.

```python
async def executor_node(state: SubagentState, runtime) -> dict:
    step = state["plan"]["steps"][state["step_index"]]
    result = await ReActExecutor(...).execute_step(step)
    get_stream_writer()({"event_type": "step_progress",
                         "step_id": step["id"], "status": result.status})
    if result.table is not None:
        get_stream_writer()({"event_type": "data_table",
                             "step_id": step["id"], "rows": result.table.rows_for_display})
    return {"step_results": [result.model_dump()],   # reducer appends
            "step_index": state["step_index"] + 1}
```
**Edge (self-loop):**
```python
def executor_router(state) -> str:
    return "executor" if state["step_index"] < len(state["plan"]["steps"]) else "interpreter"

builder.add_conditional_edges("executor", executor_router, ["executor", "interpreter"])
```
**Skill rule:** `async def` node **must** run under `ainvoke` / `astream`. The executor's own ReAct loop (reason→act→observe→evaluate) stays inside the node; only the **across-step** loop becomes graph edges.

**Genie access inside executor:** already goes through `tool_handler`. After the Phase B move, imports change from `agents.genie_client` → `tools.genie_client`. The skill anti-pattern ("don't build your own Postgres saver") has an analogue here: prefer `databricks_langchain.genie.GenieAgent` if we ever drop our REST client, but for now REST is intentional (per CLAUDE.md: "migrating from REST to MCP").

---

### 3.6 Phase 3a → `interpreter` node

**What it does today:** `interpreter.interpret(step_results, intent)`.
**Pattern needed:** plain node + conditional out-edge to skip recommender for non-diagnostic intents.

```python
def interpreter_node(state: SubagentState, runtime) -> dict:
    interp = Interpreter(...).interpret(state["step_results"], state["intent"])
    get_stream_writer()({"event_type": "analysis", "summary": interp.summary})
    return {"interpretation": interp.model_dump()}

def interpreter_router(state) -> str:
    return "recommender" if state["intent"]["intent_type"] in {
        "diagnostic", "strategic_recommendation"
    } else "reflector"

builder.add_conditional_edges("interpreter", interpreter_router, ["recommender", "reflector"])
```
**Skill element:** simple conditional edge by return value.

---

### 3.7 Phase 3b → `recommender` node (conditional)

**What it does today:** `recommender.recommend(interpretation, step_results, intent)`.

```python
def recommender_node(state: SubagentState, runtime) -> dict:
    recs = Recommender(...).recommend(
        Interpretation(**state["interpretation"]),
        state["step_results"], state["intent"],
    )
    get_stream_writer()({"event_type": "recommendations",
                         "count": len(recs)})
    return {"recommendations": [r.model_dump() for r in recs]}
```
**Edge:** static `add_edge("recommender", "reflector")`.

---

### 3.8 Phase 4 → `reflector` node

**What it does today:** `reflector.verify(...)`, then picks `verified.interpretation or interpretation`.
**Pattern needed:** node applies verification and writes chosen outputs. Selection is not branching — it's in-node value choice — so no edge logic.

```python
def reflector_node(state: SubagentState, runtime) -> dict:
    verified = Reflector(...).verify(
        state["interpretation"], state.get("recommendations", []), state["step_results"]
    )
    return {
        "interpretation": verified.interpretation or state["interpretation"],
        "recommendations": verified.recommendations or state.get("recommendations", []),
        "reflection": {"issues_found": verified.issues_found},
    }
```
**Edge:** static `add_edge("reflector", "output_builder")`.

---

### 3.9 Phase 5 → `output_builder` node

**What it does today:** builds chart, collects caveats, assembles `SubagentOutput`, returns `{"subagent_output": output}`.
**Pattern needed:** terminal node; emits `chart` and `complete` SSE; writes `output` into `SubagentState`. Parent graph reads `state["output"]` after subgraph returns.

```python
def output_builder_node(state: SubagentState, runtime) -> dict:
    chart = ChartBuilder(...).build_chart(state["intent"]["intent_type"], state["step_results"])
    caveats = _collect_caveats(state["step_results"], state["dim_config"])
    get_stream_writer()({"event_type": "chart", "spec": chart})
    out = OutputBuilder().build(
        state["request_id"], state["step_results"],
        state["interpretation"], state.get("recommendations", []),
        chart, caveats, {...metadata...},
    )
    get_stream_writer()({"event_type": "complete"})
    return {"chart": chart, "caveats": caveats, "output": out.model_dump()}
```
**Edge:** `add_edge("output_builder", END)`.

**Parent-side read:** after `compiled_subgraph.ainvoke(parent_slice)` returns, parent writes `state["subagent_output"] = result["output"]` so `supervisor_synthesize` can consume it via the existing contract. Alternative: use **`Command(update=…, goto="supervisor_synthesize", graph=Command.PARENT)`** inside `output_builder` to atomically hand control back to parent. Skill explicitly documents this pattern: *"Inside a worker, hand control back to parent with `Command.PARENT`."*

---

## 4. Error handling

Today: one try/except wraps all 5 phases.
Target: a graph-level error policy. Options per skill:

1. **Per-node try/except** that writes an `error` field + routes to `output_builder` with `status=ERROR`. Simple; keeps current behavior.
2. **LangGraph's built-in `on_error` node hook** (skill mentions retry/error policies on nodes). Use this for transient Genie errors; surfaces errors into traces without custom boilerplate.

Recommendation: option 1 for Phase C cutover (least risk), option 2 later.

---

## 5. Streaming event mapping

The langgraph-skill's `get_stream_writer()` pattern replaces the current `writer` callback. Parent invokes with `graph.astream(..., stream_mode="custom")` (and optionally combine with `"updates"` for per-node diffs). 8 SSE events map:

| Vision event | Emitted by node |
|---|---|
| `acknowledgment` | outside subgraph (ack_orchestrator, pre-graph) |
| `plan_ready` | `adaptive_planner` |
| `step_progress` | `executor` (each iteration) |
| `data_table` | `executor` (when table present) |
| `chart` | `output_builder` |
| `analysis` | `interpreter` |
| `recommendations` | `recommender` |
| `complete` | `output_builder` |

---

## 6. Checkpointer / memory

Skill-mandated setup for Databricks:
- **STM:** `AsyncCheckpointSaver(instance_name=…)` + `await checkpointer.setup()` at startup.
- **LTM:** `DatabricksStore(instance_name=…)` passed to `parent.compile(checkpointer=…, store=…)`.
- **NEVER** use raw `AsyncPostgresSaver` with Lakebase connection string (explicit skill anti-pattern).
- Subgraph inherits parent's checkpointer automatically via default `.compile()` — do **not** pass a second checkpointer to the subgraph.

Existing `memory/ltm_manager.py` stays; only the wiring in the to-be-renamed `memory/stm_manager.py` needs to swap to `AsyncCheckpointSaver` if it isn't already.

---

## 7. LLM client choice

`agent.py` currently uses `ChatOpenAI(api_key=sp_token, base_url=AI_GATEWAY_URL)`. Per databricks-langchain-skill, on Databricks serving the idiomatic path is `ChatDatabricks(endpoint=…)` with OBO auth via `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())`. Switching is optional for Phase C — the subgraph conversion can keep `ChatOpenAI` short-term. Flag for `migration_plan.md` open-question #4.

---

## 8. Readiness summary

| Requirement | Status |
|---|---|
| Discrete, composable phase modules | ✅ Already 14 modules, each a class with a single entrypoint |
| Clean phase I/O (no hidden globals) | ✅ All inputs explicit, outputs are dataclasses |
| Async tolerance | ✅ Only `executor` is async; rest are sync-safe |
| Contracts (`contracts.py`) | ✅ `SubagentOutput`, `DimensionClassification`, `StepResult` etc. already dataclass-based |
| `SubagentState` TypedDict | ❌ Must be created |
| `graph.py` builder | ❌ Must be created |
| Per-node `*_node(state)` wrappers | ❌ Must be added (thin; delegate to existing classes) |
| Reducer on `step_results` | ❌ Required for Send fan-out safety |
| `get_stream_writer()` adoption | ❌ Replace `_safe_stream(writer, …)` |
| Parent wiring as compiled subgraph | ❌ Replace `campaign_insight_agent_node` in `core/graph.py` |
| Checkpointer semantics | ✅ Default `.compile()` is correct; no change needed |
| Streaming events mapped | ✅ 6/8 already emitted; 2 to add (`analysis`, `recommendations`) |

**Conclusion:** the orchestrator is **cleanly shaped** for conversion. No phase has hidden state, side effects in constructors, or shared mutable globals. The conversion is mechanical: add `state.py` + `graph.py`, add 8 node wrappers, replace the writer callback, and swap the parent-graph node reference. No phase logic needs to change.

**Estimated delta:** ~300 new LOC (state + graph + node wrappers), ~50 LOC removed from `agent.py` (becomes a thin `build_campaign_insight_graph()` entry), zero change to the 14 existing phase modules' internals.
