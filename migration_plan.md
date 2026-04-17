# CoMarketer — Migration Plan: Current Flat Architecture → v4.1 Two-Layer Supervisor

**Date:** 2026-04-16
**Source of truth:** `comarketer_vision_v4.md` + current repo audit
**Status:** PLAN ONLY — no code changes until explicitly approved

---

## 0. TL;DR

The repo is ~70% migrated toward v4 already. A new `supervisor/` package and `agents/campaign_insight/` subagent exist alongside legacy nodes. The remaining work is:

1. **Convert Campaign Insight Agent from a single Python-class node into a true compiled LangGraph subgraph** (5 phases become 5 graph nodes with their own StateGraph and SubagentState).
2. **Cut over the parent graph** to invoke the compiled subgraph and delete every legacy path.
3. **Slim the supervisor prompt** from 1,062 LOC to ~150 LOC (move content into YAML).
4. **Stand up the agent registry** so topology is config-driven.
5. **Delete ~3,000 LOC of dead legacy code** (old supervisor, old synthesizer, genie_* nodes, agentbricks, capabilities/).

No Genie contract changes. No auth changes. No memory-schema changes.

---

## 1. Current State (from 2026-04-16 audit)

### 1.1 What is already v4-shaped
| Area | State |
|---|---|
| `supervisor/` package (intent_classifier, planner, router, synthesizer, supervisor_node, domain_context) | ✅ Exists, wired into `core/graph.py` |
| `agents/campaign_insight/` with 14 modules + 3 prompts + 8 YAMLs | ✅ Exists; 5-phase orchestrator runs as **one** Python class inside **one** LangGraph node |
| Domain YAMLs (domain_context, metrics, interpretation, recommendations, constraints, intent, audience_knowledge, content_knowledge) | ✅ All 8 present (1,810 LOC) |
| `core/graph.py` topology | ✅ 6 nodes: supervisor_classify → {greeting, clarification, out_of_scope, campaign_insight_agent} → supervisor_synthesize → END |
| Memory (STM checkpointer + LTM store) | ✅ Present in `memory/` |
| Multi-tenant auth + SecureTokenProvider | ✅ Production-grade in `core/auth.py` |
| MLflow tracing scaffold | ✅ `core/tracing.py` |

### 1.2 What is NOT yet v4.1
| Gap | Impact |
|---|---|
| Campaign Insight Agent is a **Python class invoked in a single node** (`graph.py:campaign_insight_agent_node`), not a compiled LangGraph subgraph | No per-phase streaming, no per-phase trace spans as first-class nodes, no checkpoint resume at phase granularity, no Send-API fan-out for parallel ReAct steps |
| `agents/supervisor.py`, `agents/synthesizer.py`, `agents/planner.py`, `agents/genie_agent.py`, `agents/genie_analysis.py`, `agents/genie_worker.py`, `agents/genie_client.py` still present and importable | Dead-but-importable code; `tools/registry.py` and `agents/campaign_insight/tool_handler.py` still pull from `agents.genie_client` |
| `agents/agentbricks.py` (479 LOC) | Dead code per vision §20 |
| `capabilities/` (insight_agent.py, registry.py) | Dead subgraph wrapper from an earlier design |
| `agents/registry.py` | Stub — read but graph.py imports nodes directly |
| `prompts/supervisor_prompt.py` (1,062 LOC) | Must shrink to ~150 LOC per vision §5.5; remainder should be derived from YAMLs loaded by `SupervisorDomainContext` |
| `agent.py` at repo root (1,494 LOC) | Way over 200-line rule; mixes ResponsesAgent wrapper, streaming, memory injection, acknowledgment |
| `agents/acknowledgment.py`, `agents/followup.py` | Pre/post-graph features not in v4 topology — confirm keep or move |
| `parsers/subagent_parser.py` | Only used in legacy synth path once legacy is deleted |
| No `agents/base_agent.py` | Vision §19 requires abstract base for registry-driven dispatch |
| Streaming emits 6 of 8 SSE event types | Missing `plan_ready` and `step_progress` as distinct events (they fire but not at subgraph-node granularity) |

### 1.3 Current node topology (`core/graph.py`)
```
START
  └─ supervisor_classify  (supervisor_node.py)
        │ Command(goto=…)
        ├─ greeting                 → END
        ├─ clarification            → END
        ├─ out_of_scope             → END
        └─ campaign_insight_agent   ──► supervisor_synthesize ──► END
             (opaque: runs all 5 phases inside one node)
```

---

## 2. Target State (v4.1)

### 2.1 Parent graph (unchanged node count, one node becomes a subgraph)
```
START
  └─ supervisor_classify
        ├─ greeting                 → END
        ├─ clarification            → END
        ├─ out_of_scope             → END
        └─ campaign_insight_agent   ← compiled StateGraph (subgraph)
                                       ──► supervisor_synthesize ──► END
```

### 2.2 Campaign Insight subgraph (new — compiled as `StateGraph(SubagentState)`)
```
subgraph START
  └─ dimension_classifier           (Phase 1A, LLM)
  └─ dimension_validator            (Phase 1B, deterministic)
  └─ adaptive_planner               (Phase 1C, LLM)
        │ conditional_edge
        │  if steps_independent: Send(...) fan-out to executor
        │  else: sequential loop
  └─ executor                       (Phase 2, ReAct per step; loops via conditional edge)
        ↳ uses tool_handler (Genie) and table_analyzer internally
  └─ interpreter                    (Phase 3a)
  └─ recommender                    (Phase 3b)
        │ conditional_edge: skip if intent ∉ {diagnostic, strategic_recommendation}
  └─ reflector                      (Phase 4)
  └─ output_builder                 (Phase 5 — assembles SubagentOutput)
subgraph END
```

Each subgraph node is a first-class LangGraph node → independent trace span, independent SSE event, independent checkpoint boundary.

### 2.3 Directory layout (target, rooted at `agent_server/`)
Matches vision §19. Net deltas from today:
- NEW: `agents/base_agent.py`, `agents/registry.py` (real), `agents/campaign_insight/graph.py`, `agents/campaign_insight/state.py`
- MOVE: `auth.py`, tracing split into `utils/tracing.py`, memory manager renames
- DELETE: all legacy agents + capabilities + old prompts

---

## 3. Gap Analysis — Key Deltas

| # | Gap | Resolution |
|---|---|---|
| G1 | Subagent is a class, not a subgraph | Build `agents/campaign_insight/graph.py` + `state.py`; refactor 14 modules into node callables |
| G2 | No `SubagentState` TypedDict | Define in `agents/campaign_insight/state.py`; carry plan, step_results, interpretation, recommendations, reflection, output |
| G3 | Parent graph invokes class, not subgraph | In `core/graph.py` replace `campaign_insight_agent_node` with `graph_builder.add_node("campaign_insight_agent", compiled_subgraph)` |
| G4 | Legacy agents still importable | Delete 7 files; fix imports in `tools/registry.py`, `agents/campaign_insight/tool_handler.py` |
| G5 | Supervisor prompt 1,062 LOC | Rewrite to ~150 LOC consuming `SupervisorDomainContext` injections |
| G6 | `agents/registry.py` is stub | Implement `AgentRegistration` dataclass + AGENT_REGISTRY with real entries; `core/graph.py` reads registry instead of direct imports |
| G7 | Base-class absent | Create `agents/base_agent.py` with `run(state) -> SubagentOutput` abstract + `build_graph() -> CompiledGraph` |
| G8 | `agent.py` 1,494 LOC | Split into `agent.py` (≤200 LOC), `core/streaming.py`, `core/memory_injector.py` (move from memory/), and `core/ack_orchestrator.py` |
| G9 | Parallel ReAct absent | Phase 2 uses Send API on `adaptive_planner → executor` edge when `plan.parallelizable=True` |
| G10 | `domain_context.yaml` data_source section outdated | Merge `domain_context_data_source_update.yaml` in-place |
| G11 | Feedback not wired | `feedback.py` endpoint exists; add route in `start_server.py` + UI hook (acknowledged scope question) |
| G12 | `prompts/` vs `agents/campaign_insight/prompts/` split | Delete `prompts/supervisor_prompt.py` legacy bulk; move what supervisor keeps into `supervisor/prompts/supervisor_prompt.py` |

---

## 4. Migration Strategy — 6 Phases (Sequential)

Each phase is independently verifiable. No phase leaves the repo in a broken state.

### Phase A — Freeze & Snapshot (no code changes)
- Tag current commit as `pre-v4.1-migration`.
- Record current baseline: test pass rate, 30-question eval scores, latency p50/p95.
- Confirm open questions in §8 with you before Phase B.

### Phase B — Cleanup Dead Code (reversible deletes)
Goal: shrink surface area without changing runtime behavior.
- Delete dead files listed in §5.4.
- Fix import breaks (only `tools/registry.py` and `tool_handler.py` depend on `agents/genie_client.py` — move `genie_client.py` to `tools/genie_client.py` instead of deleting).
- Re-run tests. Ship.

### Phase C — Subagent Becomes a Subgraph
Goal: compile Campaign Insight as `StateGraph(SubagentState)`.
- Add `agents/campaign_insight/state.py` (SubagentState).
- Add `agents/campaign_insight/graph.py` (builder + compile).
- Convert each of the 14 modules' entry points from class methods to node callables `(state) -> dict`.
- `agents/campaign_insight/agent.py` shrinks to a thin wrapper that exposes `compiled_graph` and a `run()` method that invokes it.
- Parent `core/graph.py` wires the compiled subgraph as the `campaign_insight_agent` node (LangGraph supports `add_node(name, compiled_subgraph)` natively).
- Preserve the `SubagentOutput` contract at the subgraph boundary — no supervisor-side change needed.
- Add per-node MLflow spans (`campaign_insight.<phase>`).

### Phase D — Streaming Upgrade
Goal: deliver vision §11 (8 SSE events).
- Add `core/streaming.py` as the single SSE writer.
- Wire subgraph nodes to emit `plan_ready` (from adaptive_planner), `step_progress` (from executor per-iteration), `data_table` (from executor observe), `chart` (from output_builder), `analysis` (from interpreter), `recommendations` (from recommender), `complete` (from output_builder).
- Remove ad-hoc streaming calls inside `agent.py`.

### Phase E — Supervisor Slimming + Registry Activation
Goal: supervisor prompt ≤150 LOC; graph topology read from AGENT_REGISTRY.
- Rewrite `supervisor/prompts/supervisor_prompt.py` as 7-section template per vision §5.5, filled from `SupervisorDomainContext`.
- Delete `prompts/supervisor_prompt.py` (legacy 1,062 LOC).
- Build `agents/base_agent.py` + real `agents/registry.py`.
- `core/graph.py` builds parent graph by iterating `AGENT_REGISTRY` and calling `registration.build_subgraph()`.

### Phase F — Hygiene + Splits
Goal: adhere to 200-line rule; final file layout per vision §19.
- Split root `agent.py` (1,494 LOC) into: `agent.py` (ResponsesAgent wrapper), `core/streaming.py`, `core/memory_injector.py`, `core/ack_orchestrator.py`.
- Move `core/auth.py` → `auth/` package (client_resolver.py, secrets_loader.py, token_provider.py).
- Move `agents/genie_client.py` → `tools/genie_client.py`; add `tools/genie_rate_limiter.py` (currently inside `agents/genie_worker.py` logic — re-extract before worker is deleted).
- Wire `feedback.py` endpoint.

---

## 5. File-by-File Plan

Legend: **KEEP** unchanged · **MODIFY** edit in place · **CREATE** new file · **DELETE** remove · **MOVE** relocate/rename · **SPLIT** decompose into multiple files

### 5.1 `agent_server/` root
| File | Action | Detail |
|---|---|---|
| `agent.py` (1,494) | **SPLIT** | → thin `agent.py` (~150 LOC: ResponsesAgent wrapper + predict/stream dispatch) + `core/streaming.py` + `core/memory_injector.py` + `core/ack_orchestrator.py` |
| `start_server.py` (82) | **MODIFY** | Add `/feedback` route wire-up; otherwise unchanged |
| `feedback.py` (59) | **KEEP** | Wire-up happens in `start_server.py` |

### 5.2 `core/`
| File | Action | Detail |
|---|---|---|
| `state.py` (181) | **MODIFY** | Keep `AgentState`; delete `CapabilityState` (vestigial); add `subagent_input`, `subagent_output` typed fields only if not already present |
| `graph.py` (289) | **MODIFY** | Replace direct node imports with registry-driven build; replace `campaign_insight_agent_node` with compiled subgraph reference; keep module-level domain/context caching |
| `config.py` (96) | **MODIFY** | Add env knobs from vision §16 that are missing (DIMENSION_QUERY_BUDGET, ENABLE_AUDIENCE_ANALYSIS, ENABLE_CONTENT_ANALYSIS, MAX_ITERATIONS_PER_STEP, TABLE_ANALYZER_THRESHOLD) |
| `auth.py` (191) | **SPLIT/MOVE** | → `auth/client_resolver.py` + `auth/secrets_loader.py` + `auth/token_provider.py` (vision §19) |
| `models.py` (66) | **KEEP** | — |
| `tracing.py` (204) | **MOVE** | → `utils/tracing.py` (vision §19) |
| `filtered_chat.py` (49) | **KEEP** | — |
| `streaming.py` | **CREATE** | Single SSE emitter; called by subgraph nodes |
| `memory_injector.py` | **CREATE** | Extracted from `agent.py` |
| `ack_orchestrator.py` | **CREATE** | Extracted from `agent.py` |

### 5.3 `supervisor/`
| File | Action | Detail |
|---|---|---|
| `supervisor_node.py` (112) | **MODIFY** | Replace direct router import with registry lookup; no behavior change |
| `intent_classifier.py` (161) | **KEEP** | Verify intent labels align with `intent.yaml` (should already) |
| `planner.py` (82) | **KEEP** | — |
| `router.py` (89) | **MODIFY** | Read from `AGENT_REGISTRY` instead of hardcoded names |
| `synthesizer.py` (174) | **KEEP** | Assembles response_items from SubagentOutput; contract unchanged |
| `domain_context.py` (46) | **MODIFY** | Load only the subset from §5.3 of vision (currently loads more than needed) |
| `prompts/supervisor_prompt.py` | **CREATE** | ~150 LOC, 7 sections per vision §5.5, injected via domain context |
| `__init__.py` | **KEEP** | — |

### 5.4 `agents/` — top level
| File | Action | Detail |
|---|---|---|
| `registry.py` (38 stub) | **REWRITE** | Real `AgentRegistration` dataclass + populated `AGENT_REGISTRY` per vision §9 |
| `base_agent.py` | **CREATE** | Abstract base with `build_graph()`, `handles()`, `input_contract()`, `output_contract()` |
| `greeting.py` (49) | **KEEP** | Registered as `GreetingAgent` in AGENT_REGISTRY |
| `clarification.py` (103) | **KEEP** | Registered as `ClarificationAgent`; still STM-only |
| `acknowledgment.py` (68) | **KEEP** | Used pre-graph by `ack_orchestrator.py` |
| `followup.py` (57) | **KEEP** | Used post-graph; confirm in §8 open questions |
| `agent_model.py` (3) | **DELETE** | Empty stub |
| `agentbricks.py` (479) | **DELETE** | Dead per vision §20 |
| `capability_registry.py` (29) | **DELETE** | Unused stub |
| `genie_agent.py` (286) | **DELETE** | Legacy path |
| `genie_analysis.py` (147) | **DELETE** | Legacy |
| `genie_client.py` (141) | **MOVE** | → `tools/genie_client.py` (referenced by `tool_handler.py` + `tools/registry.py`) |
| `genie_worker.py` (242) | **EXTRACT + DELETE** | Pull rate-limiter into `tools/genie_rate_limiter.py`, then delete the rest |
| `planner.py` (181) | **DELETE** | Legacy planner; new one in `supervisor/planner.py` |
| `supervisor.py` (208) | **DELETE** | Legacy hybrid supervisor; new one in `supervisor/` |
| `synthesizer.py` (292) | **DELETE** | Legacy; replaced by `supervisor/synthesizer.py` |

### 5.5 `agents/campaign_insight/` — subagent (the main refactor)
| File | Action | Detail |
|---|---|---|
| `agent.py` (268) | **REWRITE** (thin wrapper) | Class becomes `CampaignInsightAgent` with `build_graph() -> CompiledGraph` (returns the compiled subgraph) + `run(parent_state)` adapter for registry dispatch |
| `graph.py` | **CREATE** | StateGraph(SubagentState) builder: wires the 8 node callables (dim_classifier, dim_validator, adaptive_planner, executor, interpreter, recommender, reflector, output_builder) with conditional edges + Send-API fan-out |
| `state.py` | **CREATE** | `SubagentState` TypedDict: request_id, input, dimensions, plan, step_results[], step_index, interpretation, recommendations, reflection, output, traces |
| `contracts.py` (221) | **KEEP** | Remains the authoritative dataclass set |
| `dimension_classifier.py` (183) | **MODIFY** | Expose `dimension_classifier_node(state) -> dict`; keep class as internal helper |
| `dimension_validator.py` (174) | **MODIFY** | Expose `dimension_validator_node(state) -> dict` |
| `adaptive_planner.py` (250) | **MODIFY** | Expose `adaptive_planner_node(state) -> dict`; emit `plan_ready` SSE |
| `executor.py` (296) | **MODIFY** | Expose `executor_node(state) -> dict`; per-iteration emit `step_progress` + `data_table` SSE; support Send-API invocation for parallel steps |
| `tool_handler.py` (302) | **MODIFY** | Update import: `agents.genie_client` → `tools.genie_client`; rate limiter import → `tools.genie_rate_limiter` |
| `table_analyzer.py` (265) | **KEEP** | Called from executor node |
| `table_builder.py` (172) | **KEEP** | Called from output_builder node |
| `interpreter.py` (243) | **MODIFY** | Expose `interpreter_node(state) -> dict`; emit `analysis` SSE |
| `recommender.py` (226) | **MODIFY** | Expose `recommender_node(state) -> dict`; emit `recommendations` SSE; conditional-skip edge based on intent |
| `reflector.py` (215) | **MODIFY** | Expose `reflector_node(state) -> dict` |
| `chart_builder.py` (175) | **KEEP** | Called from output_builder node |
| `output_builder.py` (95) | **MODIFY** | Expose `output_builder_node(state) -> dict`; emit `chart` + `complete` SSE |
| `domain_knowledge.py` (478) | **KEEP** | Loader is fine |
| `__init__.py` (40) | **MODIFY** | Export `build_campaign_insight_graph()` |
| `prompts/insight_prompt.py` | **KEEP** | — |
| `prompts/interpretation_prompt.py` | **KEEP** | — |
| `prompts/reflection_prompt.py` | **KEEP** | — |
| `domain_knowledge/domain_context.yaml` (222) | **MODIFY** | Merge `domain_context_data_source_update.yaml` into data_source section |
| `domain_knowledge/domain_context_data_source_update.yaml` | **DELETE** | Consumed by merge |
| `domain_knowledge/metrics.yaml` (257) | **KEEP** | — |
| `domain_knowledge/interpretation.yaml` (266) | **KEEP** | — |
| `domain_knowledge/recommendations.yaml` (233) | **KEEP** | — |
| `domain_knowledge/constraints.yaml` (169) | **KEEP** | — |
| `domain_knowledge/intent.yaml` (189) | **MODIFY** | Add audience_composition, audience_engagement, content_quality, content_benchmark intents per vision §22 |
| `domain_knowledge/audience_knowledge.yaml` (158) | **KEEP** | Validate against vision §6.4 Layer 7 checklist |
| `domain_knowledge/content_knowledge.yaml` (253) | **KEEP** | Validate against vision §6.4 Layer 8 checklist |

### 5.6 `capabilities/` (entire directory)
| File | Action | Detail |
|---|---|---|
| `insight_agent.py` (192) | **DELETE** | Old pattern, superseded by subgraph |
| `registry.py` (61) | **DELETE** | Stub |
| `__init__.py` | **DELETE** | — |
| Directory | **DELETE** | Empty after above |

### 5.7 `memory/`
| File | Action | Detail |
|---|---|---|
| `ltm_manager.py` (345) | **KEEP** | — |
| `stm_trimmer.py` (27) | **RENAME** | → `stm_manager.py` per vision §19; expand to own the checkpointer lifecycle (+InMemorySaver fallback hook) |
| `extractors.py` (103) | **KEEP** | — |
| `context_formatter.py` (81) | **KEEP** | Called by `core/memory_injector.py` |
| `constants.py` (37) | **KEEP** | — |
| `__init__.py` (35) | **MODIFY** | Update exports after rename |

### 5.8 `tools/`
| File | Action | Detail |
|---|---|---|
| `registry.py` (297) | **MODIFY** | Update import for relocated `genie_client` |
| `genie_client.py` | **CREATE** (move) | From `agents/genie_client.py` |
| `genie_rate_limiter.py` | **CREATE** (extract) | From `agents/genie_worker.py` |

### 5.9 `parsers/`
| File | Action | Detail |
|---|---|---|
| `formatters.py` (214) | **MODIFY** | Remove code paths that serve the legacy synthesizer |
| `subagent_parser.py` (215) | **KEEP** | Still needed by `supervisor/synthesizer.py` |
| `table_truncator.py` (202) | **MOVE** | → `parsers/table_truncation.py` (vision §19 naming) |
| `filters.py` (57) | **KEEP** | — |
| `validators.py` (133) | **KEEP** | — |

### 5.10 `prompts/` (top-level, pre-split)
| File | Action | Detail |
|---|---|---|
| `supervisor_prompt.py` (1,062) | **DELETE** | Replaced by `supervisor/prompts/supervisor_prompt.py` |
| `genie_agent_prompt.py` (174) | **DELETE** | Legacy Genie path gone |
| `insight_agent_prompt.py` (60) | **DELETE** | Replaced by per-phase prompts under `agents/campaign_insight/prompts/` |
| `planner_prompt.py` (39) | **DELETE** | Legacy |
| `synthesizer_prompt.py` (38) | **MOVE** | → `supervisor/prompts/synthesizer_prompt.py` if still referenced by `supervisor/synthesizer.py`; else delete |
| `acknowledgment_prompt.py` (35) | **MOVE** | → `prompts/acknowledgment_prompt.py` (keeps), or co-locate with `core/ack_orchestrator.py` |
| `__init__.py` | **DELETE** once empty | — |

### 5.11 `ui/`
| File | Action | Detail |
|---|---|---|
| `routes.py` (266) | **MODIFY** | Add feedback endpoint wiring |
| `__init__.py` | **KEEP** | — |

### 5.12 `tests/`
| File | Action | Detail |
|---|---|---|
| `test_local.py` (38) | **EXPAND** | Add per-phase subgraph tests; add registry-dispatch tests; add contract-boundary tests (supervisor↔subagent) |
| Everything new | **CREATE** | One test file per new module (CLAUDE.md rule: every new module has tests) |

---

## 6. Node-by-Node Plan (at graph level)

### 6.1 Parent graph (`core/graph.py`)
| Node | Action | Implementation source |
|---|---|---|
| `supervisor_classify` | **KEEP** | `supervisor.supervisor_node` (minor mods for registry lookup) |
| `greeting` | **KEEP** | `agents.greeting.greeting_node` |
| `clarification` | **KEEP** | `agents.clarification.clarification_node` |
| `out_of_scope` | **KEEP** | Currently inline in `core/graph.py`; fine |
| `campaign_insight_agent` | **REPLACE** | Was: class invocation inside one node. Becomes: `graph_builder.add_node("campaign_insight_agent", build_campaign_insight_graph())` — LangGraph compiles the subgraph and invokes it |
| `supervisor_synthesize` | **KEEP** | `supervisor.synthesizer.synthesize_node` |

### 6.2 Campaign Insight subgraph (new)
| Node | Phase | LLM? | Emits SSE |
|---|---|---|---|
| `dimension_classifier` | 1A | yes | — |
| `dimension_validator` | 1B | no | — |
| `adaptive_planner` | 1C | yes | `plan_ready` |
| `executor` | 2 | yes (per-step ReAct) | `step_progress`, `data_table` |
| `interpreter` | 3a | yes | `analysis` |
| `recommender` | 3b (conditional) | yes | `recommendations` |
| `reflector` | 4 | yes | — |
| `output_builder` | 5 | no | `chart`, `complete` |

Edges:
- `adaptive_planner → executor`: conditional. If `plan.parallelizable` and independent step count > 1, use `Send` API to fan out. Otherwise sequential.
- `executor → executor | interpreter`: conditional on `step_index < len(plan)`.
- `interpreter → recommender | reflector`: skip recommender for intents ∉ {diagnostic, strategic_recommendation}.
- `reflector → output_builder`: always.

---

## 7. Sequenced Work Plan (checklist you can ratify phase-by-phase)

### Phase A — Alignment (no code)
- [ ] You approve this plan
- [ ] Resolve open questions in §8
- [ ] Tag `pre-v4.1-migration`

### Phase B — Dead code removal
- [ ] Move `agents/genie_client.py` → `tools/genie_client.py`; fix 2 imports
- [ ] Extract `GenieRateLimiter` from `genie_worker.py` → `tools/genie_rate_limiter.py`
- [ ] Delete: agentbricks.py, agent_model.py, capability_registry.py, genie_agent.py, genie_analysis.py, genie_worker.py, planner.py (agents/), supervisor.py (agents/), synthesizer.py (agents/), capabilities/*
- [ ] Delete legacy prompts: supervisor_prompt.py (top), genie_agent_prompt.py, insight_agent_prompt.py, planner_prompt.py
- [ ] `pytest` green; smoke-test chat

### Phase C — Subgraph conversion
- [ ] Create `agents/campaign_insight/state.py`
- [ ] Create `agents/campaign_insight/graph.py` with node callables + edges
- [ ] Add `_node` wrappers on each of 8 phase modules
- [ ] Shrink `agents/campaign_insight/agent.py` to thin builder/entry
- [ ] Wire subgraph as node in `core/graph.py`
- [ ] Add per-node MLflow spans
- [ ] Tests: unit per node, integration across subgraph

### Phase D — Streaming upgrade
- [ ] Create `core/streaming.py`
- [ ] Subgraph nodes emit events; remove ad-hoc emission from `agent.py`
- [ ] Verify 8-event sequence end-to-end

### Phase E — Supervisor slim + registry
- [ ] Rewrite `supervisor/prompts/supervisor_prompt.py` (~150 LOC)
- [ ] Implement `agents/base_agent.py` + real `agents/registry.py`
- [ ] `core/graph.py` builds from `AGENT_REGISTRY`
- [ ] Verify intent classification accuracy on 30-question set ≥ baseline

### Phase F — Hygiene
- [ ] Split `agent.py` per §5.1
- [ ] Split `core/auth.py` into `auth/` package
- [ ] Move `core/tracing.py` → `utils/tracing.py`
- [ ] Rename `parsers/table_truncator.py` → `table_truncation.py`
- [ ] Merge `domain_context_data_source_update.yaml` into `domain_context.yaml`
- [ ] Add new intents to `intent.yaml`
- [ ] Wire `/feedback` route
- [ ] Final lint + 200-line rule audit

---

## 8. Open Questions (need your call before Phase B)

1. **Acknowledgment + followup**: Not in v4 topology. Keep (as pre/post-graph helpers) or delete? Current plan: **keep**, treat as app-layer wrappers outside the graph.
2. **`capabilities/` delete**: Confirm no external caller imports it. Audit shows only internal references; safe to delete.
3. **Parallel ReAct (Send API)**: v4 permits it but baseline is sequential. Default in Phase C: **sequential**, toggle via flag, enable in a later phase after trace data.
4. **LLM model for subagent**: Vision says `databricks-gpt-5-2`. CLAUDE.md mentions migration to `databricks-claude-sonnet-4-5` for reasoning. Which is canonical for v4.1?
5. **Registry-driven graph**: Does adding an agent require zero `core/graph.py` edits (pure config), or is one-line registration acceptable? Plan assumes pure-config.
6. **`agent.py` location**: Vision §19 keeps it at `agent_server/agent.py`. Current 1,494 LOC includes streaming + memory injection that belong in `core/`. Confirm split targets.
7. **STM fallback UX**: On Lakebase failure, the vision requires a user-visible warning. No such warning exists today. Add in Phase D?
8. **Legacy prompt `synthesizer_prompt.py`**: Move under `supervisor/prompts/` or inline into `supervisor/synthesizer.py`?

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Subgraph compilation breaks existing checkpointer format | Medium | Use separate thread_id namespace for subgraph; test STM resume before cutover |
| Streaming event ordering regressions | Medium | Golden-file tests on SSE sequence |
| Supervisor prompt shrink degrades intent accuracy | Medium | Keep 30-question eval as gate; do not merge Phase E until parity |
| Send-API fan-out for parallel ReAct exceeds Genie rate limits | Low | Rate limiter + staggered workers; keep sequential default |
| `AgentState` field churn breaks in-flight conversations | Low | Migration done in single deploy window; no mid-flight schema change |
| Losing legacy genie_worker rate-limiter before extraction | Medium | Extract BEFORE delete in Phase B |

---

## 10. Non-Goals (explicitly out of scope for v4.1)

- Genie managed MCP migration (CLAUDE.md mentions it; keep REST for now).
- Custom 2D output format (vision §22 — waiting on your spec).
- MLflow Evaluation harness (vision §22).
- Prompt caching implementation (vision §18 — strategy only).
- Additional subagents beyond Campaign Insight.
- Unity Catalog RLS changes (production-grade today).

---

## 11. Deliverables at end of migration

- Compiled parent graph with 6 nodes, one of which is a compiled subgraph.
- 8 first-class subgraph nodes with individual traces + SSE events.
- All dead code removed (~3,000 LOC).
- Supervisor prompt ≤150 LOC.
- `agent.py` ≤200 LOC.
- Registry-driven topology.
- 30-question eval ≥ baseline.
- All new modules have tests.
- `pre-v4.1-migration` tag preserved for rollback.

---

**End of plan. Awaiting your approval before any code changes.**
