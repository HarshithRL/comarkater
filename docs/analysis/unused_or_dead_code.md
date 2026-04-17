# 6. Unused / Dead / Redundant Code

All items below were checked for inbound imports / graph wiring.

---

## 6.1 Legacy agent nodes (importable, never added to `core/graph.py`)

| File | Notes |
|------|-------|
| `agents/planner.py` | No inbound `from agents.planner import` outside stale references. |
| `agents/genie_agent.py` | Referenced only from legacy `capability_registry`. |
| `agents/genie_analysis.py` | Not added to compiled graph. |
| `agents/genie_worker.py` | Parallel fan-out (Send API) — unreachable. |
| `agents/synthesizer.py` | Superseded by `supervisor/synthesizer.py`. |
| `agents/registry.py` | Legacy capability registry. |
| `capabilities/insight_agent.py` | `insight_agent_node` — replaced by `CampaignInsightAgent`. |
| `capabilities/capability_registry.py` | Legacy. |

## 6.2 Legacy prompts

| File | Notes |
|------|-------|
| `prompts/planner_prompt.py` | No live consumer. |
| `prompts/genie_agent_prompt.py` | No live consumer. |
| `prompts/synthesizer_prompt.py` | No live consumer (current synthesizer uses `supervisor_prompt.py`). |

## 6.3 Supervisor stubs

| File | Notes |
|------|-------|
| `supervisor/intent_classifier.py` | Duplicate greeting regex; not imported anywhere live. |
| `supervisor/router.py` | `SupervisorRouter` class — unreferenced. |
| `supervisor/planner.py` | `SupervisorPlanner` class — unreferenced. |
| `supervisor/domain_context.py` | UNKNOWN — existence confirmed; direct inbound imports not verified. Flag for audit. |

`supervisor/synthesizer.py` IS active (confirmed).

## 6.4 Commented-out / superseded branches in `agent.py`

- `agent.py` ~lines 500–700: legacy graph-invoke branch for `genie_data → genie_analysis → planner → genie_worker → synthesizer` path.
- `agent.py` ~lines 1100–1200: legacy streaming event mapping for those old nodes.

These branches remain as code paths guarded by intent/state checks that never fire under the current compiled graph.

## 6.5 Stubbed / missing features referenced in spec (see `gaps_unknowns.md`)

- Stats engine (11 derived metrics in Python).
- Anomaly scanner node.
- Analytical cache layer (Delta, daily refresh).
- Filter persistence across requests.
- Error recovery chain beyond `genie_retry_count`.

## 6.6 Duplicate bundle configuration

- Top-level `databricks.yml` and `bundles/netcore-insight-agent/databricks.yml` both exist. UNKNOWN which is the authoritative deployment path; `CLAUDE.md` quick-commands target the top-level one. Recommend consolidating.

## 6.7 Actions

1. Delete or quarantine legacy agent/prompt files (`planner`, `genie_agent`, `genie_analysis`, `genie_worker`, `synthesizer`, corresponding prompts, `capabilities/`).
2. Delete unreferenced `supervisor/intent_classifier.py`, `supervisor/router.py`, `supervisor/planner.py`.
3. Resolve bundle-config duplication.
4. Remove commented-out legacy branches from `agent.py` as part of splitting it (see `risks_and_issues.md`).
