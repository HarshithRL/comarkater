# 8. Risks and Issues

Ranked by severity. All items cite a file/pattern.

---

## 8.1 HIGH

### R1 — `agent.py` is 1,494 LOC
- File: `agent_server/agent.py`.
- Spec cap: 200 LOC per file (`CLAUDE.md`).
- Impact: Monolith mixes request parsing, greeting path, LTM/STM glue, LLM build, ack generation, graph invoke, SSE event mapping, LTM write, trace finalization. Single point of failure for virtually every code path.
- Action: Split into `entry.py`, `lifecycle/ltm.py`, `lifecycle/stm.py`, `lifecycle/ack.py`, `streaming/sse_mapper.py`, `lifecycle/post_response.py`.

### R2 — Missing architecture components with no fallback
- `CLAUDE.md` promises stats engine (11 metrics in Python), anomaly scanner, analytical cache, filter persistence, error recovery chain. None exist.
- Impact: Product commitments (e.g., guaranteed numeric correctness, anomaly surfacing) are unmet; agent relies on Genie-produced metrics for correctness.
- Action: Either implement per spec or update `CLAUDE.md` to match reality.

### R3 — Spec vs. implementation LLM mismatch
- Spec: Haiku for supervisor, Sonnet for reasoning.
- Reality: `databricks-gpt-5-2` for both. Quality/cost assumptions baked into the architecture doc are invalid.
- Action: Decide canonical model routing; either wire Haiku/Sonnet via `ChatDatabricks` or update the spec.

---

## 8.2 MEDIUM

### R4 — Async-in-sync boundary
- `agent.py:~1457–1468`, `core/graph.py:~113`.
- Pattern: LangGraph nodes are sync; `CampaignInsightAgent.run` is async. Bridged with `loop.run_in_executor(None, run_sync)` using a dedicated thread to preserve MLflow `ContextVar`s.
- Risk: Any future code that introduces nested event loops or MLflow spans off this thread will silently lose trace context.
- Action: Add unit test that asserts every insight-agent step produces a child span under the predict trace.

### R5 — Unbounded STM growth
- `memory/stm_trimmer.py:get_trim_removals` enforces bounds, but Lakebase checkpoint rows remain until trimmed.
- Risk: Long-lived `thread_id`s bloat Lakebase; no explicit TTL visible.
- Action: Document trim policy; add periodic compaction job.

### R6 — Secret handling
- `core/auth.py:SecretsLoader` caches secrets in process memory with no TTL or invalidation on auth failure.
- Risk: Rotated secrets won't take effect without a process restart; cached secrets persist in memory.
- Action: Add TTL + cache bust on 401/403 from Genie.

### R7 — No rate limiting on `/chat` or `/invocations`
- `ui/routes.py`, `start_server.py`.
- Risk: Burst traffic exhausts Genie quota, SP tokens, or LLM budget. Databricks App proxy limits are UNKNOWN at this layer.
- Action: Add per-client quota middleware.

### R8 — Supervisor `Command(goto=...)` uses LLM-chosen target
- `agents/supervisor.py:~83`.
- Risk: If LLM output is malformed or injected, goto target could miss the whitelist. Today only 4 valid targets are wired, so blast radius is limited.
- Action: Post-process supervisor output through a strict whitelist before building the `Command`.

### R9 — Genie polling error handling
- `agents/campaign_insight/executor.py`.
- Pattern: Up to 3 iters, 120s total. On `FAILED` the step returns an error but the flow continues and emits "please rephrase".
- Risk: Users cannot distinguish Genie errors (transient) from genuinely unsupported queries; no escalation path.
- Action: Differentiate error classes; add bounded retry w/ backoff.

### R10 — MLflow `update_current_trace` footgun
- `agent.py:~313–337` with warning comment "only safe with tags={}".
- Risk: Any future call that passes `params=` or `metadata=` truncates the trace.
- Action: Wrap in a helper that enforces tags-only.

### R11 — Bundle config duplication
- Top-level `databricks.yml` vs. `bundles/netcore-insight-agent/databricks.yml`.
- Risk: Drift between environments; unclear which is authoritative.

### R12 — `COMARKETER_ENV=development` in production deployment
- `databricks.yml`.
- Risk: Code paths gated on env may misbehave. Worth auditing all `settings.ENV` branches.

---

## 8.3 LOW

- **R13** — No SQL injection surface in agent (Genie translates NL → SQL), but supervisor/insight prompts include user-provided strings; prompt-injection risk exists but is bounded by strict JSON schema on output.
- **R14** — LTM write is fire-and-forget; failures logged, not surfaced. Acceptable today.
- **R15** — Test coverage is 2 unit tests + 17 notebooks. No pytest exercise of the compiled graph end-to-end.
- **R16** — `docs/` contains potentially stale scaffolding — risk of onboarding developers off-spec. `CLAUDE.md` should be pointed to explicitly.

---

## 8.4 Security posture summary

| Control | State |
|---------|-------|
| RLS | Enforced by Unity Catalog; agent never injects `WHERE cid` |
| Per-client OBO | ✅ per-request SP token |
| Secret source | ✅ Databricks Secrets (`agent-sp-credentials`) |
| Input validation | Length cap only (`MAX_QUERY_LENGTH=5000`) |
| Rate limit | ❌ |
| Trace redaction | UNKNOWN — traces carry prompt content |
| PII handling | UNKNOWN — no explicit redaction in ltm extractors |
