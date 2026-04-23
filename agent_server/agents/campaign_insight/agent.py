"""Campaign Insight Agent — 5-phase orchestrator.

Phases: adaptive plan -> ReAct execute -> interpret + recommend -> reflect ->
build output. Stateless components are constructed once; LLM and tool handler
are rebuilt per-request because they depend on the SP token.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
from langchain_openai import ChatOpenAI

from agents.campaign_insight.adaptive_planner import AdaptivePlanner
from agents.campaign_insight.chart_builder import ChartBuilder
from agents.campaign_insight.contracts import (
    AgentStatus,
    DimensionClassification,
    Interpretation,
    StepStatus,
    SubagentOutput,
)
from agents.campaign_insight.dimension_classifier import DimensionClassifier
from agents.campaign_insight.dimension_validator import DimensionValidator
from agents.campaign_insight.domain_knowledge import InsightAgentDomainKnowledge
from agents.campaign_insight.executor import ReActExecutor
from agents.campaign_insight.genie_validator import GenieResultValidator
from agents.campaign_insight.interpreter import Interpreter
from agents.campaign_insight.output_builder import OutputBuilder
from agents.campaign_insight.query_router import (
    QueryRouter,
    RoutingDecision,
    RoutingStrategy,
)
from agents.campaign_insight.recommender import Recommender
from agents.campaign_insight.reflector import Reflector
from agents.campaign_insight.table_analyzer import TableAnalyzer
from agents.campaign_insight.table_builder import TableBuilder
from agents.campaign_insight.tool_handler import ToolHandler
from core.config import settings
from core.tracing import flow_debug, flow_log

logger = logging.getLogger(__name__)


def _safe_stream(writer: Any, event: dict) -> None:
    """Best-effort stream emit - never raise."""
    if writer is None:
        return
    try:
        writer(event)
    except Exception:  # pragma: no cover
        pass


class CampaignInsightAgent:
    """Orchestrates the 5-phase Campaign Insight analysis."""

    def __init__(self, config: dict | None = None) -> None:
        """Build the agent and its stateless collaborators.

        Args:
            config: Optional dict of overrides. Recognized keys:
                ``genie_space_id``, ``domain_knowledge_path``, ``model_name``,
                ``feature_flags``, ``max_iterations_per_step``,
                ``total_timeout_seconds``, ``databricks_host``.
        """
        cfg = config or {}
        self.genie_space_id: str = cfg.get("genie_space_id", settings.GENIE_SPACE_ID)
        default_kb = Path(__file__).parent / "domain_knowledge"
        self.domain_knowledge_path: Path = Path(cfg.get("domain_knowledge_path", default_kb))
        self.model_name: str = cfg.get("model_name", settings.LLM_ENDPOINT_NAME)
        self.feature_flags: dict = cfg.get("feature_flags", {}) or {}
        self.max_iterations_per_step: int = int(cfg.get("max_iterations_per_step", 2))
        self.total_timeout_seconds: int = int(cfg.get("total_timeout_seconds", 180))
        self.step_timeout_seconds: int = int(cfg.get("step_timeout_seconds", 60))
        self.databricks_host: str = cfg.get("databricks_host", settings.DATABRICKS_HOST)

        self.domain_knowledge = InsightAgentDomainKnowledge(self.domain_knowledge_path)
        self.table_analyzer = TableAnalyzer()
        self.table_builder = TableBuilder()
        self.dimension_validator = DimensionValidator()
        self.output_builder = OutputBuilder()

    async def run(self, state: dict, config: dict) -> dict:
        """Execute all 5 phases and return a state update.

        Args:
            state: Graph state carrying ``rewritten_question`` /
                ``original_question``, optional ``subagent_input``,
                ``intent``, ``plan``, and context.
            config: Runnable config - must include ``configurable.sp_token``.

        Returns:
            A dict with a single ``subagent_output`` key holding the
            :class:`SubagentOutput`.
        """
        try:
            from langgraph.config import get_stream_writer
            writer = get_stream_writer()
        except Exception:
            writer = None

        t0 = time.monotonic()
        request_id = state.get("request_id", "")
        query = state.get("rewritten_question") or state.get("original_question", "") or ""

        sub_in = state.get("subagent_input") or {}
        intent = sub_in.get("intent") if sub_in else state.get("intent") or {}
        if not isinstance(intent, dict):
            intent = {"intent_type": str(intent)} if intent else {}
        supervisor_plan = sub_in.get("plan") if sub_in else state.get("plan") or []
        if not isinstance(supervisor_plan, list):
            supervisor_plan = []
        feature_flags = (sub_in.get("config") or {}).get("feature_flags") if sub_in else None
        feature_flags = feature_flags or self.feature_flags or {}

        flow_log(
            request_id,
            "insight.start",
            intent=intent.get("intent_type", ""),
            query=query[:160],
        )

        # Declared outside the try so the error path can cancel it if an
        # exception in Phase 3/4 skips the await at Phase 5.
        chart_task: asyncio.Task | None = None

        try:
            sp_token = config["configurable"]["sp_token"]
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=sp_token,
                base_url=settings.AI_GATEWAY_URL,
                temperature=0.0,
            )
            tool_handler = ToolHandler(self.genie_space_id, sp_token, self.databricks_host)

            routing_enabled = bool(feature_flags.get("balanced_routing", False))
            validator_enabled = routing_enabled and bool(
                feature_flags.get("balanced_routing.validator_enabled", True)
            )
            llm_tiebreaker = routing_enabled and not bool(
                feature_flags.get("balanced_routing.rules_only", True)
            )
            fallback_enabled = routing_enabled and bool(
                feature_flags.get("balanced_routing.fallback_enabled", True)
            )

            validator = GenieResultValidator() if validator_enabled else None
            query_router = (
                QueryRouter(
                    llm,
                    self.domain_knowledge,
                    llm_tiebreaker_enabled=llm_tiebreaker,
                )
                if routing_enabled
                else None
            )

            dim_classifier = DimensionClassifier(llm, self.domain_knowledge)
            adaptive_planner = AdaptivePlanner(llm, self.domain_knowledge)
            executor = ReActExecutor(
                llm,
                tool_handler,
                self.table_analyzer,
                self.table_builder,
                self.domain_knowledge,
                self.max_iterations_per_step,
                self.total_timeout_seconds,
                self.step_timeout_seconds,
                validator=validator,
            )
            interpreter = Interpreter(llm, self.domain_knowledge)
            recommender = Recommender(llm, self.domain_knowledge)
            reflector = Reflector(llm, self.domain_knowledge)
            chart_llm = ChatOpenAI(
                model=settings.CHART_LLM_ENDPOINT_NAME,
                api_key=sp_token,
                base_url=settings.AI_GATEWAY_URL,
                temperature=0.0,
            )
            chart_builder = ChartBuilder(chart_llm)

            # Phase 1 - Adaptive plan.
            with mlflow.start_span(name="dim_classify") as _s:
                _s.set_inputs({"query": query[:500], "intent_type": intent.get("intent_type", "")})
                dim_config = dim_classifier.classify(query, intent)
                _s.set_outputs({
                    "primary_analysis": dim_config.primary_analysis,
                    "channel": dim_config.channel or "all",
                    "campaign_role": dim_config.campaign.role.value,
                    "audience_role": dim_config.audience.role.value,
                    "content_role": dim_config.content.role.value,
                    "total_budget": dim_config.total_budget,
                })
            flow_log(
                request_id,
                "dim_classify",
                primary=dim_config.primary_analysis,
                channel=dim_config.channel or "all",
                campaign=f"{dim_config.campaign.role.value}/{dim_config.campaign.budget}",
                audience=f"{dim_config.audience.role.value}/{dim_config.audience.budget}",
                content=f"{dim_config.content.role.value}/{dim_config.content.budget}",
            )

            dim_config = self.dimension_validator.validate(dim_config, query, feature_flags)
            flow_log(
                request_id,
                "dim_validate",
                total_budget=dim_config.total_budget,
                active_dims=",".join(dim_config.active_dimensions),
            )

            routing: RoutingDecision | None = None
            if query_router is not None:
                try:
                    with mlflow.start_span(name="query_router_classify") as _s:
                        _s.set_inputs({
                            "query": query[:500],
                            "intent_type": intent.get("intent_type", ""),
                            "primary_analysis": dim_config.primary_analysis,
                        })
                        routing = query_router.classify(query, intent, dim_config)
                        _s.set_outputs({
                            "strategy": routing.strategy.value,
                            "source": routing.source,
                            "confidence": routing.confidence,
                            "reason": routing.reason[:500],
                        })
                    flow_log(
                        request_id,
                        "routing",
                        strategy=routing.strategy.value,
                        source=routing.source,
                        confidence=round(routing.confidence, 2),
                        reason=routing.reason[:80],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("QueryRouter failed, falling through: %s", exc)
                    flow_log(
                        request_id,
                        "routing_exception",
                        error=str(exc)[:160],
                    )
                    routing = None

            with mlflow.start_span(name="adaptive_plan") as _s:
                _s.set_inputs({
                    "query": query[:500],
                    "intent_type": intent.get("intent_type", ""),
                    "primary_analysis": dim_config.primary_analysis,
                    "total_budget": dim_config.total_budget,
                    "supervisor_plan": list(supervisor_plan)[:10],
                    "routing_strategy": routing.strategy.value if routing else None,
                })
                plan = adaptive_planner.plan(
                    query, intent, dim_config, supervisor_plan, routing=routing
                )
                _s.set_outputs({
                    "plan_steps": len(plan.steps),
                    "total_budget": plan.total_budget,
                    "dims": sorted({s.dimension for s in plan.steps}),
                    "steps": [
                        {"step_id": s.step_id, "dim": s.dimension, "query": s.query[:250], "purpose": s.purpose[:250]}
                        for s in plan.steps
                    ],
                })
            flow_log(
                request_id,
                "plan",
                steps=len(plan.steps),
                budget=plan.total_budget,
                dims=",".join(sorted({s.dimension for s in plan.steps})),
            )
            flow_debug(
                request_id,
                "plan",
                queries_json=json.dumps([
                    {"step_id": s.step_id, "dim": s.dimension, "query": s.query, "purpose": s.purpose}
                    for s in plan.steps
                ]),
            )
            # plan_ready is emitted from inside AdaptivePlanner.plan() (richer schema).

            # Phase 2 - ReAct execute.
            with mlflow.start_span(name="execute_plan") as _s:
                _s.set_inputs({
                    "plan_steps": len(plan.steps),
                    "channel": dim_config.channel or "all",
                    "routing_strategy": routing.strategy.value if routing else None,
                })
                step_results = await executor.execute_plan(
                    plan,
                    stream_callback=lambda ev: _safe_stream(writer, ev),
                    channel=dim_config.channel,
                    request_id=request_id,
                    routing=routing,
                )
                _status_counts = {s.value: 0 for s in StepStatus}
                for _sr in step_results.values():
                    _status_counts[_sr.status.value] = _status_counts.get(_sr.status.value, 0) + 1
                _s.set_outputs({
                    "steps_ran": len(step_results),
                    "success": _status_counts.get("success", 0),
                    "partial": _status_counts.get("partial", 0),
                    "error": _status_counts.get("error", 0),
                    "timeout": _status_counts.get("timeout", 0),
                    "rows_per_step": [
                        (sr.table_summary.row_count if sr.table_summary else 0)
                        for sr in step_results.values()
                    ],
                })
            flow_log(
                request_id,
                "execute",
                steps_ran=len(step_results),
                success=_status_counts.get("success", 0),
                partial=_status_counts.get("partial", 0),
                error=_status_counts.get("error", 0),
                timeout=_status_counts.get("timeout", 0),
            )

            # Phase 2b - Fallback to AGENT_DECOMPOSE when a single-step path
            # could not satisfy the output-shape contract.
            _should_fallback = (
                fallback_enabled
                and routing is not None
                and routing.strategy in {
                    RoutingStrategy.GENIE_DIRECT,
                    RoutingStrategy.HYBRID,
                }
                and bool(step_results)
                and self._all_contract_violations(step_results)
            )
            if _should_fallback and routing is not None:
                violations_summary = "; ".join(
                    v
                    for sr in step_results.values()
                    for v in sr.validation_violations[:2]
                )[:200]
                flow_log(
                    request_id,
                    "routing_fallback",
                    from_strategy=routing.strategy.value,
                    to_strategy=RoutingStrategy.AGENT_DECOMPOSE.value,
                    reason=violations_summary or "single-step validator failed",
                )
                fallback_routing = RoutingDecision(
                    strategy=RoutingStrategy.AGENT_DECOMPOSE,
                    reason="fallback after contract violations",
                    confidence=routing.confidence,
                    source="fallback",
                )
                plan = adaptive_planner.plan(
                    query,
                    intent,
                    dim_config,
                    supervisor_plan,
                    routing=fallback_routing,
                )
                step_results = await executor.execute_plan(
                    plan,
                    stream_callback=lambda ev: _safe_stream(writer, ev),
                    channel=dim_config.channel,
                    request_id=request_id,
                    routing=fallback_routing,
                )
                _status_counts = {s.value: 0 for s in StepStatus}
                for _sr in step_results.values():
                    _status_counts[_sr.status.value] = _status_counts.get(_sr.status.value, 0) + 1
                flow_log(
                    request_id,
                    "execute_fallback",
                    steps_ran=len(step_results),
                    success=_status_counts.get("success", 0),
                    partial=_status_counts.get("partial", 0),
                    error=_status_counts.get("error", 0),
                    timeout=_status_counts.get("timeout", 0),
                )

            # Short-circuit when every step errored. Running interpret /
            # recommend / reflect on empty data lets the LLM hallucinate
            # "rephrase your query" advice as if it were a real insight.
            # Skip Phases 3-5 and surface a clean message instead. TIMEOUT
            # and PARTIAL still flow through the normal path because they
            # may carry usable rows.
            if step_results and all(
                sr.status == StepStatus.ERROR for sr in step_results.values()
            ):
                error_messages = [
                    (sr.error_message or "")[:120]
                    for sr in step_results.values()
                ]
                flow_log(
                    request_id,
                    "short_circuit",
                    reason="all_steps_errored",
                    steps=len(step_results),
                    first_error=error_messages[0] if error_messages else "",
                )
                caveats = self._collect_caveats(step_results, dim_config)
                summary = (
                    "I couldn't retrieve data for that question. Try "
                    "simplifying it — pick one channel, one metric, and a "
                    "clear time window, and avoid combined filters in a "
                    "single ask."
                )
                metadata = {
                    "model": self.model_name,
                    "total_duration_ms": int((time.monotonic() - t0) * 1000),
                    "request_id": request_id,
                    "streamed_types": [],
                    "short_circuit": "all_errored",
                    "step_errors": error_messages,
                }
                return {
                    "subagent_output": SubagentOutput(
                        request_id=request_id,
                        status=AgentStatus.ERROR,
                        execution_summary={
                            "steps_run": len(step_results),
                            "errored": len(step_results),
                        },
                        interpretation=Interpretation(
                            summary=summary, severity="warning"
                        ),
                        recommendations=[],
                        caveats=caveats,
                        metadata=metadata,
                    )
                }

            # Kick off chart building in parallel with Phase 3/4. It only
            # needs the (now-final) step_results + intent — neither is
            # mutated by interpret/recommend/reflect — so running it
            # concurrently is safe. The chart_ready event still fires via
            # the stream writer from inside the builder. Any exception here
            # is isolated: we fall back to chart=None without failing the
            # rest of the pipeline.
            chart_task = asyncio.create_task(
                asyncio.to_thread(
                    chart_builder.build_chart,
                    intent.get("intent_type", ""),
                    step_results,
                )
            )

            # Phase 3 - Interpret + recommend.
            interpretation = interpreter.interpret(step_results, intent, dim_config)
            flow_log(
                request_id,
                "interpret",
                insights=len(interpretation.insights),
                patterns=len(interpretation.patterns),
                severity=interpretation.severity or "info",
            )
            flow_debug(
                request_id,
                "interpret",
                summary=interpretation.summary,
                insights_json=json.dumps(interpretation.insights),
                patterns_json=json.dumps([asdict(p) for p in interpretation.patterns]),
            )

            recommendations = recommender.recommend(interpretation, step_results, intent)
            flow_log(request_id, "recommend", recs=len(recommendations))
            flow_debug(
                request_id,
                "recommend",
                recs_json=json.dumps([asdict(r) for r in recommendations]),
            )

            # Phase 4 - Reflect.
            with mlflow.start_span(name="reflect") as _s:
                _s.set_inputs({
                    "interpretation_summary": (interpretation.summary or "")[:1000],
                    "insights_count": len(interpretation.insights),
                    "patterns_count": len(interpretation.patterns),
                    "recs_count": len(recommendations),
                    "severity": interpretation.severity or "info",
                })
                verified = reflector.verify(interpretation, recommendations, step_results)
                final_interp = verified.interpretation or interpretation
                final_recs = verified.recommendations or recommendations
                _s.set_outputs({
                    "passed": verified.passed,
                    "issues_found": list(verified.issues_found)[:10],
                    "interp_rewritten": bool(verified.interpretation),
                    "recs_rewritten": bool(verified.recommendations),
                    "final_summary": (final_interp.summary or "")[:1000],
                    "final_recs_count": len(final_recs),
                })
            flow_log(
                request_id,
                "reflect",
                passed=verified.passed,
                issues=len(verified.issues_found),
                interp_rewritten=bool(verified.interpretation),
                recs_rewritten=bool(verified.recommendations),
            )
            flow_debug(
                request_id,
                "reflect",
                issues_list=json.dumps(verified.issues_found),
                fixes_applied=json.dumps(verified.fixes_applied),
            )

            # Phase 5 - Build output.
            # Chart was kicked off in parallel before Phase 3; just await it
            # here. A failure inside the task is isolated so the rest of the
            # output still builds.
            with mlflow.start_span(name="chart_await") as _s:
                try:
                    chart = await chart_task
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Chart builder task failed: %s", exc)
                    chart = None
                _chart_type = None
                if isinstance(chart, dict):
                    _chart_type = chart.get("chart_type") or (chart.get("chart") or {}).get("type")
                _s.set_outputs({"chart_type": _chart_type or "none", "skipped": chart is None})
            flow_log(
                request_id,
                "chart",
                type=_chart_type or "none",
                skipped=chart is None,
            )
            if chart is not None:
                flow_debug(request_id, "chart", chart_json=json.dumps(chart))

            caveats = self._collect_caveats(step_results, dim_config)
            streamed_types: list[str] = []
            if writer is not None:
                if any(sr.display_table for sr in step_results.values()):
                    streamed_types.append("table")
                if chart is not None:
                    streamed_types.append("chart")
            metadata = {
                "model": self.model_name,
                "total_duration_ms": int((time.monotonic() - t0) * 1000),
                "reflection_issues_found": len(getattr(verified, "issues_found", []) or []),
                "request_id": request_id,
                "streamed_types": streamed_types,
            }
            output = self.output_builder.build(
                request_id,
                step_results,
                final_interp,
                final_recs,
                chart,
                caveats,
                metadata,
            )
            flow_log(
                request_id,
                "output",
                status=output.status.value,
                tables=len(output.tables_for_display),
                recs=len(output.recommendations),
                caveats=len(output.caveats),
                has_chart=output.chart_spec is not None,
                total_ms=metadata["total_duration_ms"],
            )
            # Hoist flat fields from step_results + interpretation so the
            # top-level graph state (and the MLflow root-trace tag writer in
            # agent.py) can see them. Uses the last non-empty value across
            # steps so multi-step plans surface the most recent sql/trace.
            last_sql = ""
            last_trace_id = ""
            for sid in sorted(step_results):
                sr = step_results[sid]
                if sr.sql:
                    last_sql = sr.sql
                if sr.genie_trace_id:
                    last_trace_id = sr.genie_trace_id
            return {
                "subagent_output": output,
                "genie_sql": last_sql,
                "genie_trace_id": last_trace_id,
                "genie_summary": final_interp.summary or "",
            }

        except Exception as exc:
            logger.exception("CampaignInsightAgent.run failed: %s", exc)
            # Cancel the parallel chart task so it can't emit a chart_ready
            # event into the stream after the error output has been sent.
            if chart_task is not None and not chart_task.done():
                chart_task.cancel()
            total_ms = int((time.monotonic() - t0) * 1000)
            flow_log(
                request_id,
                "output",
                status="error",
                error=str(exc)[:200],
                total_ms=total_ms,
            )
            output = SubagentOutput(
                status=AgentStatus.ERROR,
                interpretation=Interpretation(summary="The analysis failed."),
                caveats=["Internal error"],
                metadata={
                    "model": self.model_name,
                    "total_duration_ms": total_ms,
                    "error": str(exc),
                },
            )
            return {"subagent_output": output}

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _all_contract_violations(step_results: dict) -> bool:
        """Return True if every step finished with a validator violation.

        Used by the single-step fallback path: if every step in a
        ``GENIE_DIRECT`` or ``HYBRID`` plan came back ``PARTIAL`` with
        ``validation_violations`` populated, we re-plan under
        ``AGENT_DECOMPOSE``.
        """
        if not step_results:
            return False
        for sr in step_results.values():
            if sr.status == StepStatus.SUCCESS:
                return False
            if not getattr(sr, "validation_violations", None):
                return False
        return True

    def _collect_caveats(
        self,
        step_results: Any,
        dim_config: DimensionClassification,
    ) -> list[str]:
        """Gather human-readable caveats from step outcomes and dimension config.

        Args:
            step_results: Dict mapping step_id -> :class:`StepResult`.
            dim_config: Validated dimension classification.

        Returns:
            Deduplicated list of caveat strings.
        """
        _ = dim_config  # reserved for future dimension-aware caveats
        caveats: list[str] = []
        if isinstance(step_results, dict):
            results = list(step_results.values())
        else:
            results = list(step_results or [])

        usable_step = False
        for sr in results:
            status = getattr(sr, "status", None)
            step_id = getattr(sr, "step_id", "?")
            if status == StepStatus.ERROR:
                caveats.append(f"Step {step_id} failed; results may be incomplete.")
            elif status == StepStatus.TIMEOUT:
                caveats.append(f"Step {step_id} timed out; results may be incomplete.")
            elif status == StepStatus.PARTIAL:
                caveats.append(f"Step {step_id} returned partial data.")
                if getattr(sr, "display_table", None) is not None:
                    usable_step = True
            elif status == StepStatus.SUCCESS:
                usable_step = True

        if results and not usable_step:
            caveats.insert(
                0,
                "We could not retrieve data for this question after retries. "
                "Please try rephrasing, narrowing the time window, or picking a "
                "specific channel or metric.",
            )

        channels_seen: set[str] = set()
        for sr in results:
            display = getattr(sr, "display_table", None)
            if display is None:
                continue
            columns = [str(c) for c in getattr(display, "columns", []) or []]
            rows = list(getattr(display, "rows", []) or [])
            lower_cols = [c.lower() for c in columns]
            if "channel" in lower_cols:
                idx = lower_cols.index("channel")
                for row in rows:
                    if idx < len(row) and isinstance(row[idx], str):
                        channels_seen.add(row[idx])

        if "SMS" in channels_seen or "WhatsApp" in channels_seen:
            caveats.append(
                "SMS/WhatsApp do not track opens; open-rate metrics are not applicable for those rows."
            )
        if "BPN" in channels_seen:
            caveats.append(
                "BPN typically converts around 4%; benchmark comparisons should use channel-specific baselines."
            )

        seen: set[str] = set()
        deduped: list[str] = []
        for c in caveats:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped
