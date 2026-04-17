"""Campaign Insight Agent — 5-phase orchestrator.

Phases: adaptive plan -> ReAct execute -> interpret + recommend -> reflect ->
build output. Stateless components are constructed once; LLM and tool handler
are rebuilt per-request because they depend on the SP token.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

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
from agents.campaign_insight.interpreter import Interpreter
from agents.campaign_insight.output_builder import OutputBuilder
from agents.campaign_insight.recommender import Recommender
from agents.campaign_insight.reflector import Reflector
from agents.campaign_insight.table_analyzer import TableAnalyzer
from agents.campaign_insight.table_builder import TableBuilder
from agents.campaign_insight.tool_handler import ToolHandler
from core.config import settings

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
        self.max_iterations_per_step: int = int(cfg.get("max_iterations_per_step", 3))
        self.total_timeout_seconds: int = int(cfg.get("total_timeout_seconds", 120))
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

        try:
            sp_token = config["configurable"]["sp_token"]
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=sp_token,
                base_url=settings.AI_GATEWAY_URL,
                temperature=0.0,
            )
            tool_handler = ToolHandler(self.genie_space_id, sp_token, self.databricks_host)

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
            )
            interpreter = Interpreter(llm, self.domain_knowledge)
            recommender = Recommender(llm, self.domain_knowledge)
            reflector = Reflector(llm, self.domain_knowledge)
            chart_builder = ChartBuilder(llm)

            # Phase 1 - Adaptive plan.
            dim_config = dim_classifier.classify(query, intent)
            dim_config = self.dimension_validator.validate(dim_config, query, feature_flags)
            plan = adaptive_planner.plan(query, intent, dim_config, supervisor_plan)
            _safe_stream(writer, {
                "event_type": "plan_ready",
                "steps": [getattr(s, "purpose", "") for s in getattr(plan, "steps", [])],
            })

            # Phase 2 - ReAct execute.
            step_results = await executor.execute_plan(
                plan,
                stream_callback=lambda ev: _safe_stream(writer, ev),
                channel=dim_config.channel,
            )

            # Phase 3 - Interpret + recommend.
            interpretation = interpreter.interpret(step_results, intent, dim_config)
            recommendations = recommender.recommend(interpretation, step_results, intent)

            # Phase 4 - Reflect.
            verified = reflector.verify(interpretation, recommendations, step_results)
            final_interp = verified.interpretation or interpretation
            final_recs = verified.recommendations or recommendations

            # Phase 5 - Build output.
            chart = chart_builder.build_chart(intent.get("intent_type", ""), step_results)
            caveats = self._collect_caveats(step_results, dim_config)
            metadata = {
                "model": self.model_name,
                "total_duration_ms": int((time.monotonic() - t0) * 1000),
                "reflection_issues_found": len(getattr(verified, "issues_found", []) or []),
                "request_id": request_id,
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
            return {"subagent_output": output}

        except Exception as exc:
            logger.exception("CampaignInsightAgent.run failed: %s", exc)
            output = SubagentOutput(
                request_id=request_id,
                status=AgentStatus.ERROR,
                interpretation=Interpretation(summary="The analysis failed."),
                caveats=["Internal error"],
                metadata={
                    "model": self.model_name,
                    "total_duration_ms": int((time.monotonic() - t0) * 1000),
                    "error": str(exc),
                },
            )
            return {"subagent_output": output}

    # ---------------------------------------------------------------- helpers

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
        caveats: list[str] = []
        if isinstance(step_results, dict):
            results = list(step_results.values())
        else:
            results = list(step_results or [])

        for sr in results:
            status = getattr(sr, "status", None)
            step_id = getattr(sr, "step_id", "?")
            if status == StepStatus.ERROR:
                caveats.append(f"Step {step_id} failed; results may be incomplete.")
            elif status == StepStatus.TIMEOUT:
                caveats.append(f"Step {step_id} timed out; results may be incomplete.")
            elif status == StepStatus.PARTIAL:
                caveats.append(f"Step {step_id} returned partial data.")

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

        try:
            audience_role = getattr(dim_config.audience.role, "value", str(dim_config.audience.role))
            content_role = getattr(dim_config.content.role, "value", str(dim_config.content.role))
            if audience_role == "none":
                caveats.append("Audience dimension is disabled for this query.")
            if content_role == "none":
                caveats.append("Content dimension is disabled for this query.")
        except Exception:  # pragma: no cover
            pass

        seen: set[str] = set()
        deduped: list[str] = []
        for c in caveats:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped
