"""Adaptive execution planner for the Campaign Insight Agent.

Produces a minimal :class:`ExecutionPlan` via one structured LLM call. The
plan is constrained by the dimension budgets handed down from the
:class:`DimensionClassifier` + :class:`DimensionValidator`.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import (
    DimensionClassification,
    ExecutionPlan,
    PlanStep,
)
from agents.campaign_insight.domain_knowledge import InsightAgentDomainKnowledge

logger = logging.getLogger(__name__)


_DIMENSION_LITERAL = Literal["campaign", "audience", "content"]
_MAX_STEPS_HARD = 8


class _PlanStepSchema(BaseModel):
    """Pydantic schema for one plan step."""

    step_id: int = Field(ge=1, description="1-indexed step id.")
    dimension: _DIMENSION_LITERAL = Field(description="Which dimension this step belongs to.")
    query: str = Field(description="Natural-language question to send to Genie.")
    purpose: str = Field(description="One-sentence rationale for the step.")
    depends_on: list[int] = Field(
        default_factory=list,
        description="Step ids that must complete before this step runs.",
    )
    can_parallel: bool = Field(
        default=True,
        description="False if this step has dependencies; otherwise True.",
    )


class _ExecutionPlanSchema(BaseModel):
    """Pydantic schema mirroring :class:`ExecutionPlan`."""

    steps: list[_PlanStepSchema] = Field(default_factory=list)


_SYSTEM_PROMPT = """You are the Adaptive Planner for a marketing analytics agent.

Produce the minimum set of Genie queries that answers the user's question
given the active dimensions and their per-dimension query budgets.

Planning rules
--------------
- Each step targets exactly one dimension ("campaign", "audience", "content").
- Never exceed a dimension's budget; never exceed the total_budget cap.
- Simpler is better - if 1 step suffices (common for performance_lookup),
  return 1 step.
- Dependency rules:
    * Audience drill-downs usually depend on a prior campaign step that
      identifies which campaign(s) to drill into.
    * Content benchmark queries (from the content-insights table) can run
      in parallel with everything - NO dependencies.
    * If ``depends_on`` is non-empty, ``can_parallel`` MUST be false.
- Use step ids 1..N in issue order.
- Each ``query`` field MUST be a PLAIN ENGLISH natural-language question.
  STRICTLY FORBIDDEN in the query field: SQL keywords (SELECT, FROM, WHERE,
  JOIN, GROUP BY, ORDER BY, WITH, UNION, LIMIT), table names
  (campaign_details_metric_view_v2, audience_metric_view,
  campaign_content_metric_view, igp_content_insights), snake_case column
  names, backticks, or semicolons. Describe WHAT you want in English — the
  data tool does the SQL translation itself.

Domain vocabulary
-----------------
- Channels: email, sms, apn (mobile push), bpn (browser push), whatsapp.
- Metrics: CTR, CVR, open_rate, bounce_rate, unsub_rate, complaint_rate,
  delivery_rate, CTOR, rev_per_delivered, rev_per_click, conv_from_click.
"""


class AdaptivePlanner:
    """Generate an :class:`ExecutionPlan` via one structured LLM call."""

    def __init__(self, llm, domain_knowledge: InsightAgentDomainKnowledge) -> None:
        """Initialize the planner.

        Args:
            llm: Pre-configured LangChain chat model supporting
                ``with_structured_output``. Caller controls temperature.
            domain_knowledge: Shared domain-knowledge helper, used for
                minimum-volume thresholds.
        """
        self._llm = llm
        self._domain = domain_knowledge

    def plan(
        self,
        query: str,
        intent: dict,
        dimension_config: DimensionClassification,
        supervisor_plan: Optional[list[str]] = None,
    ) -> ExecutionPlan:
        """Produce an :class:`ExecutionPlan` for the given query.

        Args:
            query: Raw user question.
            intent: Dict with ``intent_type`` / ``complexity`` keys.
            dimension_config: Validated dimension roles + budgets.
            supervisor_plan: Optional high-level plan from the supervisor
                (a list of NL step strings). If given, the planner validates
                and re-expresses each step as a :class:`PlanStep`.

        Returns:
            :class:`ExecutionPlan`. On any failure, returns a minimal
            single-step plan answering the user's question directly.
        """
        total_budget = dimension_config.total_budget
        max_steps = max(1, min(total_budget, _MAX_STEPS_HARD))
        intent_type = (intent or {}).get("intent_type", "")
        intent_complexity = (intent or {}).get("complexity", "")

        active_lines: list[str] = []
        for name in ("campaign", "audience", "content"):
            cfg = getattr(dimension_config, name)
            active_lines.append(
                f"- {name}: role={cfg.role.value} budget={cfg.budget}"
            )

        min_vols = self._domain.get_minimum_volume_thresholds()
        min_vol_str = ", ".join(f"{k}={v}" for k, v in min_vols.items() if v) or "(none specified)"

        user_parts: list[str] = [
            f"User query:\n{query.strip()}",
            "",
            f"Intent: {intent_type} (complexity={intent_complexity})",
            "",
            "Active dimensions:",
            *active_lines,
            "",
            f"Total budget cap: {total_budget} queries (hard max {max_steps} steps).",
            "",
            f"Minimum volume thresholds: {min_vol_str}",
        ]

        if supervisor_plan:
            user_parts.extend(
                [
                    "",
                    "Supervisor-suggested plan (validate and re-express; "
                    f"do NOT exceed {len(supervisor_plan)} steps):",
                    *(f"  {i + 1}. {s}" for i, s in enumerate(supervisor_plan)),
                ]
            )
            max_steps = min(max_steps, len(supervisor_plan))
        else:
            user_parts.extend(
                [
                    "",
                    "No supervisor plan provided - generate the minimal plan "
                    "(often 1 step).",
                ]
            )

        user_parts.extend(
            [
                "",
                f"Return at most {max_steps} steps. Each step's dimension must "
                "be an active dimension with remaining budget.",
            ]
        )
        user_prompt = "\n".join(user_parts)

        try:
            structured = self._llm.with_structured_output(_ExecutionPlanSchema)
            raw: _ExecutionPlanSchema = structured.invoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            plan = self._to_dataclass(raw, total_budget=total_budget, max_steps=max_steps)
            logger.info(
                "AdaptivePlanner produced %d step(s) total_budget=%d",
                len(plan.steps),
                plan.total_budget,
            )
            return plan
        except Exception as exc:  # noqa: BLE001 - defensive fallback
            logger.error(
                "AdaptivePlanner failed (%s) - using single-step fallback.", exc,
                exc_info=True,
            )
            return self._fallback_plan(query, total_budget)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_dataclass(
        raw: _ExecutionPlanSchema,
        total_budget: int,
        max_steps: int,
    ) -> ExecutionPlan:
        """Convert the LLM schema output into :class:`ExecutionPlan`."""
        steps: list[PlanStep] = []
        for idx, s in enumerate(raw.steps[:max_steps], start=1):
            depends_on = [d for d in s.depends_on if d < s.step_id]
            can_parallel = bool(s.can_parallel) and not depends_on
            steps.append(
                PlanStep(
                    step_id=int(s.step_id) if s.step_id >= 1 else idx,
                    dimension=str(s.dimension),
                    query=str(s.query).strip(),
                    purpose=str(s.purpose).strip(),
                    depends_on=list(depends_on),
                    can_parallel=can_parallel,
                )
            )
        if not steps:
            steps = [
                PlanStep(
                    step_id=1,
                    dimension="campaign",
                    query="",
                    purpose="Answer the user's question",
                    depends_on=[],
                    can_parallel=True,
                )
            ]
        return ExecutionPlan(steps=steps, total_budget=total_budget)

    @staticmethod
    def _fallback_plan(query: str, total_budget: int) -> ExecutionPlan:
        """One-step plan used when the LLM call fails."""
        return ExecutionPlan(
            steps=[
                PlanStep(
                    step_id=1,
                    dimension="campaign",
                    query=query.strip(),
                    purpose="Answer the user's question",
                    depends_on=[],
                    can_parallel=True,
                )
            ],
            total_budget=max(1, total_budget),
        )
