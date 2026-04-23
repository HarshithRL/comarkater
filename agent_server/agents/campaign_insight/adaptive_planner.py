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
from agents.campaign_insight.query_router import RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)


_DIMENSION_LITERAL = Literal["campaign", "audience", "content"]
_MAX_STEPS_HARD = 4


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

Your job: produce the MINIMUM set of Genie queries that answers the user's
question. Default to ONE step. Decompose only when the data truly lives in
different tables, or when a later step depends on an earlier step's output.

Genie is a schema-aware NL-to-SQL engine — trust it to do the heavy data
work. Your job is to hand it ONE rich, well-scoped question per step. The
LLM does reasoning (patterns, verdicts, recommendations) in LATER phases.

==============================================================================
GENIE CAPABILITY ENVELOPE — what Genie handles in ONE call
==============================================================================
A single NL question to Genie can deliver:
  - Multi-metric aggregation (sent, delivered, opened, clicked, revenue
    together in one row per group).
  - Multi-group breakdowns (per channel, per campaign_type, per segment,
    per emotion, per template_type).
  - Comparisons ("email vs SMS", "Regular vs A/B", "last month vs this
    month") — Genie returns both sides in one result set.
  - Period-over-period trends (WoW, MoM) grouped however you specify.
  - Top-N / bottom-N with ties and minimum-volume filters.
  - Distributions per group.
  - Cross-column correlations within one table.
  - Simple joins when the schema exposes the relationship.

So ONE step should pull the WHOLE aggregated cut Genie can deliver — not a
sliver you will stitch together across steps.

==============================================================================
DECOMPOSITION RULES — when to split, when to fuse
==============================================================================
DEFAULT = 1 step. Before issuing step 2 or step 3, prove it is necessary.

SPLIT INTO MULTIPLE STEPS ONLY WHEN one of these is true:
  1. CROSS-TABLE: The question needs facts from two different data objects
     that do not share a join path — e.g., campaign_details +
     audience_metric_view, or campaign content + igp_content_insights
     benchmark.
  2. DEPENDENCY: Step 2's phrasing requires a VALUE produced by step 1
     (e.g., "drill into the top 3 campaigns from step 1 by segment").
  3. BENCHMARK COMPARISON: A content-feature vs client-benchmark question
     needs both the measured-values table AND the optimal-values table.

FUSE INTO ONE STEP WHEN any of these apply:
  - Two candidate steps share the same table family AND the same filter
    set AND the same time window, differing only in metric or grouping.
  - Multiple metrics on the same dimension ("show opens, clicks, and
    revenue per channel") — one multi-metric request.
  - Comparison of two values the same table can produce ("Regular vs A/B
    test", "email vs SMS") — fuse.
  - Period comparison inside one table ("last month vs this month") — fuse.
  - Top-N + its supporting metrics (sent/delivered/clicked alongside the
    ranked rate) — fuse.

NEVER split the SAME dimension into multiple granular steps just to
"cover different angles" — one well-phrased question returns the full cut.
Granular decomposition wastes calls, loses cross-row context, and forces
the interpreter to reconstruct relationships Genie already resolved.

==============================================================================
USER-QUERY FIDELITY (non-negotiable)
==============================================================================
- Preserve the user's exact top-N count, metric name, time range, and
  any filters they mentioned.
- Do NOT add thresholds, volume floors, or filters the user did not
  specify (except the minimum_volume thresholds required for rankings).
- Do NOT substitute metrics (open rate stays open rate — never CTR, CTOR).
- Do NOT change the count or the time window ("top 30" / "last month"
  stay literal — do not rewrite to "top 10" or "last 30 days").
- You MAY lightly rephrase for Genie consumption (add channel
  disambiguation, or add "per campaign / per segment / per channel"
  grouping phrases), but user parameters stay literal.

==============================================================================
CAUSAL-PAIRING RULE — content & audience exist to explain campaign outcomes
==============================================================================
Campaign is always the OUTCOME (Y). Content and audience are DIAGNOSTIC
LENSES that must explain Y. Never ask for content or audience facts in
isolation.

When the active dimensions include CONTENT:
  - The content step MUST fetch actual feature values AND the corresponding
    optimal benchmark values, so the interpreter can compute X-vs-X* gaps.
  - Ask for feature importance signals (click feature points, conversion
    feature points) so the interpreter can weight which gaps actually
    hurt performance.
  - Pair the content cut with the relevant campaign outcome metric
    (CTR / CVR / open_rate / conversion) in the same or a parallel step,
    so the interpreter can answer "did this gap cause the drop?"

When the active dimensions include AUDIENCE:
  - The audience step MUST return per-segment send, click, and conversion
    volumes so the interpreter can compute lift = click_share − send_share.
  - Group by the segment attribute the user implied (lifecycle, intent,
    communication health, device, retarget frequency).
  - Pair with the campaign outcome metric so the interpreter can answer
    "which segments drove or suppressed Y?"

When BOTH content and audience are active:
  - Step phrasing must allow the interpreter to combine them into a
    "right message × right user" narrative tied to the campaign outcome.

==============================================================================
OUTPUT-SHAPE CONTRACT
==============================================================================
Every step's NL query MUST request an AGGREGATED, interpreter-friendly
result:
  - Use verbs: "average", "total", "top N by", "per <dimension>",
    "distribution of", "compare X vs Y by", "trend of X over <window>".
  - Never ask for raw per-row transaction details unless the user
    explicitly requested a raw list.
  - Expected rows per step <= 50. Group by real dimension columns only
    (campaign, channel, segment, emotion, template_type, content_type,
    lifecycle_stage, campaign_type).
  - NEVER group by a derived metric (CTR, CVR, open_rate, bounce_rate,
    conversion_rate, CTOR). These are values, not dimensions.
  - When the user wants a ranking, include the supporting funnel metrics
    (sent / delivered / clicked / converted) alongside the ranked rate in
    the SAME step so the interpreter can explain the ranking.

==============================================================================
QUERY STRING FORMAT — plain English ONLY
==============================================================================
The `query` field is a natural-language question passed to Genie.
STRICTLY FORBIDDEN in the query field:
  - SQL keywords: SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, WITH,
    UNION, LIMIT.
  - Table names: campaign_details_metric_view_v2, audience_metric_view,
    campaign_content_metric_view, igp_content_insights.
  - snake_case column names, backticks, semicolons.

Describe WHAT you want in English — Genie writes the SQL.

==============================================================================
DIMENSION & DEPENDENCY MECHANICS
==============================================================================
- Each step targets exactly one dimension: "campaign", "audience", or
  "content".
- Never exceed a dimension's budget; never exceed the total_budget cap.
- `depends_on` lists earlier step_ids that must finish first.
- If `depends_on` is non-empty, `can_parallel` MUST be false.
- Audience and content steps CAN run in parallel with a campaign step
  when they are independent questions (no drill-down dependency).
- Use step ids 1..N in issue order.

==============================================================================
DOMAIN VOCABULARY
==============================================================================
- Channels: email, sms, apn (mobile push), bpn (browser push), whatsapp.
- Derived metrics: CTR, CVR, open_rate, bounce_rate, unsub_rate,
  complaint_rate, delivery_rate, CTOR, rev_per_delivered, rev_per_click,
  conv_from_click.

==============================================================================
WORKED EXAMPLES — prefer fewer, richer steps
==============================================================================
Q: "Top 5 email campaigns by open rate last month"
Plan: 1 step (campaign)
  "Show the top 5 email campaigns by open rate for last month, with
   sent, delivered, opened, and open rate per campaign."

Q: "Compare email and SMS CTR this month vs last month"
Plan: 1 step (campaign) — do NOT split by channel or by month
  "Compare click-through rate for email and SMS campaigns between last
   month and this month, aggregated per channel per month with sent,
   clicked, and CTR."

Q: "Monthly trend of WhatsApp open rate, Jan to Mar 2026"
Plan: 1 step (campaign)
  "Show the monthly open rate for WhatsApp campaigns from January 2026
   through March 2026, with sent, delivered, and opened per month."

Q: "Lifecycle stage distribution of WhatsApp campaign audiences (sends,
    opens, clicks per segment)"
Plan: 2 steps (cross-table)
  step 1 (campaign): "Show total sent, delivered, opened, and clicked
                      for WhatsApp campaigns in the requested window."
  step 2 (audience, can_parallel=true):
                     "Show the lifecycle-stage distribution for WhatsApp
                      campaign audiences, with sent, opened, and clicked
                      per lifecycle segment."

Q: "Which WhatsApp campaigns have content quality below the client
    benchmark?"
Plan: 2 steps (benchmark comparison — two tables)
  step 1 (content): "For WhatsApp campaigns in the recent window, show
                     per-campaign content feature scores (CTA, emotion,
                     subject line, readability, overall click score)."
  step 2 (content, can_parallel=true):
                    "Show the client optimal-value benchmarks for
                     WhatsApp content features (CTA, emotion, subject
                     line, readability, overall click score)."

Q: "For our top 10 email campaigns last month, show lifecycle breakdown
    AND dominant-emotion breakdown"
Plan: 3 steps (cross-dimension with drill-down dependency)
  step 1 (campaign): "Show the top 10 email campaigns last month with
                      sent, delivered, opened, clicked, and open/click
                      rates per campaign."
  step 2 (audience, depends_on=[1]): "For the top 10 email campaigns
                      identified above, show the lifecycle-stage
                      distribution with sent, opened, and clicked per
                      segment."
  step 3 (content, depends_on=[1], can_parallel=false):
                     "For those same campaigns, show the dominant-emotion
                      distribution with opens and clicks per emotion."

Return the minimal plan. Prefer 1 step. Never exceed the given step or
budget cap.
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
        routing: Optional[RoutingDecision] = None,
    ) -> ExecutionPlan:
        """Produce an :class:`ExecutionPlan` for the given query.

        Args:
            query: Raw user question.
            intent: Dict with ``intent_type`` / ``complexity`` keys.
            dimension_config: Validated dimension roles + budgets.
            supervisor_plan: Optional high-level plan from the supervisor
                (a list of NL step strings). If given, the planner validates
                and re-expresses each step as a :class:`PlanStep`.
            routing: Optional :class:`RoutingDecision`. When the strategy is
                ``GENIE_DIRECT`` or ``HYBRID`` the plan is forced to a single
                step. The user query is NOT mutated; the output-shape
                contract in the system prompt handles aggregation phrasing.

        Returns:
            :class:`ExecutionPlan`. On any failure, returns a minimal
            single-step plan answering the user's question directly.
        """
        total_budget = dimension_config.total_budget
        max_steps = max(1, min(total_budget, _MAX_STEPS_HARD))
        if routing is not None and routing.strategy != RoutingStrategy.AGENT_DECOMPOSE:
            max_steps = 1
        intent_type = (intent or {}).get("intent_type", "")
        intent_complexity = (intent or {}).get("complexity", "")

        active_lines: list[str] = []
        for name in ("campaign", "audience", "content"):
            cfg = getattr(dimension_config, name)
            active_lines.append(
                f"- {name}: role={cfg.role.value} budget={cfg.budget}"
            )

        user_parts: list[str] = [
            f"User query:\n{query.strip()}",
            "",
            f"Intent: {intent_type} (complexity={intent_complexity})",
            "",
            "Active dimensions:",
            *active_lines,
            "",
            f"Total budget cap: {total_budget} queries (hard max {max_steps} steps).",
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
            for step in plan.steps:
                logger.info(
                    "PLANNER.step_nl_query | user_query=%r | planner_nl=%r",
                    query[:200], step.query[:200],
                )
            try:
                from langgraph.config import get_stream_writer
                get_stream_writer()({
                    "event_type": "plan_ready",
                    "item_id": "plan",
                    "steps": [
                        {"step_id": s.step_id, "query": s.query[:120], "dim": s.dimension}
                        for s in plan.steps
                    ],
                    "budget": plan.total_budget,
                })
            except Exception:
                pass
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
