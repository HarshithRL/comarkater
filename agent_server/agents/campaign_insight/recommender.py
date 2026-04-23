"""Translate interpretations and patterns into evidence-backed Recommendations."""
from __future__ import annotations

import logging
from typing import Any

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import (
    Interpretation,
    Recommendation,
    StepResult,
    StepStatus,
)
from agents.campaign_insight.prompts.interpretation_prompt import RECOMMENDATION_PROMPT

logger = logging.getLogger(__name__)

_SKIP_INTENTS = {"greeting", "clarification"}


class _RecommendationModel(BaseModel):
    action: str = ""
    detail: str = ""
    expected_impact: str = ""
    evidence: str = ""
    source_pattern: str = ""
    category: str = ""  # "apply" | "avoid" | "explore"


class _RecommendationListModel(BaseModel):
    recommendations: list[_RecommendationModel] = Field(default_factory=list)
    # Section J "Contextual Nudge" — a natural, business-oriented follow-up
    # question tied to the findings. Rendered as the final line of the
    # recommendations block.
    nudge: str = ""


class Recommender:
    """Produce actionable, evidence-tied recommendations."""

    def __init__(self, llm: Any, domain_knowledge: Any) -> None:
        """Initialize.

        Args:
            llm: Pre-built LLM instance.
            domain_knowledge: ``InsightAgentDomainKnowledge`` instance.
        """
        self.llm = llm
        self.domain_knowledge = domain_knowledge

    def recommend(
        self,
        interpretation: Interpretation,
        step_results: dict[int, StepResult],
        intent: dict,
    ) -> list[Recommendation]:
        """Return a merged, deduped list of recommendations (up to 6)."""
        intent_type = (intent.get("intent_type") or "").strip().lower()
        nudge_text = ""
        if intent_type in _SKIP_INTENTS:
            final: list[Recommendation] = []
        else:
            base = self._collect_deterministic(interpretation, step_results)
            try:
                llm_pair = self._call_llm(interpretation, step_results, intent_type)
                from_llm: list[Recommendation] = llm_pair[0]
                nudge_text = llm_pair[1]
                merged = base + from_llm
                final = self._dedupe_and_cap(merged)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Recommendation LLM call failed: %s", exc)
                final = self._dedupe_and_cap(base)

        try:
            from dataclasses import asdict as _asdict

            from langgraph.config import get_stream_writer
            writer = get_stream_writer()
            writer({
                "event_type": "phase_progress",
                "phase": "recommend",
                "recs": len(final),
            })
            writer({
                "event_type": "recommendations_ready",
                "item_id": "recommendations",
                "recommendations": [_asdict(r) for r in final],
                "nudge": nudge_text,
            })
        except Exception:
            pass
        return final

    # ---- helpers -------------------------------------------------------

    def _collect_deterministic(
        self,
        interpretation: Interpretation,
        step_results: dict[int, StepResult],
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []

        # Pattern-sourced recommendations.
        # NOTE: get_recommendations_for_pattern may return either
        # list[Recommendation] or list[str] depending on the YAML shape.
        # Handle both to avoid nesting a Recommendation inside action=.
        for pattern in interpretation.patterns or []:
            try:
                items = self.domain_knowledge.get_recommendations_for_pattern(
                    pattern.name
                )
            except Exception:  # noqa: BLE001
                items = []
            for item in items or []:
                if isinstance(item, Recommendation):
                    recs.append(item)
                    continue
                recs.append(
                    Recommendation(
                        action=str(item),
                        detail="",
                        expected_impact="",
                        evidence=pattern.description or "",
                        source_pattern=f"pattern:{pattern.name}",
                    )
                )

        # Channel-sourced recommendations (for underperforming channels).
        underperforming = (interpretation.severity or "info").lower() in {
            "warning",
            "critical",
        }
        channels = self._extract_channels(step_results)
        for ch in channels:
            try:
                strings = self.domain_knowledge.get_channel_recommendations(
                    ch, underperforming=underperforming
                )
            except Exception:  # noqa: BLE001
                strings = []
            for s in strings or []:
                recs.append(
                    Recommendation(
                        action=s,
                        detail="",
                        expected_impact="",
                        evidence=f"Channel {ch} flagged as {interpretation.severity or 'info'}",
                        source_pattern=f"channel:{ch}",
                    )
                )
        return recs

    def _extract_channels(self, step_results: dict[int, StepResult]) -> list[str]:
        channels: set[str] = set()
        for sr in step_results.values():
            if sr.table_summary is None:
                continue
            for key, dist in (sr.table_summary.categorical_distribution or {}).items():
                if "channel" in key.lower() and isinstance(dist, dict):
                    for ch in dist:
                        if isinstance(ch, str):
                            channels.add(ch)
        return sorted(channels)

    def _call_llm(
        self,
        interpretation: Interpretation,
        step_results: dict[int, StepResult],
        intent_type: str,
    ) -> tuple[list[Recommendation], str]:
        interp_block = self._format_interpretation(interpretation)
        step_block = self._format_step_results(step_results)
        channel_recs = self._format_channel_recs(step_results)

        user_prompt = RECOMMENDATION_PROMPT.format(
            interpretation=interp_block,
            step_results=step_block,
            intent_type=intent_type or "(unspecified)",
            channel_recommendations=channel_recs,
        )
        with mlflow.start_span(name="recommender_llm") as _span:
            _span.set_inputs({
                "interpretation_block": interp_block[:2000],
                "step_results_block": step_block[:4000],
                "channel_recommendations": channel_recs[:1500],
                "intent_type": intent_type or "(unspecified)",
                "prompt_length": len(user_prompt),
                "user_prompt_preview": user_prompt[:4000],
            })
            structured_llm = self.llm.with_structured_output(_RecommendationListModel)
            result: _RecommendationListModel = structured_llm.invoke(
                [
                    SystemMessage(
                        content="You are the Campaign Insight Agent recommendation engine."
                    ),
                    HumanMessage(content=user_prompt),
                ]
            )
            _span.set_outputs({
                "recommendations_count": len(result.recommendations or []),
                "nudge": (result.nudge or "")[:500],
            })
        out: list[Recommendation] = []
        for r in result.recommendations or []:
            if not r.action:
                continue
            if not (r.evidence or "").strip():
                # Synthesize evidence from step_results instead of dropping the rec.
                # Find first non-empty aggregate or top_row value to anchor the evidence.
                synthesized = self._synthesize_evidence(step_results)
                if synthesized:
                    r.evidence = synthesized
                    logger.info(
                        "Recommender: synthesized evidence for action=%r (LLM omitted citation)",
                        r.action[:60],
                    )
                else:
                    logger.warning(
                        "Recommender: dropping rec with no evidence and no synthesizable fallback: action=%r",
                        r.action[:60],
                    )
                    continue
            cat = (r.category or "").strip().lower()
            if cat not in ("apply", "avoid", "explore"):
                cat = "apply"
            out.append(
                Recommendation(
                    action=r.action,
                    detail=r.detail,
                    expected_impact=r.expected_impact,
                    evidence=r.evidence,
                    source_pattern=r.source_pattern or "llm",
                    category=cat,
                )
            )
        return out, (result.nudge or "").strip()

    def _format_interpretation(self, interp: Interpretation) -> str:
        patterns = "; ".join(
            f"{p.name}({p.severity})" for p in (interp.patterns or [])
        )
        insights = "\n".join(f"  - {i}" for i in (interp.insights or []))
        return (
            f"summary: {interp.summary}\n"
            f"severity: {interp.severity}\n"
            f"patterns: {patterns}\n"
            f"insights:\n{insights}"
        )

    def _format_step_results(self, step_results: dict[int, StepResult]) -> str:
        lines: list[str] = []
        for sid, sr in sorted(step_results.items()):
            if sr.status != StepStatus.SUCCESS or sr.table_summary is None:
                continue
            ts = sr.table_summary
            lines.append(
                f"- step {sid} [{sr.dimension}] rows={ts.row_count} "
                f"aggregates={ts.aggregates} top={ts.top_rows[:3] if ts.top_rows else []}"
            )
        return "\n".join(lines) or "(no step results)"

    def _format_channel_recs(self, step_results: dict[int, StepResult]) -> str:
        lines: list[str] = []
        for ch in self._extract_channels(step_results):
            try:
                recs = self.domain_knowledge.get_channel_recommendations(
                    ch, underperforming=True
                )
            except Exception:  # noqa: BLE001
                recs = []
            if recs:
                lines.append(f"- {ch}: {recs}")
        return "\n".join(lines) or "(none)"

    def _synthesize_evidence(self, step_results: dict[int, StepResult]) -> str:
        """Build a generic evidence string from available aggregates when LLM omits citation."""
        for sid in sorted(step_results.keys()):
            sr = step_results[sid]
            if sr.table_summary is None:
                continue
            ts = sr.table_summary
            if ts.aggregates:
                first_key = next(iter(ts.aggregates))
                first_val = ts.aggregates[first_key]
                return f"step_{sid} aggregates: {first_key}={first_val}"
            if ts.statistical_summary:
                first_key = next(iter(ts.statistical_summary))
                first_val = ts.statistical_summary[first_key]
                if isinstance(first_val, dict):
                    mean = first_val.get("mean", "")
                    return f"step_{sid} {first_key} mean={mean}"
            if ts.top_rows:
                return f"step_{sid} top_row sample: {ts.top_rows[0]}"
        return ""

    def _dedupe_and_cap(self, recs: list[Recommendation]) -> list[Recommendation]:
        seen: set[str] = set()
        out: list[Recommendation] = []
        for r in recs:
            # Defensive: action should be str, but upstream bugs may nest a
            # Recommendation here. Coerce to str before dedup so we never
            # crash the whole request on a shape issue.
            action_val = r.action
            if isinstance(action_val, Recommendation):
                action_val = getattr(action_val, "action", "") or ""
            key = (str(action_val) if action_val else "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            r.action = str(action_val)
            out.append(r)
            if len(out) >= 6:
                break
        return out
