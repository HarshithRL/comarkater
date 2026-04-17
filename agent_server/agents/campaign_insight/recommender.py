"""Translate interpretations and patterns into evidence-backed Recommendations."""
from __future__ import annotations

import logging
from typing import Any

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

_SKIP_INTENTS = {"performance_lookup", "ranking", "greeting", "clarification"}


class _RecommendationModel(BaseModel):
    action: str = ""
    detail: str = ""
    expected_impact: str = ""
    evidence: str = ""
    source_pattern: str = ""


class _RecommendationListModel(BaseModel):
    recommendations: list[_RecommendationModel] = Field(default_factory=list)


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
        if intent_type in _SKIP_INTENTS:
            return []

        base = self._collect_deterministic(interpretation, step_results)

        try:
            from_llm = self._call_llm(interpretation, step_results, intent_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recommendation LLM call failed: %s", exc)
            return self._dedupe_and_cap(base)

        merged = base + from_llm
        return self._dedupe_and_cap(merged)

    # ---- helpers -------------------------------------------------------

    def _collect_deterministic(
        self,
        interpretation: Interpretation,
        step_results: dict[int, StepResult],
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []

        # Pattern-sourced recommendations.
        for pattern in interpretation.patterns or []:
            try:
                strings = self.domain_knowledge.get_recommendations_for_pattern(
                    pattern.name
                )
            except Exception:  # noqa: BLE001
                strings = []
            for s in strings or []:
                recs.append(
                    Recommendation(
                        action=s,
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
    ) -> list[Recommendation]:
        interp_block = self._format_interpretation(interpretation)
        step_block = self._format_step_results(step_results)
        channel_recs = self._format_channel_recs(step_results)

        user_prompt = RECOMMENDATION_PROMPT.format(
            interpretation=interp_block,
            step_results=step_block,
            intent_type=intent_type or "(unspecified)",
            channel_recommendations=channel_recs,
        )
        structured_llm = self.llm.with_structured_output(_RecommendationListModel)
        result: _RecommendationListModel = structured_llm.invoke(
            [
                SystemMessage(
                    content="You are the Campaign Insight Agent recommendation engine."
                ),
                HumanMessage(content=user_prompt),
            ]
        )
        out: list[Recommendation] = []
        for r in result.recommendations or []:
            if not r.action:
                continue
            if not (r.evidence or "").strip():
                # enforce non-empty evidence
                continue
            out.append(
                Recommendation(
                    action=r.action,
                    detail=r.detail,
                    expected_impact=r.expected_impact,
                    evidence=r.evidence,
                    source_pattern=r.source_pattern or "llm",
                )
            )
        return out

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

    def _dedupe_and_cap(self, recs: list[Recommendation]) -> list[Recommendation]:
        seen: set[str] = set()
        out: list[Recommendation] = []
        for r in recs:
            key = (r.action or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= 6:
                break
        return out
