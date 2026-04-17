"""Verify interpretation + recommendations against evidence (VerificationResult)."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import (
    Interpretation,
    PatternMatch,
    Recommendation,
    StepResult,
    StepStatus,
    VerificationResult,
)
from agents.campaign_insight.prompts.reflection_prompt import REFLECTION_PROMPT

logger = logging.getLogger(__name__)


class _CorrectedPattern(BaseModel):
    name: str = ""
    description: str = ""
    likely_causes: list[str] = Field(default_factory=list)
    severity: str = "info"


class _CorrectedInterpretation(BaseModel):
    summary: str = ""
    insights: list[str] = Field(default_factory=list)
    patterns: list[_CorrectedPattern] = Field(default_factory=list)
    severity: str = "info"


class _CorrectedRecommendation(BaseModel):
    action: str = ""
    detail: str = ""
    expected_impact: str = ""
    evidence: str = ""
    source_pattern: str = ""


class _ReflectionModel(BaseModel):
    passed: bool = True
    issues_found: list[str] = Field(default_factory=list)
    fixes_applied: list[str] = Field(default_factory=list)
    corrected_interpretation: Optional[_CorrectedInterpretation] = None
    corrected_recommendations: Optional[list[_CorrectedRecommendation]] = None


class Reflector:
    """Single-pass verifier. Fails open on LLM error."""

    def __init__(self, llm: Any, domain_knowledge: Any) -> None:
        """Initialize.

        Args:
            llm: Pre-built LLM instance.
            domain_knowledge: ``InsightAgentDomainKnowledge`` instance.
        """
        self.llm = llm
        self.domain_knowledge = domain_knowledge

    def verify(
        self,
        interpretation: Interpretation,
        recommendations: list[Recommendation],
        step_results: dict[int, StepResult],
    ) -> VerificationResult:
        """Run one verification pass and return a VerificationResult."""
        step_data_summary = self._format_step_data_summary(step_results)
        channel_constraints = self._collect_channel_constraints(step_results)

        user_prompt = REFLECTION_PROMPT.format(
            interpretation=self._format_interpretation(interpretation),
            recommendations=self._format_recommendations(recommendations),
            step_data_summary=step_data_summary,
            channel_constraints=channel_constraints,
        )

        try:
            structured_llm = self.llm.with_structured_output(_ReflectionModel)
            result: _ReflectionModel = structured_llm.invoke(
                [
                    SystemMessage(
                        content="You are the Reflector for the Campaign Insight Agent."
                    ),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflector LLM call failed; failing open: %s", exc)
            verification = VerificationResult(
                passed=True,
                issues_found=[],
                fixes_applied=[],
                interpretation=interpretation,
                recommendations=recommendations,
            )
        else:
            if result.passed:
                verification = VerificationResult(
                    passed=True,
                    issues_found=list(result.issues_found or []),
                    fixes_applied=list(result.fixes_applied or []),
                    interpretation=interpretation,
                    recommendations=recommendations,
                )
            else:
                corrected_interp = interpretation
                if result.corrected_interpretation is not None:
                    ci = result.corrected_interpretation
                    corrected_interp = Interpretation(
                        summary=ci.summary or interpretation.summary,
                        insights=list(ci.insights or []),
                        patterns=[
                            PatternMatch(
                                name=p.name,
                                description=p.description,
                                likely_causes=list(p.likely_causes or []),
                                severity=p.severity or "info",
                            )
                            for p in (ci.patterns or [])
                        ],
                        severity=ci.severity or interpretation.severity or "info",
                    )

                corrected_recs = recommendations
                if result.corrected_recommendations is not None:
                    corrected_recs = [
                        Recommendation(
                            action=r.action,
                            detail=r.detail,
                            expected_impact=r.expected_impact,
                            evidence=r.evidence,
                            source_pattern=r.source_pattern,
                        )
                        for r in result.corrected_recommendations
                        if r.action
                    ]

                verification = VerificationResult(
                    passed=False,
                    issues_found=list(result.issues_found or []),
                    fixes_applied=list(result.fixes_applied or []),
                    interpretation=corrected_interp,
                    recommendations=corrected_recs,
                )

        try:
            from langgraph.config import get_stream_writer
            get_stream_writer()({
                "event_type": "phase_progress",
                "phase": "reflect",
                "rewritten": bool(verification.interpretation) and not verification.passed,
            })
        except Exception:
            pass
        return verification

    # ---- helpers -------------------------------------------------------

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

    def _format_recommendations(self, recs: list[Recommendation]) -> str:
        lines: list[str] = []
        for i, r in enumerate(recs or [], 1):
            lines.append(
                f"{i}. action={r.action} | detail={r.detail} | "
                f"impact={r.expected_impact} | evidence={r.evidence} | "
                f"src={r.source_pattern}"
            )
        return "\n".join(lines) or "(no recommendations)"

    def _format_step_data_summary(
        self, step_results: dict[int, StepResult]
    ) -> str:
        lines: list[str] = []
        for sid, sr in sorted(step_results.items()):
            if sr.status != StepStatus.SUCCESS or sr.table_summary is None:
                lines.append(
                    f"- step {sid} [{sr.dimension}] status={sr.status.value}"
                )
                continue
            ts = sr.table_summary
            lines.append(
                f"- step {sid} [{sr.dimension}] rows={ts.row_count} "
                f"aggregates={ts.aggregates} stats={ts.statistical_summary} "
                f"top={ts.top_rows[:3] if ts.top_rows else []} "
                f"bottom={ts.bottom_rows[:3] if ts.bottom_rows else []}"
            )
        return "\n".join(lines) or "(no step data)"

    def _collect_channel_constraints(
        self, step_results: dict[int, StepResult]
    ) -> str:
        channels: set[str] = set()
        for sr in step_results.values():
            if sr.table_summary is None:
                continue
            for key, dist in (sr.table_summary.categorical_distribution or {}).items():
                if "channel" in key.lower() and isinstance(dist, dict):
                    for ch in dist:
                        if isinstance(ch, str):
                            channels.add(ch)
        out: list[str] = []
        for ch in sorted(channels):
            try:
                rules = self.domain_knowledge.get_channel_constraints(ch)
                if rules:
                    out.append(f"- {ch}: {rules}")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(out) or "(no channel constraints)"
