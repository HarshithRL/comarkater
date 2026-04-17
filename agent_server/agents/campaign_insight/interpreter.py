"""Produce Interpretation objects from TableSummary + domain context."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import (
    Interpretation,
    PatternMatch,
    StepResult,
    StepStatus,
)
from agents.campaign_insight.prompts.interpretation_prompt import INTERPRETATION_PROMPT

logger = logging.getLogger(__name__)


class _PatternModel(BaseModel):
    name: str = ""
    description: str = ""
    likely_causes: list[str] = Field(default_factory=list)
    severity: str = "info"


class _InterpretationModel(BaseModel):
    summary: str = ""
    insights: list[str] = Field(default_factory=list)
    patterns: list[_PatternModel] = Field(default_factory=list)
    severity: str = "info"


class Interpreter:
    """Single LLM-call interpretation with deterministic pattern augmentation."""

    def __init__(self, llm: Any, domain_knowledge: Any) -> None:
        """Initialize.

        Args:
            llm: Pre-built LLM instance with ``with_structured_output``.
            domain_knowledge: ``InsightAgentDomainKnowledge`` instance.
        """
        self.llm = llm
        self.domain_knowledge = domain_knowledge

    def interpret(
        self,
        step_results: dict[int, StepResult],
        intent: dict,
        dim_config: Any,
    ) -> Interpretation:
        """Build an Interpretation grounded in step_results and domain rules."""
        step_results_block = self._summarize_step_results(step_results)
        metric_values = self._extract_metric_values(step_results)
        metric_thresholds = self._collect_metric_thresholds(metric_values, intent)
        combination_patterns = self._format_combination_patterns()
        channel_rules = self._collect_channel_rules(intent, step_results)

        detected_patterns = self._detect_deterministic_patterns(metric_values)

        user_prompt = INTERPRETATION_PROMPT.format(
            query=intent.get("query", "") or intent.get("user_query", ""),
            step_results=step_results_block,
            metric_thresholds=metric_thresholds,
            combination_patterns=combination_patterns,
            channel_rules=channel_rules,
            primary_analysis=dim_config.primary_analysis,
        )

        try:
            structured_llm = self.llm.with_structured_output(_InterpretationModel)
            result: _InterpretationModel = structured_llm.invoke(
                [
                    SystemMessage(content="You are the Campaign Insight Agent interpreter."),
                    HumanMessage(content=user_prompt),
                ]
            )
            llm_patterns = [
                PatternMatch(
                    name=p.name,
                    description=p.description,
                    likely_causes=list(p.likely_causes or []),
                    severity=p.severity or "info",
                )
                for p in (result.patterns or [])
            ]
            merged = self._merge_patterns(detected_patterns, llm_patterns)
            interpretation = Interpretation(
                summary=result.summary or "",
                insights=list(result.insights or []),
                patterns=merged,
                severity=result.severity or self._roll_up_severity(merged),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Interpretation LLM call failed: %s", exc)
            interpretation = Interpretation(
                summary="Interpretation could not be generated.",
                insights=[],
                patterns=detected_patterns,
                severity=self._roll_up_severity(detected_patterns),
            )
        try:
            from langgraph.config import get_stream_writer
            get_stream_writer()({
                "event_type": "phase_progress",
                "phase": "interpret",
                "insights": len(interpretation.insights),
            })
        except Exception:
            pass
        return interpretation

    # ---- helpers -------------------------------------------------------

    def _summarize_step_results(self, step_results: dict[int, StepResult]) -> str:
        lines: list[str] = []
        for sid, sr in sorted(step_results.items()):
            if sr.status != StepStatus.SUCCESS or sr.table_summary is None:
                lines.append(
                    f"- step {sid} [{sr.dimension}] status={sr.status.value} "
                    f"error={sr.error_message or ''}"
                )
                continue
            ts = sr.table_summary
            lines.append(
                f"- step {sid} [{sr.dimension}] rows={ts.row_count} "
                f"aggregates={ts.aggregates} "
                f"stats={ts.statistical_summary} "
                f"top={ts.top_rows[:3] if ts.top_rows else []} "
                f"bottom={ts.bottom_rows[:3] if ts.bottom_rows else []}"
            )
        return "\n".join(lines) or "(no step results)"

    def _extract_metric_values(
        self, step_results: dict[int, StepResult]
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for sr in step_results.values():
            if sr.table_summary is None:
                continue
            for key, val in (sr.table_summary.aggregates or {}).items():
                if isinstance(val, (int, float)):
                    values[key] = float(val)
            for key, stats in (sr.table_summary.statistical_summary or {}).items():
                if isinstance(stats, dict):
                    mean = stats.get("mean")
                    if isinstance(mean, (int, float)):
                        values.setdefault(key, float(mean))
        return values

    def _collect_metric_thresholds(
        self, metric_values: dict[str, float], intent: dict
    ) -> str:
        names = set(metric_values.keys())
        for m in intent.get("metrics", []) or []:
            if isinstance(m, str):
                names.add(m)
        out: list[str] = []
        for name in sorted(names):
            try:
                th = self.domain_knowledge.get_metric_thresholds(name)
                if th:
                    out.append(f"- {name}: {th}")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(out) or "(no threshold data)"

    def _format_combination_patterns(self) -> str:
        try:
            patterns = self.domain_knowledge.detect_combination_patterns({})
            if patterns:
                return "\n".join(f"- {p.name}: {p.description}" for p in patterns)
        except Exception:  # noqa: BLE001
            pass
        try:
            formatted = self.domain_knowledge.format_for_subagent()
            return formatted[:2000]
        except Exception:  # noqa: BLE001
            return "(combination patterns unavailable)"

    def _collect_channel_rules(
        self, intent: dict, step_results: dict[int, StepResult]
    ) -> str:
        channels = set()
        for ch in intent.get("channels", []) or []:
            if isinstance(ch, str):
                channels.add(ch)
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
        return "\n".join(out) or "(no channel rules)"

    def _detect_deterministic_patterns(
        self, metric_values: dict[str, float]
    ) -> list[PatternMatch]:
        if not metric_values:
            return []
        try:
            detected = self.domain_knowledge.detect_combination_patterns(metric_values)
        except Exception as exc:  # noqa: BLE001
            logger.debug("detect_combination_patterns failed: %s", exc)
            return []
        out: list[PatternMatch] = []
        for p in detected or []:
            if isinstance(p, PatternMatch):
                out.append(p)
            elif isinstance(p, dict):
                out.append(
                    PatternMatch(
                        name=p.get("name", ""),
                        description=p.get("description", ""),
                        likely_causes=list(p.get("likely_causes", []) or []),
                        severity=p.get("severity", "info"),
                    )
                )
        return out

    def _merge_patterns(
        self,
        deterministic: list[PatternMatch],
        from_llm: list[PatternMatch],
    ) -> list[PatternMatch]:
        by_name: dict[str, PatternMatch] = {}
        for p in deterministic + from_llm:
            key = (p.name or "").strip().lower()
            if not key:
                continue
            if key not in by_name:
                by_name[key] = p
        return list(by_name.values())

    def _roll_up_severity(self, patterns: list[PatternMatch]) -> str:
        order = {"info": 0, "warning": 1, "critical": 2}
        worst = 0
        for p in patterns:
            worst = max(worst, order.get((p.severity or "info").lower(), 0))
        for name, rank in order.items():
            if rank == worst:
                return name
        return "info"
