"""Assemble the final SubagentOutput envelope for the supervisor."""
from __future__ import annotations

import logging
from typing import Optional

from agents.campaign_insight.contracts import (
    AgentStatus,
    DisplayTable,
    Interpretation,
    Recommendation,
    StepResult,
    StepStatus,
    SubagentOutput,
)

logger = logging.getLogger(__name__)


class OutputBuilder:
    """Pure assembly - no LLM calls."""

    def build(
        self,
        request_id: str,
        step_results: dict[int, StepResult],
        interpretation: Interpretation,
        recommendations: list[Recommendation],
        chart_spec: Optional[dict],
        caveats: list[str],
        metadata: dict,
    ) -> SubagentOutput:
        """Build and return a ``SubagentOutput``.

        Args:
            request_id: Upstream request id to echo back.
            step_results: Mapping of ``step_id`` to ``StepResult``.
            interpretation: Final interpretation (post reflection).
            recommendations: Final recommendations (post reflection).
            chart_spec: Optional Highcharts options dict.
            caveats: List of caveat strings (e.g. BPN conversion).
            metadata: Arbitrary metadata (must include ``total_duration_ms``).
        """
        # Sorted tables for display.
        tables_for_display: list[DisplayTable] = []
        for sid in sorted(step_results):
            dt = step_results[sid].display_table
            if dt is not None:
                tables_for_display.append(dt)

        # Overall status roll-up.
        statuses = [sr.status for sr in step_results.values()]
        if statuses and all(s == StepStatus.SUCCESS for s in statuses):
            status = AgentStatus.SUCCESS
        elif any(s == StepStatus.SUCCESS for s in statuses):
            status = AgentStatus.PARTIAL
        else:
            status = AgentStatus.ERROR

        success_count = sum(1 for s in statuses if s == StepStatus.SUCCESS)

        # Genie call count: prefer iterations_used total, fall back to step count.
        genie_calls_made = sum(
            max(sr.iterations_used, 0) for sr in step_results.values()
        )
        if genie_calls_made == 0:
            genie_calls_made = len(step_results)

        dimensions_activated = sorted(
            {
                sr.dimension
                for sr in step_results.values()
                if sr.status == StepStatus.SUCCESS and sr.dimension
            }
        )

        execution_summary = {
            "steps_planned": len(step_results),
            "steps_completed": success_count,
            "genie_calls_made": genie_calls_made,
            "dimensions_activated": dimensions_activated,
            "total_duration_ms": metadata.get("total_duration_ms", 0),
        }

        return SubagentOutput(
            request_id=request_id,
            status=status,
            execution_summary=execution_summary,
            tables_for_display=tables_for_display,
            chart_spec=chart_spec,
            interpretation=interpretation,
            recommendations=list(recommendations or []),
            caveats=list(caveats or []),
            metadata=dict(metadata or {}),
        )
