"""Assemble Highcharts chart_spec dicts from DisplayTable + intent."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import StepResult, StepStatus

logger = logging.getLogger(__name__)

_CHART_TYPE_MAP = {
    "trend_analysis": "line",
    "comparison": "bar",
    "funnel_analysis": "funnel",
    "ranking": "bar",  # rendered horizontally via chart options
    "diagnostic": "line",
}

_SKIP_INTENTS = {
    "performance_lookup",
    "clarification",
    "out_of_scope",
    "greeting",
}


class _ChartSpecModel(BaseModel):
    chart_type: str = "line"
    title: str = ""
    series: list[dict] = Field(default_factory=list)
    xAxis: dict = Field(default_factory=dict)
    yAxis: dict = Field(default_factory=dict)


class ChartBuilder:
    """Produce a Highcharts 11.x options dict - exactly one LLM call."""

    def __init__(self, llm: Any) -> None:
        """Initialize.

        Args:
            llm: Pre-built LLM instance.
        """
        self.llm = llm

    def build_chart(
        self,
        intent_type: str,
        step_results: dict[int, StepResult],
    ) -> Optional[dict]:
        """Return a Highcharts options dict or ``None`` when not chartable."""
        def _emit(chart_type: Optional[str], skipped: bool) -> None:
            try:
                from langgraph.config import get_stream_writer
                get_stream_writer()({
                    "event_type": "chart_ready",
                    "chart_type": chart_type,
                    "skipped": skipped,
                })
            except Exception:
                pass

        it = (intent_type or "").strip().lower()
        if it in _SKIP_INTENTS:
            _emit(None, True)
            return None

        chart_type = _CHART_TYPE_MAP.get(it)
        if not chart_type:
            _emit(None, True)
            return None

        data_points = self._extract_points(step_results)
        if not data_points:
            _emit(chart_type, True)
            return None

        total_rows = sum(len(rows) for _, _, rows in data_points)
        if total_rows < 2:
            _emit(chart_type, True)
            return None

        horizontal = it == "ranking"

        try:
            spec = self._llm_chart_spec(
                intent_type=it,
                chart_type=chart_type,
                data_points=data_points,
                horizontal=horizontal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChartBuilder LLM call failed: %s", exc)
            _emit(chart_type, True)
            return None

        _emit(chart_type, False)
        return spec

    # ---- helpers -------------------------------------------------------

    def _extract_points(
        self, step_results: dict[int, StepResult]
    ) -> list[tuple[str, list[str], list[list]]]:
        """Return list of (title, columns, rows) per usable step."""
        out: list[tuple[str, list[str], list[list]]] = []
        for sid in sorted(step_results):
            sr = step_results[sid]
            if sr.status != StepStatus.SUCCESS or sr.display_table is None:
                continue
            dt = sr.display_table
            if not dt.columns or not dt.rows:
                continue
            # Cap rows to keep prompts small.
            rows = list(dt.rows[:50])
            out.append((dt.title or f"step {sid}", list(dt.columns), rows))
        return out

    def _llm_chart_spec(
        self,
        intent_type: str,
        chart_type: str,
        data_points: list[tuple[str, list[str], list[list]]],
        horizontal: bool,
    ) -> Optional[dict]:
        payload_lines: list[str] = []
        for title, cols, rows in data_points:
            payload_lines.append(
                f"TABLE '{title}' columns={cols} rows={rows}"
            )
        payload = "\n".join(payload_lines)

        system_content = (
            "You produce Highcharts 11.x options dictionaries for marketing analytics. "
            "Return precise series/xAxis/yAxis/title fields. "
            "Do not use currency symbols. Do not name internal systems."
        )
        orientation = (
            "Render as a horizontal bar (swap x/y axes)." if horizontal else ""
        )
        user_content = (
            f"Intent type: {intent_type}\n"
            f"Chart type to produce: {chart_type}\n"
            f"{orientation}\n"
            f"DATA:\n{payload}\n\n"
            "Return a Highcharts options object with fields: "
            "chart_type, title, series (list of dicts with name + data), "
            "xAxis, yAxis."
        )

        try:
            structured_llm = self.llm.with_structured_output(_ChartSpecModel)
            result: _ChartSpecModel = structured_llm.invoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_content),
                ]
            )
            spec = {
                "chart_type": result.chart_type or chart_type,
                "title": {"text": result.title or ""},
                "series": result.series or [],
                "xAxis": result.xAxis or {},
                "yAxis": result.yAxis or {},
            }
        except Exception:
            # Fallback: plain invoke + JSON parse.
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(
                        content=user_content + "\nReturn JSON only."
                    ),
                ]
            )
            text = getattr(response, "content", "") or ""
            try:
                spec = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChartBuilder JSON parse failed: %s", exc)
                return None

        if horizontal and isinstance(spec, dict):
            spec.setdefault("chart", {})
            if isinstance(spec["chart"], dict):
                spec["chart"]["inverted"] = True

        return spec
