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
    "performance_lookup": "bar",
    "strategic_recommendation": "bar",
    "what_if": "bar",
    "deep_dive": "bar",
}

# Only skip visualization for non-data intents (nothing to chart).
# For EVERY other intent, if any step produced a display_table the chart is
# MANDATORY — we fall back to a deterministic spec if the LLM path fails,
# so the UI never shows an empty viz slot beside a non-empty table.
_SKIP_INTENTS = {
    "clarification",
    "out_of_scope",
    "greeting",
}

# Default chart_type when the intent isn't explicitly mapped.
_DEFAULT_CHART_TYPE = "bar"

# Heuristic hint: first-column names that indicate the x-axis is ordered
# (time / period / wave / cohort) and a line chart tells the story better
# than bars. Lowercased substring match.
_TEMPORAL_HINTS = (
    "wave", "month", "week", "day", "date", "quarter", "year",
    "period", "time", "hour",
)


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
        def _emit(
            chart_type: Optional[str],
            skipped: bool,
            spec: Optional[dict] = None,
        ) -> None:
            try:
                from langgraph.config import get_stream_writer
                get_stream_writer()({
                    "event_type": "chart_ready",
                    "chart_type": chart_type,
                    "skipped": skipped,
                    "chart_spec": spec,
                    "item_id": "chart",
                })
            except Exception:
                pass

        it = (intent_type or "").strip().lower()
        if it in _SKIP_INTENTS:
            _emit(None, True)
            return None

        data_points = self._extract_points(step_results)
        if not data_points:
            # Genuinely no data to chart — nothing we can do.
            _emit(_CHART_TYPE_MAP.get(it, _DEFAULT_CHART_TYPE), True)
            return None

        # Chart type: intent hint first, then schema-shape refinement. If the
        # first column of the first usable table looks temporal (wave, date,
        # month, ...), prefer a line chart even when the intent isn't
        # explicitly "trend_analysis".
        chart_type = _CHART_TYPE_MAP.get(it, _DEFAULT_CHART_TYPE)
        _first_col = (data_points[0][1][0] if data_points[0][1] else "").lower()
        if any(hint in _first_col for hint in _TEMPORAL_HINTS):
            chart_type = "line"

        horizontal = it == "ranking"

        spec: Optional[dict] = None
        try:
            spec = self._llm_chart_spec(
                intent_type=it,
                chart_type=chart_type,
                data_points=data_points,
                horizontal=horizontal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChartBuilder LLM call failed: %s", exc)

        # Validate the LLM spec; fall back to a deterministic chart built
        # directly from the display table if anything is missing or malformed.
        if not self._is_valid_spec(spec):
            logger.info("ChartBuilder: using deterministic fallback spec")
            spec = self._fallback_spec(chart_type, data_points, horizontal)

        _emit(chart_type, False, spec)
        return spec

    # ---- helpers -------------------------------------------------------

    def _extract_points(
        self, step_results: dict[int, StepResult]
    ) -> list[tuple[str, list[str], list[list]]]:
        """Return list of (title, columns, rows) per usable step.

        Chart rendering is mandatory whenever data exists, so we accept any
        step that produced a non-empty display_table — including PARTIAL
        steps (e.g. contract-validation failed but Genie still returned
        rows). Only ERROR and TIMEOUT steps are skipped, because they
        carry no display_table.
        """
        out: list[tuple[str, list[str], list[list]]] = []
        for sid in sorted(step_results):
            sr = step_results[sid]
            if sr.status in (StepStatus.ERROR, StepStatus.TIMEOUT):
                continue
            if sr.display_table is None:
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

    # ---- validation + deterministic fallback ---------------------------

    def _is_valid_spec(self, spec: Optional[dict]) -> bool:
        """True iff `spec` has a non-empty series that Highcharts can render."""
        if not isinstance(spec, dict):
            return False
        series = spec.get("series")
        if not isinstance(series, list) or not series:
            return False
        for s in series:
            if not isinstance(s, dict):
                return False
            d = s.get("data")
            if not isinstance(d, list) or not d:
                return False
        return True

    def _fallback_spec(
        self,
        chart_type: str,
        data_points: list[tuple[str, list[str], list[list]]],
        horizontal: bool,
    ) -> dict:
        """Build a guaranteed-renderable Highcharts spec directly from tables.

        Picks the first numeric column as the y-series and the first column as
        the category axis. Uses only the first table in ``data_points`` so the
        chart stays readable when multiple steps exist.
        """
        title, cols, rows = data_points[0]

        # Locate the first numeric column; category column = first non-numeric
        # (or column 0 if everything is numeric).
        def _is_number(v) -> bool:
            if isinstance(v, bool):
                return False
            if isinstance(v, (int, float)):
                return True
            if isinstance(v, str):
                try:
                    float(v.replace(",", "").replace("%", "").strip())
                    return True
                except Exception:
                    return False
            return False

        def _to_num(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.replace(",", "").replace("%", "").strip())
                except Exception:
                    return 0.0
            return 0.0

        num_col_idx = -1
        for idx in range(len(cols)):
            col_values = [row[idx] for row in rows if idx < len(row)]
            if col_values and all(_is_number(v) for v in col_values):
                num_col_idx = idx
                break

        if num_col_idx == -1:
            # No numeric column — render a count-per-category chart on col 0.
            cat_idx = 0
            counts: dict[str, int] = {}
            for row in rows:
                key = str(row[cat_idx]) if cat_idx < len(row) else ""
                counts[key] = counts.get(key, 0) + 1
            categories = list(counts.keys())
            series_data = [counts[k] for k in categories]
            y_title = "count"
        else:
            cat_idx = 0 if num_col_idx != 0 else (1 if len(cols) > 1 else 0)
            categories = []
            series_data = []
            for row in rows:
                cat = str(row[cat_idx]) if cat_idx < len(row) else ""
                val = _to_num(row[num_col_idx]) if num_col_idx < len(row) else 0.0
                categories.append(cat)
                series_data.append(val)
            y_title = cols[num_col_idx] if num_col_idx < len(cols) else "value"

        spec: dict = {
            "chart": {"type": chart_type},
            "chart_type": chart_type,
            "title": {"text": title},
            "xAxis": {"categories": categories, "title": {"text": cols[cat_idx] if cat_idx < len(cols) else ""}},
            "yAxis": {"title": {"text": y_title}},
            "series": [{"name": y_title, "data": series_data}],
            "credits": {"enabled": False},
        }
        if horizontal:
            spec["chart"]["inverted"] = True
        return spec
