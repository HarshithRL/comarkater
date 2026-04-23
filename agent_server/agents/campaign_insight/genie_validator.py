"""Output-shape validator for Genie responses.

Runs after ``ToolHandler.execute_query_with_retry`` and before
``TableAnalyzer.analyze``. Checks the returned data (row count, column count,
numeric coverage, aggregation level, metric-named grouping columns). Does NOT
parse SQL — SQL text is only recorded for forensics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agents.campaign_insight.contracts import GenieResponse

logger = logging.getLogger(__name__)


_FORBIDDEN_METRIC_COLUMNS: frozenset[str] = frozenset({
    "ctr",
    "cvr",
    "open_rate",
    "ctor",
    "bounce_rate",
    "conversion_rate",
    "click_through_rate",
    "click_to_open_rate",
    "unsub_rate",
    "complaint_rate",
    "delivery_rate",
})


@dataclass
class GenieContract:
    """Shape constraints the validator enforces."""

    max_rows: int = 50
    max_columns: int = 20
    require_numeric_column: bool = True
    forbid_metric_group_columns: frozenset[str] = _FORBIDDEN_METRIC_COLUMNS


@dataclass
class ValidationResult:
    """Outcome of a single validation pass."""

    passed: bool = True
    violations: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    has_numeric: bool = False


class GenieResultValidator:
    """Validate a :class:`GenieResponse` against the output-shape contract."""

    def __init__(self, contract: GenieContract | None = None) -> None:
        self.contract = contract or GenieContract()

    def validate(self, gr: GenieResponse) -> ValidationResult:
        """Run all checks and return a :class:`ValidationResult`."""
        columns = gr.columns or []
        data_array = gr.data_array or []
        row_count = int(gr.row_count or 0)
        column_count = len(columns)

        violations: list[str] = []

        if row_count > self.contract.max_rows:
            violations.append(
                f"row_count={row_count} exceeds cap {self.contract.max_rows}"
            )

        if column_count > self.contract.max_columns:
            violations.append(
                f"column_count={column_count} exceeds cap {self.contract.max_columns}"
            )

        numeric_cols = self._find_numeric_columns(columns, data_array)
        has_numeric = bool(numeric_cols)
        if self.contract.require_numeric_column and not has_numeric:
            violations.append(
                "no_numeric_column: result has no metric to reason over"
            )

        metric_group_col = self._find_metric_group_column(
            columns, data_array, numeric_cols
        )
        if metric_group_col:
            violations.append(
                f"metric_group_column: {metric_group_col!r} looks like a "
                f"derived-metric used as a grouping column"
            )

        agg_violation = self._check_aggregation_level(
            columns, data_array, row_count, numeric_cols
        )
        if agg_violation:
            violations.append(agg_violation)

        passed = not violations
        result = ValidationResult(
            passed=passed,
            violations=violations,
            row_count=row_count,
            column_count=column_count,
            has_numeric=has_numeric,
        )
        logger.info(
            "[DEBUG][GENIE_VALIDATE] passed=%s row_count=%s column_count=%s "
            "has_numeric=%s violations=%s",
            passed,
            row_count,
            column_count,
            has_numeric,
            violations,
        )
        return result

    def build_refinement_hint(self, vr: ValidationResult) -> str:
        """Return a natural-language hint to append to the next retry query."""
        parts: list[str] = []
        if any(v.startswith("row_count") for v in vr.violations):
            parts.append(
                f"The previous attempt returned {vr.row_count} rows. "
                f"Please aggregate to at most {self.contract.max_rows} rows "
                f"grouped by dimension columns (campaign, segment, emotion, "
                f"channel) only."
            )
        if any(v.startswith("aggregation_level") for v in vr.violations):
            parts.append(
                "The result does not look properly aggregated. Group by the "
                "dimension column(s) and compute averages/sums per group."
            )
        if any(v.startswith("metric_group_column") for v in vr.violations):
            parts.append(
                "Do not group by derived metrics such as CTR, CVR, open rate, "
                "bounce rate or conversion rate. Group by dimension columns "
                "only and compute the metric as an aggregate."
            )
        if any(v.startswith("no_numeric_column") for v in vr.violations):
            parts.append(
                "The result has no numeric metric. Include at least one "
                "aggregated metric (e.g. avg_ctr, sum_sent, count)."
            )
        if any(v.startswith("column_count") for v in vr.violations):
            parts.append(
                "The result has too many columns; select only dimension "
                "columns and aggregated metrics."
            )
        if not parts:
            parts.append(
                "Please return an aggregated result with at most "
                f"{self.contract.max_rows} rows."
            )
        return " ".join(parts)

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _find_numeric_columns(
        columns: list[dict], data_array: list[list]
    ) -> list[int]:
        """Return indexes of columns whose values are (mostly) numeric."""
        if not columns or not data_array:
            return []
        out: list[int] = []
        sample = data_array[:20]
        for idx in range(len(columns)):
            numeric_count = 0
            total = 0
            for row in sample:
                if idx >= len(row):
                    continue
                val = row[idx]
                if val is None:
                    continue
                total += 1
                if isinstance(val, (int, float)):
                    numeric_count += 1
                    continue
                if isinstance(val, str):
                    s = val.rstrip("%").replace(",", "").strip()
                    try:
                        float(s)
                        numeric_count += 1
                    except ValueError:
                        pass
            if total > 0 and numeric_count / total >= 0.6:
                out.append(idx)
        return out

    @staticmethod
    def _find_metric_group_column(
        columns: list[dict],
        data_array: list[list],
        numeric_col_idx: list[int],
    ) -> str:
        """Return the name of a forbidden metric column used as a group key.

        A column is considered a group key if its values are mostly distinct
        across rows AND its name matches a known derived-metric name.
        """
        if not columns or not data_array:
            return ""
        for idx, col in enumerate(columns):
            name = str(col.get("name", "")).strip().lower()
            if not name or name not in _FORBIDDEN_METRIC_COLUMNS:
                continue
            if idx in numeric_col_idx:
                # Numeric metric column — OK as a value, not a group key.
                continue
            values = [row[idx] for row in data_array[:50] if idx < len(row)]
            if not values:
                continue
            distinct = len({str(v) for v in values})
            if distinct >= max(2, len(values) // 2):
                return str(col.get("name", ""))
        return ""

    @staticmethod
    def _check_aggregation_level(
        columns: list[dict],
        data_array: list[list],
        row_count: int,
        numeric_col_idx: list[int],
    ) -> str:
        """Flag results that are clearly not aggregated to a dimension level."""
        if row_count <= 10 or not columns or not data_array:
            return ""
        categorical_idxs = [
            i for i in range(len(columns)) if i not in numeric_col_idx
        ]
        if len(categorical_idxs) != 1:
            return ""
        idx = categorical_idxs[0]
        distinct = len({
            str(row[idx]) for row in data_array if idx < len(row)
        })
        if distinct == 0:
            return ""
        if row_count > max(distinct, 50):
            return (
                f"aggregation_level: {row_count} rows for only {distinct} "
                f"distinct values in the grouping column"
            )
        return ""
