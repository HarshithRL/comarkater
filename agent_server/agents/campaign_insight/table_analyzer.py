"""Deterministic pandas-based table summarization for LLM consumption."""
from __future__ import annotations

import json
import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from agents.campaign_insight.contracts import TableSummary

logger = logging.getLogger(__name__)

_FULL_MODE_ROW_THRESHOLD = 20
_TOP_N = 5
_DIST_TOP = 10
_ANOMALY_CAP = 20
_TRUNCATED_CAP = 5
_MAX_PAYLOAD_CHARS = 6000


class TableAnalyzer:
    """Summarize a result table deterministically (no LLM calls)."""

    def analyze(
        self, columns: list[dict], data_array: list[list], row_count: int
    ) -> TableSummary:
        """Produce a ``TableSummary`` appropriate for LLM context.

        Small tables (<= 20 rows) are returned as ``mode="full"`` with the
        raw data intact. Larger tables are analyzed into statistical,
        categorical, anomaly, and aggregate summaries.

        Args:
            columns: Schema entries (dicts with ``name`` key).
            data_array: Row-major list of lists.
            row_count: Total row count reported by Genie.

        Returns:
            A populated ``TableSummary``.
        """
        if row_count <= _FULL_MODE_ROW_THRESHOLD:
            return TableSummary(
                mode="full", row_count=row_count,
                full_data=data_array, schema=columns,
            )
        try:
            return self._analyze_large(columns, data_array, row_count)
        except Exception:  # noqa: BLE001 — defensive: log + degrade
            logger.warning(
                "table_analyzer.analyze_failed rows=%d falling_back", row_count,
                exc_info=True,
            )
            return TableSummary(
                mode="full", row_count=row_count,
                full_data=data_array[:50], schema=columns,
            )

    def _analyze_large(
        self, columns: list[dict], data_array: list[list], row_count: int
    ) -> TableSummary:
        col_names: list[str] = [
            str(c.get("name", f"col_{i}")) for i, c in enumerate(columns)
        ]
        df = pd.DataFrame(data_array, columns=cast(Any, col_names))
        numeric_cols: list[str] = []
        for c in col_names:
            coerced = cast(pd.Series, pd.to_numeric(df[c], errors="coerce"))
            if coerced.notna().sum() > 0 and coerced.notna().mean() >= 0.5:
                df[c] = coerced
                numeric_cols.append(c)
        cat_cols = [c for c in col_names if c not in numeric_cols]
        top_rows, bottom_rows = self._top_bottom(df, numeric_cols)
        summary = TableSummary(
            mode="analyzed",
            row_count=row_count,
            schema=columns,
            statistical_summary=self._statistical_summary(df, numeric_cols),
            categorical_distribution=self._categorical_distribution(df, numeric_cols),
            anomalies=self._detect_anomalies(df, numeric_cols),
            top_rows=top_rows,
            bottom_rows=bottom_rows,
            aggregates=self._aggregates(df, numeric_cols, cat_cols),
            full_data=None,
        )
        if self._approx_size(summary) > _MAX_PAYLOAD_CHARS:
            summary.categorical_distribution = {
                k: v[:_TRUNCATED_CAP]
                for k, v in summary.categorical_distribution.items()
            }
            summary.anomalies = summary.anomalies[:_TRUNCATED_CAP]
            summary.top_rows = summary.top_rows[:_TRUNCATED_CAP]
            summary.bottom_rows = summary.bottom_rows[:_TRUNCATED_CAP]
            if summary.aggregates:
                summary.aggregates = {
                    k: (dict(list(v.items())[:_TRUNCATED_CAP])
                        if isinstance(v, dict) else v)
                    for k, v in summary.aggregates.items()
                }
        return summary

    @staticmethod
    def _statistical_summary(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
        out: dict[str, dict[str, float]] = {}
        for c in numeric_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            try:
                out[c] = {
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "std": round(float(s.std(ddof=0)) if len(s) > 0 else 0.0, 4),
                }
            except Exception:  # noqa: BLE001
                logger.debug("stat_summary skip column=%s", c, exc_info=True)
        return out

    @staticmethod
    def _categorical_distribution(
        df: pd.DataFrame, numeric_cols: list[str]
    ) -> dict[str, list[list[Any]]]:
        out: dict[str, list[list[Any]]] = {}
        for c in df.columns:
            if c in numeric_cols:
                continue
            try:
                counts = df[c].astype(object).fillna("NA").value_counts().head(_DIST_TOP)
                out[c] = [[str(k), int(v)] for k, v in counts.items()]
            except Exception:  # noqa: BLE001
                logger.debug("cat_dist skip column=%s", c, exc_info=True)
        return out

    @staticmethod
    def _detect_anomalies(
        df: pd.DataFrame, numeric_cols: list[str]
    ) -> list[dict[str, Any]]:
        anomalies: list[dict[str, Any]] = []
        for c in numeric_cols:
            nc = int(df[c].isna().sum())
            if nc > 0:
                anomalies.append({"type": "null_count", "column": c, "count": nc})
        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) < 3:
                continue
            mean = float(s.mean())
            std = float(s.std(ddof=0))
            if std == 0 or np.isnan(std):
                continue
            z = (df[c] - mean) / std
            for idx, zv in z.items():
                if pd.isna(zv) or abs(zv) <= 2.0:
                    continue
                try:
                    val = float(df[c].iloc[idx])
                except Exception:  # noqa: BLE001
                    continue
                anomalies.append({
                    "type": "outlier_2sigma", "column": c,
                    "row_index": int(idx), "value": round(val, 4),
                    "z_score": round(float(zv), 3),
                })
                if len(anomalies) >= _ANOMALY_CAP * 2:
                    break
            if len(anomalies) >= _ANOMALY_CAP * 2:
                break
        for c in numeric_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            if float((s != 0).mean()) >= 0.8:
                for idx in df.index[df[c] == 0].tolist()[:5]:
                    anomalies.append({
                        "type": "unexpected_zero", "column": c,
                        "row_index": int(idx), "value": 0,
                    })
        return anomalies[:_ANOMALY_CAP]

    @staticmethod
    def _top_bottom(
        df: pd.DataFrame, numeric_cols: list[str]
    ) -> tuple[list[list], list[list]]:
        if not numeric_cols:
            return [], []
        primary: str | None = None
        best_mean = -float("inf")
        for c in numeric_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            m = float(s.mean())
            if m > best_mean:
                best_mean = m
                primary = c
        if primary is None:
            primary = numeric_cols[0]
        try:
            sdf = df.sort_values(by=primary, ascending=False, na_position="last")
            return (
                _jsonable_rows(sdf.head(_TOP_N).values.tolist()),
                _jsonable_rows(sdf.tail(_TOP_N).values.tolist()),
            )
        except Exception:  # noqa: BLE001
            logger.debug("top_bottom failed column=%s", primary, exc_info=True)
            return [], []

    @staticmethod
    def _aggregates(
        df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]
    ) -> dict[str, dict[str, float]]:
        if len(numeric_cols) != 1 or len(cat_cols) != 1:
            return {}
        cat, num = cat_cols[0], numeric_cols[0]
        try:
            summed = cast(pd.Series, df.groupby(cat, dropna=False)[num].sum(min_count=1))
            grouped = summed.sort_values(ascending=False).head(_DIST_TOP)
            return {cat: {
                str(k): round(float(v), 4)
                for k, v in grouped.items() if pd.notna(v)
            }}
        except Exception:  # noqa: BLE001
            logger.debug("aggregates failed cat=%s num=%s", cat, num, exc_info=True)
            return {}

    @staticmethod
    def _approx_size(summary: TableSummary) -> int:
        try:
            return len(json.dumps({
                "statistical_summary": summary.statistical_summary,
                "categorical_distribution": summary.categorical_distribution,
                "anomalies": summary.anomalies,
                "top_rows": summary.top_rows,
                "bottom_rows": summary.bottom_rows,
                "aggregates": summary.aggregates,
            }, default=str))
        except Exception:  # noqa: BLE001
            return 0


def _jsonable_rows(rows: list[list]) -> list[list]:
    out: list[list] = []
    for r in rows:
        new_row: list = []
        for v in r:
            if isinstance(v, np.integer):
                new_row.append(int(v))
            elif isinstance(v, np.floating):
                fv = float(v)
                new_row.append(None if np.isnan(fv) else round(fv, 4))
            elif v is None:
                new_row.append(None)
            else:
                try:
                    if not isinstance(v, (list, dict, str)) and pd.isna(v):
                        new_row.append(None)
                        continue
                except Exception:  # noqa: BLE001
                    pass
                new_row.append(v)
        out.append(new_row)
    return out
