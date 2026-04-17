"""Deterministic user-facing table formatter (Indian grouping, percent rules)."""
from __future__ import annotations

import logging
import math
from typing import Any

from agents.campaign_insight.contracts import DisplayTable

logger = logging.getLogger(__name__)

_RATE_TOKENS = ("rate", "ctr", "cvr", "ctor", "pct", "%")
_COUNT_TOKENS = (
    "sent", "delivered", "opened", "clicked", "conversion", "conversions",
    "count", "revenue", "bounce", "unsub", "unsubscribe", "complaint",
)
_MAX_ROWS = 1000


class TableBuilder:
    """Format Genie result rows for end-user display."""

    def build_display_table(
        self,
        columns: list[dict],
        data_array: list[list],
        title: str,
        sql: str = "",
    ) -> DisplayTable:
        """Build a ``DisplayTable`` with formatted string cells.

        Args:
            columns: Genie column schema entries (dicts with ``name``).
            data_array: Row-major list of lists.
            title: Table title shown to the user.
            sql: Source SQL for traceability (optional).

        Returns:
            A ``DisplayTable`` ready for UI rendering.
        """
        col_names = [c.get("name", f"col_{i}") for i, c in enumerate(columns)]

        # Sniff column kinds + percent detection (fraction vs already-percent)
        col_kinds: list[str] = []
        percent_is_fraction: list[bool] = []
        for i, name in enumerate(col_names):
            kind = self._classify(name)
            col_kinds.append(kind)
            if kind == "rate":
                percent_is_fraction.append(self._percent_is_fraction(data_array, i))
            else:
                percent_is_fraction.append(False)

        rows_out: list[list[str]] = []
        for raw_row in data_array[:_MAX_ROWS]:
            formatted: list[str] = []
            for i, cell in enumerate(raw_row):
                kind = col_kinds[i] if i < len(col_kinds) else "other"
                is_frac = (
                    percent_is_fraction[i] if i < len(percent_is_fraction) else False
                )
                formatted.append(self._format_cell(cell, kind, is_frac))
            rows_out.append(formatted)

        return DisplayTable(
            title=title,
            columns=col_names,
            rows=rows_out,
            source_sql=sql,
        )

    # ------------------------------------------------------------------
    # Classification + per-cell formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(col_name: str) -> str:
        low = (col_name or "").lower()
        if any(tok in low for tok in _RATE_TOKENS):
            return "rate"
        if any(tok in low for tok in _COUNT_TOKENS):
            return "count"
        return "other"

    @staticmethod
    def _percent_is_fraction(data_array: list[list], col_idx: int) -> bool:
        """Return True if all non-null values in column are within [0, 1.0]."""
        seen_any = False
        for row in data_array:
            if col_idx >= len(row):
                continue
            v = row[col_idx]
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(f):
                continue
            seen_any = True
            if f < 0 or f > 1.0:
                return False
        return seen_any

    def _format_cell(self, value: Any, kind: str, percent_is_fraction: bool) -> str:
        """Format a single cell per domain rules. Returns a string."""
        if value is None:
            return "NA"

        if isinstance(value, str):
            if kind in ("rate", "count"):
                try:
                    numeric = float(value)
                    return self._format_numeric(numeric, kind, percent_is_fraction)
                except ValueError:
                    return value
            return value

        if isinstance(value, bool):
            return str(value)

        if isinstance(value, (int, float)):
            try:
                if isinstance(value, float) and math.isnan(value):
                    return "NA"
            except Exception:  # noqa: BLE001
                pass
            return self._format_numeric(float(value), kind, percent_is_fraction)

        return str(value)

    def _format_numeric(
        self, value: float, kind: str, percent_is_fraction: bool
    ) -> str:
        if kind == "rate":
            v = value * 100.0 if percent_is_fraction else value
            return f"{v:.2f}%"
        if kind == "count":
            if abs(value - round(value)) < 1e-9:
                return self._format_indian(int(round(value)))
            return f"{value:.2f}"
        # other floats
        if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
            return str(int(round(value)))
        return f"{value:.2f}"

    # ------------------------------------------------------------------
    # Indian-style grouping helper
    # ------------------------------------------------------------------
    @staticmethod
    def _format_indian(n: int) -> str:
        """Format an integer using Indian lakh/crore grouping.

        Examples:
            ``238100`` -> ``"2,38,100"``
            ``1234567`` -> ``"12,34,567"``
        """
        negative = n < 0
        s = str(abs(n))
        if len(s) <= 3:
            return ("-" + s) if negative else s
        last3 = s[-3:]
        rest = s[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        grouped_rest = ",".join(reversed(groups))
        out = f"{grouped_rest},{last3}"
        return f"-{out}" if negative else out
