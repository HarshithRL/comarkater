"""User-facing table passthrough — no value normalization, only null handling."""
from __future__ import annotations

import logging
import math
from typing import Any

from agents.campaign_insight.contracts import DisplayTable

logger = logging.getLogger(__name__)

_MAX_ROWS = 15


class TableBuilder:
    """Emit Genie result rows to the user with raw values preserved."""

    def build_display_table(
        self,
        columns: list[dict],
        data_array: list[list],
        title: str,
        sql: str = "",
    ) -> DisplayTable:
        """Build a ``DisplayTable`` with raw string-coerced cells.

        Values are passed through exactly as Genie returned them; only
        ``None`` and ``NaN`` are replaced with ``"NA"``. Capped at
        ``_MAX_ROWS`` rows for the user-facing / streamed view.

        Args:
            columns: Genie column schema entries (dicts with ``name``).
            data_array: Row-major list of lists.
            title: Table title shown to the user.
            sql: Source SQL for traceability (optional).

        Returns:
            A ``DisplayTable`` ready for UI rendering.
        """
        col_names = [c.get("name", f"col_{i}") for i, c in enumerate(columns)]

        rows_out: list[list[str]] = []
        for raw_row in data_array[:_MAX_ROWS]:
            rows_out.append([_cell_to_str(v) for v in raw_row])

        return DisplayTable(
            title=title,
            columns=col_names,
            rows=rows_out,
            source_sql=sql,
        )


def _cell_to_str(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return str(value)
