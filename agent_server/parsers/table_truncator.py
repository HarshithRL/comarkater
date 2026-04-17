"""Budget-aware table truncation for LLM prompts.

Adapts the number of rows sent to the LLM based on a character budget
(proxy for token budget at ~4 chars/token). When truncating, appends
a statistical summary of ALL rows so the LLM retains aggregate context
without hallucinating about unseen individual rows.

Anti-hallucination: truncated output embeds clear labels instructing
the LLM to ground specific claims ONLY in visible rows and use
statistics ONLY for aggregate/distribution statements.
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)

# ── Budget constants (chars, ~4 chars per token) ──

BUDGET_ANALYSIS = 12_000              # genie_analysis — primary analysis, data + stats
BUDGET_FORMAT_SUPERVISOR = 6_000      # format_supervisor — chart generation, rows only
BUDGET_SYNTHESIZER_PER_WORKER = 2_000 # synthesizer — per worker (up to 6 = 12K total)
BUDGET_WORKER = 12_000                # genie_worker — source table text

STATS_RESERVE = 600   # chars reserved for statistical summary footer
MAX_STATS_ROWS = 10_000  # cap rows scanned for stats (safety)
MIN_VISIBLE_ROWS = 5  # always show at least this many rows


def typed_value(val):
    """Convert Genie string value to typed Python value.

    Genie data_array values are ALL strings: "47,704", "0.86%", "2025-11-23".
    Strips commas and %, tries int then float, else keeps original string.
    """
    if val is None:
        return "NA"
    clean = str(val).replace(",", "").replace("%", "").strip()
    try:
        return float(clean) if "." in clean else int(clean)
    except ValueError:
        return str(val)


def _parse_numeric(val) -> float | None:
    """Try to parse a value as a number. Returns None on failure."""
    if val is None:
        return None
    clean = str(val).replace(",", "").replace("%", "").strip()
    if not clean or clean.upper() in ("NA", "NULL", "NONE", ""):
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def _compute_stats_summary(columns: list[dict], data_array: list[list]) -> str:
    """Compute per-column statistics from ALL rows (up to MAX_STATS_ROWS).

    Numeric columns: min, max, mean, median.
    Categorical columns: unique count, top 3 most frequent.
    """
    total = len(data_array)
    rows = data_array[:MAX_STATS_ROWS]
    lines = [
        f"\n=== FULL DATASET STATISTICS ({total} rows) — AGGREGATE context only ===",
        "=== Use ONLY for overall distribution statements. "
        "NEVER attribute a statistic to a specific entity. ===",
    ]

    for col_idx, col in enumerate(columns):
        col_name = col.get("name", f"col_{col_idx}")
        try:
            raw_vals = [row[col_idx] if col_idx < len(row) else None for row in rows]

            # Try numeric parsing
            numeric_vals = []
            for v in raw_vals:
                n = _parse_numeric(v)
                if n is not None:
                    numeric_vals.append(n)

            is_numeric = len(numeric_vals) > len(raw_vals) * 0.5

            if is_numeric and numeric_vals:
                sorted_vals = sorted(numeric_vals)
                n = len(sorted_vals)
                median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
                mean = sum(sorted_vals) / n

                # Check if original values had % suffix
                has_pct = any("%" in str(v) for v in raw_vals[:20] if v is not None)
                suffix = "%" if has_pct else ""

                lines.append(
                    f"- {col_name}: min={sorted_vals[0]:g}{suffix}, "
                    f"max={sorted_vals[-1]:g}{suffix}, "
                    f"mean={mean:.2f}{suffix}, median={median:g}{suffix}"
                )
            else:
                # Categorical
                str_vals = [str(v) for v in raw_vals if v is not None and str(v).strip()]
                unique_count = len(set(str_vals))
                top3 = Counter(str_vals).most_common(3)
                top3_str = ", ".join(f'"{v}" ({c})' for v, c in top3)
                lines.append(f"- {col_name}: {unique_count} unique values. Top 3: {top3_str}")

        except Exception:
            # One bad column must not break the summary
            lines.append(f"- {col_name}: (stats unavailable)")

    if total > MAX_STATS_ROWS:
        lines.append(f"(Statistics computed from first {MAX_STATS_ROWS:,} of {total:,} rows)")

    return "\n".join(lines)


def _build_pipe_row(values: list, fallback: str = "NA") -> str:
    """Build a single pipe-delimited row from values (original strings)."""
    return " | ".join(str(v) if v is not None else fallback for v in values)


def truncate_table_for_llm(
    columns: list[dict],
    data_array: list[list],
    char_budget: int = 8_000,
    include_stats: bool = True,
) -> str:
    """Convert Genie {columns, data_array} → budget-aware text for LLM.

    If all rows fit within the budget, returns clean pipe-delimited text
    with no labels or stats (identical to the old _build_table_text output).

    If truncation is needed:
      - include_stats=True: appends statistical summary + anti-hallucination labels
      - include_stats=False: appends only a row count note (for chart generation)

    Args:
        columns: Genie column metadata [{name, type_name}, ...]
        data_array: Genie raw data [[val, val, ...], ...]
        char_budget: Target character budget (~4 chars per token)
        include_stats: Whether to append statistical summary when truncating

    Returns:
        Pipe-delimited table text, optionally with stats footer.
    """
    if not columns or not data_array:
        return "No data returned."

    headers = [col.get("name", f"col_{i}") for i, col in enumerate(columns)]
    header_line = " | ".join(headers)
    header_chars = len(header_line) + 1  # +1 for newline

    # Sample row width from first 10 rows
    sample = data_array[:10]
    sample_lines = [_build_pipe_row(row) for row in sample]
    avg_row_chars = (
        sum(len(line) + 1 for line in sample_lines) // len(sample_lines)
        if sample_lines else header_chars
    )
    # Guard against zero/tiny avg
    avg_row_chars = max(avg_row_chars, 10)

    # Compute how many rows fit
    reserved = STATS_RESERVE if include_stats else 0
    available = char_budget - reserved - header_chars
    max_rows = max(MIN_VISIBLE_ROWS, available // avg_row_chars)
    total_rows = len(data_array)

    truncated = total_rows > max_rows
    shown_rows = min(total_rows, max_rows)

    # Build table text
    parts = []

    if truncated and include_stats:
        parts.append(f"=== VISIBLE DATA ({shown_rows} of {total_rows} rows) ===")
        parts.append(
            "=== Ground ALL specific claims (campaign names, exact values, rankings) "
            "ONLY in these rows ==="
        )

    parts.append(header_line)
    for row in data_array[:shown_rows]:
        parts.append(_build_pipe_row(row))

    # Append stats or count note if truncated
    if truncated:
        if include_stats:
            parts.append(_compute_stats_summary(columns, data_array))
        else:
            parts.append(f"\n[Showing {shown_rows} of {total_rows} rows]")

    result = "\n".join(parts)

    logger.info(
        "TRUNCATOR: total=%d | shown=%d | cols=%d | budget=%d | actual=%d | stats=%s | truncated=%s",
        total_rows, shown_rows, len(columns), char_budget, len(result), include_stats, truncated,
    )

    return result
