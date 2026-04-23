"""Unit tests for GenieResultValidator (output-shape checks only)."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from agents.campaign_insight.contracts import GenieResponse  # noqa: E402
from agents.campaign_insight.genie_validator import (  # noqa: E402
    GenieResultValidator,
)


def _gr(columns, data_array, row_count=None):
    return GenieResponse(
        columns=columns,
        data_array=data_array,
        row_count=len(data_array) if row_count is None else row_count,
        sql="SELECT 1",
        status="success",
    )


def test_aggregated_30_rows_passes():
    cols = [{"name": "segment"}, {"name": "avg_ctr"}]
    rows = [[f"seg-{i}", 0.01 * i] for i in range(30)]
    v = GenieResultValidator()
    r = v.validate(_gr(cols, rows))
    assert r.passed is True
    assert r.violations == []
    assert r.row_count == 30
    assert r.has_numeric is True


def test_oversize_row_count_fails():
    cols = [{"name": "campaign"}, {"name": "ctr"}]
    rows = [[f"c-{i}", 0.01] for i in range(214)]
    v = GenieResultValidator()
    r = v.validate(_gr(cols, rows))
    assert r.passed is False
    assert any(viol.startswith("row_count") for viol in r.violations)


def test_no_numeric_column_fails():
    cols = [{"name": "campaign"}, {"name": "channel"}]
    rows = [["c1", "email"], ["c2", "sms"]]
    v = GenieResultValidator()
    r = v.validate(_gr(cols, rows))
    assert r.passed is False
    assert any(viol.startswith("no_numeric_column") for viol in r.violations)


def test_metric_named_group_column_fails():
    cols = [{"name": "ctr"}, {"name": "sent"}]
    rows = [["low", 100], ["medium", 200], ["high", 300], ["vhigh", 400]]
    v = GenieResultValidator()
    r = v.validate(_gr(cols, rows))
    assert r.passed is False
    assert any("metric_group_column" in viol for viol in r.violations)


def test_refinement_hint_mentions_row_cap_and_aggregate():
    cols = [{"name": "campaign"}, {"name": "ctr"}]
    rows = [[f"c-{i}", 0.01] for i in range(200)]
    v = GenieResultValidator()
    r = v.validate(_gr(cols, rows))
    hint = v.build_refinement_hint(r)
    lower = hint.lower()
    assert "50" in hint
    assert "aggregat" in lower


def test_too_many_columns_fails():
    cols = [{"name": f"col_{i}"} for i in range(25)]
    rows = [[0.1 for _ in range(25)] for _ in range(5)]
    v = GenieResultValidator()
    r = v.validate(_gr(cols, rows))
    assert r.passed is False
    assert any(viol.startswith("column_count") for viol in r.violations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
