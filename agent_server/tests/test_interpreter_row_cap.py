"""Unit test for Interpreter._enforce_llm_row_budget."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from agents.campaign_insight.contracts import (  # noqa: E402
    StepResult,
    StepStatus,
    TableSummary,
)
from agents.campaign_insight.interpreter import Interpreter  # noqa: E402


def _make_step(step_id: int, row_count: int) -> StepResult:
    return StepResult(
        step_id=step_id,
        dimension="campaign",
        status=StepStatus.SUCCESS,
        table_summary=TableSummary(
            mode="full",
            row_count=row_count,
            full_data=[[i, 0.1] for i in range(row_count)],
        ),
    )


def test_no_trim_when_under_cap():
    step_results = {1: _make_step(1, 20), 2: _make_step(2, 10)}
    Interpreter._enforce_llm_row_budget(step_results, total_cap=50)
    assert len(step_results[1].table_summary.full_data) == 20
    assert len(step_results[2].table_summary.full_data) == 10


def test_trim_when_over_cap():
    step_results = {
        1: _make_step(1, 100),
        2: _make_step(2, 80),
        3: _make_step(3, 40),
    }
    Interpreter._enforce_llm_row_budget(step_results, total_cap=50)
    total = sum(
        len(sr.table_summary.full_data) for sr in step_results.values()
    )
    assert total <= 50
    # Each step still has at least 1 row (per_step = max(1, 50//3) = 16)
    for sr in step_results.values():
        assert len(sr.table_summary.full_data) >= 1


def test_partial_steps_skipped():
    good = _make_step(1, 100)
    bad = StepResult(step_id=2, status=StepStatus.ERROR, dimension="audience")
    step_results = {1: good, 2: bad}
    Interpreter._enforce_llm_row_budget(step_results, total_cap=50)
    assert len(good.table_summary.full_data) <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
