"""Streaming contract tests: executor must guarantee a table_ready event
for every SUCCESS / PARTIAL step, and must skip the emission for ERROR /
TIMEOUT.

These tests cover the refactor that made streaming a first-class execution
primitive:
  * non-destructive gate → late tables are no longer dropped (covered by
    not-dropping observable state: the emission contract guarantees the
    fast path always produces a table, so the gate's drop behaviour is
    structurally unreachable for the primary path)
  * placeholder tables for empty result sets keep the Data section visible
  * ERROR / TIMEOUT steps don't pollute the Data section with empty grids
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, ".")

# executor.py pulls core.tracing which imports mlflow; skip cleanly if missing.
pytest.importorskip("mlflow")

from agents.campaign_insight.contracts import (  # noqa: E402
    DisplayTable,
    PlanStep,
    StepResult,
    StepStatus,
    TableSummary,
)
from agents.campaign_insight.executor import ReActExecutor  # noqa: E402


def _make_executor_shell() -> ReActExecutor:
    """Build a ReActExecutor without exercising its async machinery.

    The _emit method we're testing only reads sr.status / sr.display_table
    and writes to stream_callback, so none of the injected collaborators
    are invoked.
    """
    return ReActExecutor.__new__(ReActExecutor)


def _capture_callback() -> tuple[list[dict], Any]:
    emitted: list[dict] = []

    def cb(event: dict) -> None:
        emitted.append(event)

    return emitted, cb


def _step(step_id: int = 1) -> PlanStep:
    return PlanStep(step_id=step_id, dimension="campaign", query="q")


def _success_result(step_id: int = 1, *, rows: bool = True) -> StepResult:
    if rows:
        display = DisplayTable(
            title="Top campaigns",
            columns=["campaign", "ctr"],
            rows=[["c-1", 0.05], ["c-2", 0.07]],
            source_sql="",
        )
        summary = TableSummary(
            mode="full",
            row_count=2,
            schema=[{"name": "campaign"}, {"name": "ctr"}],
            full_data=[["c-1", 0.05], ["c-2", 0.07]],
            aggregates={},
        )
    else:
        display = DisplayTable(
            title="", columns=[], rows=[], source_sql=""
        )
        summary = TableSummary(
            mode="full",
            row_count=0,
            schema=[],
            full_data=[],
            aggregates={},
        )
    return StepResult(
        step_id=step_id,
        dimension="campaign",
        status=StepStatus.SUCCESS,
        display_table=display,
        table_summary=summary,
    )


# ── SUCCESS with rows → one step_completed + one populated table_ready ──


def test_success_with_rows_emits_table_ready():
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    executor._emit(cb, _step(), _success_result(rows=True))

    event_types = [e["event_type"] for e in emitted]
    assert event_types == ["step_completed", "table_ready"]

    table_evt = emitted[1]
    assert table_evt["item_id"] == "table_step_1"
    assert table_evt["step_id"] == 1
    assert table_evt["placeholder"] is False
    assert table_evt["table"]["tableHeaders"] == ["campaign", "ctr"]
    assert table_evt["table"]["data"] == [["c-1", 0.05], ["c-2", 0.07]]


# ── SUCCESS with empty result → placeholder table_ready (still streams) ──


def test_success_with_empty_display_table_emits_placeholder():
    """Guarantees the fast path still produces a table event when Genie
    returned zero rows. Before the refactor this case was silently
    dropped and the table would only surface via supervisor_synthesize at
    the end — which the gate then blocked when analysis had already
    emitted."""
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    executor._emit(cb, _step(), _success_result(rows=False))

    event_types = [e["event_type"] for e in emitted]
    assert event_types == ["step_completed", "table_ready"]

    table_evt = emitted[1]
    assert table_evt["item_id"] == "table_step_1"
    assert table_evt["placeholder"] is True
    assert table_evt["table"]["tableHeaders"] == ["Status"]
    assert table_evt["table"]["data"] == [["No rows returned for this step."]]


# ── SUCCESS with display_table=None → placeholder (defensive) ──


def test_success_with_none_display_table_emits_placeholder():
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    sr = StepResult(
        step_id=2,
        dimension="campaign",
        status=StepStatus.SUCCESS,
        display_table=None,
        table_summary=None,
    )
    executor._emit(cb, _step(step_id=2), sr)

    event_types = [e["event_type"] for e in emitted]
    assert event_types == ["step_completed", "table_ready"]
    assert emitted[1]["placeholder"] is True


# ── PARTIAL → always emits table_ready (same contract as SUCCESS) ──


def test_partial_with_rows_emits_table_ready():
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    sr = _success_result(rows=True)
    sr.status = StepStatus.PARTIAL
    executor._emit(cb, _step(), sr)

    event_types = [e["event_type"] for e in emitted]
    assert event_types == ["step_completed", "table_ready"]
    assert emitted[1]["placeholder"] is False


# ── ERROR → step_completed only, no table_ready ──


def test_error_emits_only_step_completed():
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    sr = StepResult(
        step_id=3,
        dimension="campaign",
        status=StepStatus.ERROR,
        error_message="synthetic",
    )
    executor._emit(cb, _step(step_id=3), sr)

    event_types = [e["event_type"] for e in emitted]
    assert event_types == ["step_completed"]
    assert emitted[0]["status"] == "error"


# ── TIMEOUT → step_completed only, no table_ready ──


def test_timeout_emits_only_step_completed():
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    sr = StepResult(
        step_id=4,
        dimension="campaign",
        status=StepStatus.TIMEOUT,
        error_message="timed out",
    )
    executor._emit(cb, _step(step_id=4), sr)

    event_types = [e["event_type"] for e in emitted]
    assert event_types == ["step_completed"]
    assert emitted[0]["status"] == "timeout"


# ── stream_callback=None is tolerated (no emission, no crash) ──


def test_none_stream_callback_is_noop():
    executor = _make_executor_shell()
    executor._emit(None, _step(), _success_result(rows=True))  # must not raise


# ── Multi-step: each step produces its own table_ready with stable item_id ──


def test_multi_step_produces_per_step_table_ready():
    """Multiple tables stream incrementally, each keyed by step_id for
    downstream deduplication."""
    executor = _make_executor_shell()
    emitted, cb = _capture_callback()

    for sid in (1, 2, 3):
        executor._emit(cb, _step(step_id=sid), _success_result(step_id=sid, rows=True))

    tables = [e for e in emitted if e["event_type"] == "table_ready"]
    assert [t["step_id"] for t in tables] == [1, 2, 3]
    assert [t["item_id"] for t in tables] == [
        "table_step_1",
        "table_step_2",
        "table_step_3",
    ]
