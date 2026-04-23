"""Proof test: parallel fan-out over ready steps beats the current sequential executor.

This test does NOT change production code. It compares:

  1. ``ReActExecutor.execute_plan``                       — current prod, sequential
  2. ``_run_parallel_variant`` (below) that calls
     ``ReActExecutor._execute_single_step`` on ready steps
     via ``asyncio.gather``                               — hypothetical parallel

If the parallel driver finishes in ~max(step_latency) while the sequential
executor takes ~sum(step_latency), and if dependency + error semantics hold,
refactoring ``execute_plan`` to use ``asyncio.gather`` over
``plan.get_ready_steps(...)`` is safe to pursue.
"""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

sys.path.insert(0, ".")

# executor.py pulls core.tracing which imports mlflow; skip cleanly if missing.
pytest.importorskip("mlflow")

from agents.campaign_insight.contracts import (  # noqa: E402
    DisplayTable,
    ExecutionPlan,
    GenieResponse,
    PlanStep,
    StepResult,
    StepStatus,
    TableSummary,
)
from agents.campaign_insight.executor import ReActExecutor  # noqa: E402


# Per-step mocked latency. Keep small so the test suite stays fast, but large
# enough that sequential-vs-parallel timings are clearly distinguishable.
_STEP_LATENCY_S = 0.3


# ── Fakes ───────────────────────────────────────────────────────────────────


class _SlowToolHandler:
    """Sleeps for ``latency_s`` then returns a canned success."""

    def __init__(self, latency_s: float = _STEP_LATENCY_S) -> None:
        self.latency_s = latency_s
        self.call_count = 0

    async def execute_query_with_retry(self, nl_query: str) -> GenieResponse:
        self.call_count += 1
        await asyncio.sleep(self.latency_s)
        return GenieResponse(
            columns=[{"name": "campaign"}, {"name": "ctr"}],
            data_array=[["c-1", 0.05], ["c-2", 0.07]],
            row_count=2,
            sql="SELECT c, ctr FROM t",
            status="success",
        )


class _SelectivelyFailingToolHandler:
    """Succeeds for most queries; raises when the NL query contains a marker.

    Used to confirm that a failure in one concurrent step does not cancel
    sibling steps when we drive ``asyncio.gather`` ourselves.
    """

    def __init__(self, fail_marker: str, latency_s: float = 0.1) -> None:
        self.fail_marker = fail_marker
        self.latency_s = latency_s
        self.call_count = 0

    async def execute_query_with_retry(self, nl_query: str) -> GenieResponse:
        self.call_count += 1
        await asyncio.sleep(self.latency_s)
        if self.fail_marker in nl_query:
            raise RuntimeError("synthetic tool failure")
        return GenieResponse(
            columns=[{"name": "c"}, {"name": "m"}],
            data_array=[["a", 1]],
            row_count=1,
            sql="",
            status="success",
        )


class _FakeAnalyzer:
    def analyze(self, columns, data_array, row_count):
        return TableSummary(
            mode="full",
            row_count=row_count,
            schema=columns,
            full_data=list(data_array),
            aggregates={"m": 1.0},
        )


class _FakeBuilder:
    def build_display_table(self, columns, data_array, title, sql):
        return DisplayTable(
            title=title,
            columns=[str(c.get("name", "")) for c in columns],
            rows=list(data_array),
            source_sql=sql,
        )


class _FakeLLM:
    def invoke(self, messages):
        class _R:
            content = "refined"
        return _R()


class _FakeDomain:
    def get_minimum_volume_thresholds(self):
        return {}

    def format_for_subagent(self):
        return ""


def _make_executor(tool_handler) -> ReActExecutor:
    return ReActExecutor(
        llm=_FakeLLM(),
        tool_handler=tool_handler,
        table_analyzer=_FakeAnalyzer(),
        table_builder=_FakeBuilder(),
        domain_knowledge=_FakeDomain(),
        max_iterations_per_step=1,
        total_timeout_seconds=30,
        step_timeout_seconds=5,
        validator=None,
    )


# ── Parallel variant under test ─────────────────────────────────────────────


async def _run_parallel_variant(
    executor: ReActExecutor, plan: ExecutionPlan
) -> dict[int, StepResult]:
    """Run ``plan`` with ``asyncio.gather`` over each wave of ready steps.

    Mirrors ``ReActExecutor.execute_plan`` except that ready steps are awaited
    concurrently instead of in a ``for`` loop.
    """
    completed: set[int] = set()
    results: dict[int, StepResult] = {}

    while True:
        ready = plan.get_ready_steps(completed)
        if not ready:
            break

        coros = [
            executor._execute_single_step(step, results, "")
            for step in ready
        ]
        step_results = await asyncio.gather(*coros, return_exceptions=True)

        for step, sr in zip(ready, step_results):
            if isinstance(sr, Exception):
                sr = StepResult(
                    step_id=step.step_id,
                    dimension=step.dimension,
                    status=StepStatus.ERROR,
                    error_message=f"{type(sr).__name__}: {sr}",
                )
            results[step.step_id] = sr
            completed.add(step.step_id)

    return results


# ── Plan fixtures ───────────────────────────────────────────────────────────


def _three_independent_steps() -> ExecutionPlan:
    return ExecutionPlan(
        steps=[
            PlanStep(step_id=1, dimension="campaign", query="q1", can_parallel=True),
            PlanStep(step_id=2, dimension="audience", query="q2", can_parallel=True),
            PlanStep(step_id=3, dimension="content",  query="q3", can_parallel=True),
        ],
        total_budget=3,
    )


def _two_with_dependency() -> ExecutionPlan:
    # step 2 depends on step 1 → must serialise into two waves.
    return ExecutionPlan(
        steps=[
            PlanStep(step_id=1, dimension="campaign", query="q1", can_parallel=True),
            PlanStep(
                step_id=2,
                dimension="audience",
                query="q2 uses step 1 output",
                depends_on=[1],
                can_parallel=False,
            ),
        ],
        total_budget=2,
    )


# ── Tests ───────────────────────────────────────────────────────────────────


def test_sequential_baseline_is_sum_of_latencies():
    """Current prod: ``execute_plan`` ≈ N × step_latency for independent steps."""
    tool = _SlowToolHandler()
    executor = _make_executor(tool)
    plan = _three_independent_steps()

    t0 = time.monotonic()
    results = asyncio.run(executor.execute_plan(plan))
    elapsed = time.monotonic() - t0

    assert len(results) == 3
    assert tool.call_count == 3
    assert all(r.status is StepStatus.SUCCESS for r in results.values())
    # 3 × 0.3s = 0.9s sequentially; allow 10 % generous lower bound.
    lower = 3 * _STEP_LATENCY_S * 0.9
    assert elapsed >= lower, (
        f"Sequential executor finished too fast ({elapsed:.2f}s < {lower:.2f}s) — "
        "did the executor secretly gain parallelism?"
    )


def test_parallel_variant_is_max_of_latencies():
    """Hypothetical executor: asyncio.gather over ready steps ≈ max(step_latency)."""
    tool = _SlowToolHandler()
    executor = _make_executor(tool)
    plan = _three_independent_steps()

    t0 = time.monotonic()
    results = asyncio.run(_run_parallel_variant(executor, plan))
    elapsed = time.monotonic() - t0

    assert len(results) == 3
    assert tool.call_count == 3
    assert all(r.status is StepStatus.SUCCESS for r in results.values())
    # Parallel should finish near one step-latency; allow up to 1.8× slack for
    # task scheduling + mlflow tracing overhead.
    upper = _STEP_LATENCY_S * 1.8
    assert elapsed < upper, (
        f"Parallel variant too slow ({elapsed:.2f}s ≥ {upper:.2f}s) — fan-out not happening?"
    )


def test_parallel_respects_dependencies():
    """A step with ``depends_on`` still waits for its parent to finish."""
    tool = _SlowToolHandler()
    executor = _make_executor(tool)
    plan = _two_with_dependency()

    t0 = time.monotonic()
    results = asyncio.run(_run_parallel_variant(executor, plan))
    elapsed = time.monotonic() - t0

    assert len(results) == 2
    assert tool.call_count == 2
    assert all(r.status is StepStatus.SUCCESS for r in results.values())
    # Two serialised waves ⇒ ≥ 2 × step_latency.
    lower = 2 * _STEP_LATENCY_S * 0.9
    assert elapsed >= lower, (
        f"Dependent step ran in parallel with its parent ({elapsed:.2f}s < {lower:.2f}s)"
    )


def test_parallel_one_failure_does_not_abort_siblings():
    """An exception in one concurrent step yields ERROR; siblings still SUCCESS.

    ``_execute_single_step`` catches tool-handler exceptions internally and
    returns a ``StepResult(status=ERROR)`` instead of propagating. So
    ``asyncio.gather`` never actually sees an exception — it returns 3 clean
    StepResults, one of which carries ERROR status.
    """
    tool = _SelectivelyFailingToolHandler(fail_marker="q2")
    executor = _make_executor(tool)
    plan = _three_independent_steps()

    results = asyncio.run(_run_parallel_variant(executor, plan))

    assert len(results) == 3
    assert tool.call_count == 3
    statuses = {sid: sr.status for sid, sr in results.items()}
    assert statuses[1] is StepStatus.SUCCESS
    assert statuses[2] is StepStatus.ERROR
    assert statuses[3] is StepStatus.SUCCESS
    assert "synthetic tool failure" in (results[2].error_message or "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
