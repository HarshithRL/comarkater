"""Unit test: validator rejection drives ReAct refinement, then succeeds."""
from __future__ import annotations

import asyncio
import sys

import pytest

sys.path.insert(0, ".")

# executor.py pulls core.tracing which imports mlflow; skip cleanly if missing.
pytest.importorskip("mlflow")

from agents.campaign_insight.contracts import (  # noqa: E402
    ExecutionPlan,
    GenieResponse,
    PlanStep,
    StepStatus,
)
from agents.campaign_insight.executor import ReActExecutor  # noqa: E402
from agents.campaign_insight.genie_validator import (  # noqa: E402
    GenieResultValidator,
)


class _FakeToolHandler:
    """Returns a large result on the first call, small on the second."""

    def __init__(self) -> None:
        self.calls = 0

    async def execute_query_with_retry(self, nl_query: str) -> GenieResponse:
        self.calls += 1
        if self.calls == 1:
            return GenieResponse(
                columns=[{"name": "campaign"}, {"name": "ctr"}],
                data_array=[[f"c-{i}", 0.01] for i in range(200)],
                row_count=200,
                sql="SELECT ...",
                status="success",
            )
        return GenieResponse(
            columns=[{"name": "segment"}, {"name": "avg_ctr"}],
            data_array=[[f"s-{i}", 0.02 * i] for i in range(10)],
            row_count=10,
            sql="SELECT seg, avg(ctr) FROM t GROUP BY seg",
            status="success",
        )


class _FakeAnalyzer:
    def analyze(self, columns, data_array, row_count):
        from agents.campaign_insight.contracts import TableSummary
        return TableSummary(
            mode="full",
            row_count=row_count,
            schema=columns,
            full_data=list(data_array),
            aggregates={"avg_ctr": 0.08},
        )


class _FakeBuilder:
    def build_display_table(self, columns, data_array, title, sql):
        from agents.campaign_insight.contracts import DisplayTable
        return DisplayTable(
            title=title,
            columns=[str(c.get("name", "")) for c in columns],
            rows=list(data_array),
            source_sql=sql,
        )


class _FakeLLM:
    """Returns a static refined NL query for _reason_next_query."""

    def invoke(self, messages):
        class _R:
            content = "Please aggregate by segment."
        return _R()


class _FakeDomain:
    def get_minimum_volume_thresholds(self):
        return {}

    def format_for_subagent(self):
        return ""


def test_validator_rejection_triggers_refine_then_success():
    tool = _FakeToolHandler()
    executor = ReActExecutor(
        llm=_FakeLLM(),
        tool_handler=tool,
        table_analyzer=_FakeAnalyzer(),
        table_builder=_FakeBuilder(),
        domain_knowledge=_FakeDomain(),
        max_iterations_per_step=2,
        validator=GenieResultValidator(),
    )
    plan = ExecutionPlan(
        steps=[PlanStep(step_id=1, dimension="campaign", query="compare by segment")],
        total_budget=1,
    )
    results = asyncio.run(executor.execute_plan(plan))
    sr = results[1]
    assert tool.calls == 2, "validator rejection should force exactly one refine"
    assert sr.status is StepStatus.SUCCESS
    assert sr.contract_satisfied is True
    assert sr.validation_violations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
