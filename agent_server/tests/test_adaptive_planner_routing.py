"""Unit test: routing forces single-step plans without mutating the query."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from agents.campaign_insight.adaptive_planner import AdaptivePlanner  # noqa: E402
from agents.campaign_insight.contracts import (  # noqa: E402
    DimensionClassification,
    DimensionConfig,
    DimensionRole,
)
from agents.campaign_insight.query_router import (  # noqa: E402
    RoutingDecision,
    RoutingStrategy,
)


class _FakeStructured:
    def __init__(self, steps_count: int) -> None:
        self.steps_count = steps_count
        self.last_user_prompt = ""

    def invoke(self, messages):
        self.last_user_prompt = messages[-1]["content"]

        class _Schema:
            def __init__(self, n):
                class _S:
                    step_id = 1
                    dimension = "campaign"
                    query = "planner-generated question"
                    purpose = "do the thing"
                    depends_on: list[int] = []
                    can_parallel = True

                self.steps = [_S() for _ in range(n)]

        return _Schema(self.steps_count)


class _FakeLLM:
    def __init__(self, steps_count: int) -> None:
        self.structured = _FakeStructured(steps_count)

    def with_structured_output(self, schema):
        return self.structured


class _FakeDomain:
    def get_minimum_volume_thresholds(self):
        return {}


def _multi_budget_dim_config() -> DimensionClassification:
    return DimensionClassification(
        primary_analysis="campaign",
        channel="email",
        campaign=DimensionConfig(role=DimensionRole.PRIMARY, budget=2),
        audience=DimensionConfig(role=DimensionRole.SUPPORTING, budget=1),
        content=DimensionConfig(role=DimensionRole.NONE, budget=0),
    )


def test_genie_direct_forces_single_step_without_mutating_query():
    llm = _FakeLLM(steps_count=3)
    planner = AdaptivePlanner(llm=llm, domain_knowledge=_FakeDomain())
    routing = RoutingDecision(
        strategy=RoutingStrategy.GENIE_DIRECT,
        reason="rules",
        confidence=0.9,
    )
    user_query = "Compare top campaigns by CTR"
    plan = planner.plan(
        user_query, {}, _multi_budget_dim_config(), routing=routing
    )
    assert len(plan.steps) == 1
    # The user's query MUST NOT appear mutated inside the prompt — the planner
    # relies on the _SYSTEM_PROMPT aggregation contract, not string concat.
    assert user_query in llm.structured.last_user_prompt


def test_hybrid_forces_single_step():
    llm = _FakeLLM(steps_count=3)
    planner = AdaptivePlanner(llm=llm, domain_knowledge=_FakeDomain())
    routing = RoutingDecision(strategy=RoutingStrategy.HYBRID, confidence=0.7)
    plan = planner.plan("how did email perform", {}, _multi_budget_dim_config(), routing=routing)
    assert len(plan.steps) == 1


def test_agent_decompose_allows_multiple_steps():
    llm = _FakeLLM(steps_count=3)
    planner = AdaptivePlanner(llm=llm, domain_knowledge=_FakeDomain())
    routing = RoutingDecision(strategy=RoutingStrategy.AGENT_DECOMPOSE, confidence=0.9)
    plan = planner.plan("why did conv drop", {}, _multi_budget_dim_config(), routing=routing)
    # total_budget = 3, _MAX_STEPS_HARD = 4 → allows up to 3 (capped by budget)
    assert len(plan.steps) >= 2


def test_no_routing_preserves_legacy_behavior():
    llm = _FakeLLM(steps_count=3)
    planner = AdaptivePlanner(llm=llm, domain_knowledge=_FakeDomain())
    plan = planner.plan("anything", {}, _multi_budget_dim_config(), routing=None)
    assert len(plan.steps) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
