"""Unit tests for the rules-first QueryRouter."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from agents.campaign_insight.contracts import (  # noqa: E402
    DimensionClassification,
    DimensionConfig,
    DimensionRole,
)
from agents.campaign_insight.query_router import (  # noqa: E402
    QueryRouter,
    RoutingStrategy,
)


def _single_dim() -> DimensionClassification:
    return DimensionClassification(
        primary_analysis="campaign",
        channel="email",
        campaign=DimensionConfig(role=DimensionRole.PRIMARY, budget=2),
        audience=DimensionConfig(role=DimensionRole.NONE, budget=0),
        content=DimensionConfig(role=DimensionRole.NONE, budget=0),
    )


def _multi_dim() -> DimensionClassification:
    return DimensionClassification(
        primary_analysis="campaign",
        channel="email",
        campaign=DimensionConfig(role=DimensionRole.PRIMARY, budget=2),
        audience=DimensionConfig(role=DimensionRole.SUPPORTING, budget=1),
        content=DimensionConfig(role=DimensionRole.NONE, budget=0),
    )


def test_direct_keywords_route_to_genie_direct():
    r = QueryRouter(llm=None, domain_knowledge=None, llm_tiebreaker_enabled=False)
    d = r.classify("compare top campaigns by CTR", {}, _single_dim())
    assert d.strategy is RoutingStrategy.GENIE_DIRECT
    assert d.source == "rules"
    assert d.confidence > 0.6


def test_why_keyword_routes_to_agent_decompose():
    r = QueryRouter(llm=None, domain_knowledge=None, llm_tiebreaker_enabled=False)
    d = r.classify("why did open rate drop last week", {}, _single_dim())
    assert d.strategy is RoutingStrategy.AGENT_DECOMPOSE
    assert d.source == "rules"


def test_hybrid_keyword_routes_to_hybrid():
    r = QueryRouter(llm=None, domain_knowledge=None, llm_tiebreaker_enabled=False)
    d = r.classify("summary of last month", {}, _single_dim())
    assert d.strategy is RoutingStrategy.HYBRID


def test_multi_dimension_forces_decompose_without_keywords():
    r = QueryRouter(llm=None, domain_knowledge=None, llm_tiebreaker_enabled=False)
    d = r.classify("give me the numbers", {}, _multi_dim())
    assert d.strategy is RoutingStrategy.AGENT_DECOMPOSE
    assert "dimension" in d.reason


def test_llm_tiebreaker_invoked_when_rules_ambiguous():
    class _FakeStructured:
        def invoke(self, messages):
            class _R:
                strategy = "genie_direct"
                reason = "forced by llm"
            return _R()

    class _FakeLLM:
        def with_structured_output(self, schema):
            return _FakeStructured()

    r = QueryRouter(
        llm=_FakeLLM(),
        domain_knowledge=None,
        llm_tiebreaker_enabled=True,
        confidence_threshold=0.9,
    )
    d = r.classify("give me the numbers", {}, _single_dim())
    assert d.source == "llm"
    assert d.strategy is RoutingStrategy.GENIE_DIRECT


def test_rules_only_returns_best_guess_even_when_low_confidence():
    r = QueryRouter(llm=None, domain_knowledge=None, llm_tiebreaker_enabled=False)
    d = r.classify("give me the numbers", {}, _single_dim())
    assert d.source == "rules"
    assert d.strategy in {
        RoutingStrategy.GENIE_DIRECT,
        RoutingStrategy.HYBRID,
        RoutingStrategy.AGENT_DECOMPOSE,
    }


def test_llm_tiebreaker_failure_falls_back_to_rules():
    class _BrokenLLM:
        def with_structured_output(self, schema):
            raise RuntimeError("boom")

    r = QueryRouter(
        llm=_BrokenLLM(),
        domain_knowledge=None,
        llm_tiebreaker_enabled=True,
        confidence_threshold=0.9,
    )
    d = r.classify("give me the numbers", {}, _single_dim())
    assert d.source == "rules"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
