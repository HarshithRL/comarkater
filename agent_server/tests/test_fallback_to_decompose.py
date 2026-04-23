"""Unit test: `_all_contract_violations` correctly triggers the fallback."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

# agent.py pulls core.tracing which imports mlflow; skip cleanly if missing.
pytest.importorskip("mlflow")

from agents.campaign_insight.agent import CampaignInsightAgent  # noqa: E402
from agents.campaign_insight.contracts import (  # noqa: E402
    StepResult,
    StepStatus,
)


def _partial_with_violations(step_id: int = 1) -> StepResult:
    return StepResult(
        step_id=step_id,
        dimension="campaign",
        status=StepStatus.PARTIAL,
        validation_violations=["row_count=214 exceeds cap 50"],
    )


def _success() -> StepResult:
    return StepResult(
        step_id=1,
        dimension="campaign",
        status=StepStatus.SUCCESS,
    )


def test_all_partial_with_violations_triggers_fallback():
    results = {1: _partial_with_violations(1), 2: _partial_with_violations(2)}
    assert CampaignInsightAgent._all_contract_violations(results) is True


def test_one_success_blocks_fallback():
    results = {1: _success(), 2: _partial_with_violations(2)}
    assert CampaignInsightAgent._all_contract_violations(results) is False


def test_empty_results_no_fallback():
    assert CampaignInsightAgent._all_contract_violations({}) is False


def test_partial_without_violations_no_fallback():
    sr = StepResult(
        step_id=1,
        dimension="campaign",
        status=StepStatus.PARTIAL,
        validation_violations=[],
    )
    assert CampaignInsightAgent._all_contract_violations({1: sr}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
