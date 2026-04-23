"""Unit tests for the dev-eval judge wrappers and runner aggregation.

The LLM call itself is stubbed; we verify that each judge:

1. Formats a prompt containing the fields it claims to score.
2. Routes the structured output back through :class:`JudgeVerdict`.
3. The runner's ``aggregate`` math is correct.
4. Non-data intents short-circuit the rewrite/sql/summary judges so we
   don't waste LLM calls (and don't pollute aggregate pass-rates).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.judges import (  # noqa: E402
    IntentJudge,
    JudgeVerdict,
    RewriteFidelityJudge,
    SqlCorrectnessJudge,
    SummaryGroundednessJudge,
)
from evals.runner import (  # noqa: E402
    CaseResult,
    _collect_fields,
    _judge_case,
    aggregate,
)


def _make_llm(verdict: JudgeVerdict) -> tuple[MagicMock, MagicMock]:
    """Build a stub LLM that returns ``verdict`` for every invoke call."""
    structured = MagicMock()
    structured.invoke.return_value = verdict
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


def test_intent_judge_passes_query_and_intent_to_llm():
    v = JudgeVerdict(score=5, passed=True, reasoning="ok")
    llm, structured = _make_llm(v)
    out = IntentJudge(llm).judge(query="top campaigns", intent="ranking")
    assert out == v
    prompt = structured.invoke.call_args.args[0]
    assert "top campaigns" in prompt
    assert "ranking" in prompt


def test_rewrite_judge_includes_both_versions():
    v = JudgeVerdict(score=4, passed=True, reasoning="minor paraphrase")
    llm, structured = _make_llm(v)
    RewriteFidelityJudge(llm).judge(original="who opened?",
                                    rewritten="who opened in last 30d?")
    prompt = structured.invoke.call_args.args[0]
    assert "who opened?" in prompt
    assert "last 30d" in prompt


def test_sql_judge_surfaces_sql_and_client_context():
    v = JudgeVerdict(score=3, passed=True, reasoning="ok")
    llm, structured = _make_llm(v)
    SqlCorrectnessJudge(llm).judge(rewritten="top 5", sql="SELECT 1",
                                   client_id="72994", client_name="IGP")
    prompt = structured.invoke.call_args.args[0]
    assert "SELECT 1" in prompt
    assert "72994" in prompt
    assert "IGP" in prompt


def test_summary_judge_includes_table_and_summary():
    v = JudgeVerdict(score=2, passed=False, reasoning="invented",
                     issues=["INVENTED: 99%"])
    llm, structured = _make_llm(v)
    SummaryGroundednessJudge(llm).judge(
        original="how are we doing?",
        table="| metric | value |",
        summary="CTR was 99%",
    )
    prompt = structured.invoke.call_args.args[0]
    assert "metric" in prompt and "CTR was 99%" in prompt


def test_judge_error_returns_failing_verdict():
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("boom")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    v = IntentJudge(llm).judge(query="x", intent="y")
    assert not v.passed and v.score == 0 and "judge_error" in v.reasoning


def test_aggregate_computes_means_and_pass_rates():
    r1 = CaseResult(case_id="a", original_question="", client_id="",
                    client_name="", intent="ranking", rewritten_question="",
                    genie_sql="", genie_summary="", genie_trace_id="")
    r1.verdicts["intent"] = JudgeVerdict(score=5, passed=True, reasoning="")
    r1.verdicts["rewrite"] = JudgeVerdict(score=4, passed=True, reasoning="")
    r2 = CaseResult(case_id="b", original_question="", client_id="",
                    client_name="", intent="ranking", rewritten_question="",
                    genie_sql="", genie_summary="", genie_trace_id="")
    r2.verdicts["intent"] = JudgeVerdict(score=1, passed=False, reasoning="")
    r2.verdicts["rewrite"] = JudgeVerdict(score=3, passed=True, reasoning="")
    agg = aggregate([r1, r2])
    assert agg["intent_mean_score"] == pytest.approx(3.0)
    assert agg["intent_pass_rate"] == pytest.approx(0.5)
    assert agg["rewrite_pass_rate"] == pytest.approx(1.0)
    assert agg["overall_pass_rate"] == pytest.approx(0.75)


def test_non_data_intents_skip_downstream_judges():
    # out_of_scope should auto-pass rewrite/sql/summary without LLM calls.
    intent_llm, intent_struct = _make_llm(
        JudgeVerdict(score=5, passed=True, reasoning="correct OOS"))
    fail_llm = MagicMock()  # if called, .with_structured_output raises
    fail_llm.with_structured_output.side_effect = AssertionError(
        "should not be called for non-data intents")

    intent_j = IntentJudge(intent_llm)
    rewrite_j = RewriteFidelityJudge.__new__(RewriteFidelityJudge)
    rewrite_j._llm = MagicMock(invoke=MagicMock(
        side_effect=AssertionError("rewrite should not run")))
    sql_j = SqlCorrectnessJudge.__new__(SqlCorrectnessJudge)
    sql_j._llm = MagicMock(invoke=MagicMock(
        side_effect=AssertionError("sql should not run")))
    summary_j = SummaryGroundednessJudge.__new__(SummaryGroundednessJudge)
    summary_j._llm = MagicMock(invoke=MagicMock(
        side_effect=AssertionError("summary should not run")))

    case = {"case_id": "t", "original_question": "weather?",
            "client_id": "", "client_name": ""}
    state = {"intent": "out_of_scope", "rewritten_question": "",
             "genie_sql": "", "genie_summary": "", "genie_trace_id": ""}
    result = _collect_fields(case, state)
    _judge_case(result, state, intent_j=intent_j, rewrite_j=rewrite_j,
                sql_j=sql_j, summary_j=summary_j)

    assert result.verdicts["rewrite"].reasoning == "not_applicable"
    assert result.verdicts["sql"].reasoning == "not_applicable"
    assert result.verdicts["summary"].reasoning == "not_applicable"
    assert intent_struct.invoke.called
