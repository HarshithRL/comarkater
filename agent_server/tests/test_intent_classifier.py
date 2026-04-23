"""Unit tests for IntentClassifier helpers and routing safeguards.

These tests cover the deterministic pieces only — the LLM call itself is
mocked. The goal is to lock in:

1. Query normalization strips leading punctuation noise (the original bug:
   ": Identify recurring themes..." was treated as malformed).
2. The greeting fast-path still matches what it used to.
3. The LLM-error fallback never routes to ``clarification`` — it must
   default to a data intent so we don't silently swallow real questions.
4. Meta clarification keywords pass through; new analytical phrasings
   (themes/topics/categories/patterns/highlights) do not get hardcoded
   here — they are the LLM's job, but we assert the prompt contains the
   relevant signals so the prompt can't regress.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, ".")

from supervisor.domain_context import SupervisorDomainContext  # noqa: E402
from supervisor.intent_classifier import (  # noqa: E402
    IntentClassifier,
    _PROMPT,
    _is_greeting,
    _normalize_query,
)


def _stub_domain_context() -> SupervisorDomainContext:
    """Build a MagicMock typed as SupervisorDomainContext for tests."""
    stub = MagicMock(spec=SupervisorDomainContext)
    stub.format_for_prompt.return_value = "# DOMAIN CONTEXT\n(stub)"
    return stub


# --- _normalize_query ------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (": Identify recurring themes for past 6 months",
         "Identify recurring themes for past 6 months"),
        ("- show me top campaigns", "show me top campaigns"),
        (">> what's working last month", "what's working last month"),
        ("...summarize last week", "summarize last week"),
        ("   ::  best segments  ", "best segments"),
        ("|=> compare email vs sms", "compare email vs sms"),
        ("normal question with no noise", "normal question with no noise"),
        ("", ""),
        ("   ", ""),
        ("multi   spaces    inside", "multi spaces inside"),
        # Punctuation INSIDE the query is preserved.
        ("what's the CTR?", "what's the CTR?"),
        ("compare A/B vs C/D", "compare A/B vs C/D"),
    ],
)
def test_normalize_query(raw: str, expected: str) -> None:
    assert _normalize_query(raw) == expected


# --- _is_greeting ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["hi", "Hello!", "thanks", "good morning", "hey there", "ok", "got it"],
)
def test_is_greeting_positive(text: str) -> None:
    assert _is_greeting(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "show me top campaigns",
        "Identify recurring themes",
        "what's the CTR for email last week",
        "hi can you show me opens by channel",  # too long to be a greeting
        "",
    ],
)
def test_is_greeting_negative(text: str) -> None:
    assert _is_greeting(text) is False


# --- _PROMPT regression guards --------------------------------------------


def test_prompt_includes_theme_and_pattern_signals() -> None:
    """Lock in the keywords that previously caused misrouting to clarification."""
    for keyword in ("theme", "topic", "category", "pattern", "performance_lookup"):
        assert keyword in _PROMPT, f"prompt missing '{keyword}' signal"


def test_prompt_warns_against_clarification_for_vague_analytical_queries() -> None:
    """The prompt must explicitly tell the LLM not to punt vague queries to clarification."""
    assert "DO NOT use \"clarification\"" in _PROMPT
    assert "META about the conversation" in _PROMPT or "meta about the conversation" in _PROMPT.lower()


def test_prompt_keeps_predictive_and_advisory_queries_in_scope() -> None:
    """Predictive/forecast/recommend/lookalike queries must be in-scope.

    Regression guard for: 'Based on past data, predict performance of a
    campaign with similar attributes to our best-performing campaigns'
    was misclassified as out_of_scope.
    """
    for keyword in ("predict", "forecast", "recommend", "lookalike", "what should we do"):
        assert keyword in _PROMPT, f"prompt missing in-scope signal '{keyword}'"
    assert "never mark them out_of_scope" in _PROMPT.lower()


# --- IntentClassifier.classify --------------------------------------------


def _make_llm_returning(intent: dict) -> MagicMock:
    """Build a mock LLM whose with_structured_output(...).invoke(...) returns ``intent``."""
    from supervisor.intent_classifier import _IntentModel

    structured = MagicMock()
    structured.invoke.return_value = _IntentModel(**intent)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_greeting_fast_path_skips_llm() -> None:
    llm = MagicMock()
    classifier = IntentClassifier(llm, _stub_domain_context())
    result = classifier.classify("hi")
    assert result["intent_type"] == "greeting"
    assert result["target_agent"] is None
    llm.with_structured_output.assert_not_called()


def test_normalization_runs_before_llm() -> None:
    """The leading colon must be stripped before the LLM sees the query."""
    llm = _make_llm_returning({
        "intent_type": "performance_lookup",
        "complexity": "simple",
        "channels_mentioned": [],
        "metrics_mentioned": [],
        "time_context": "past 6 months",
        "requires_audience": False,
        "requires_content": True,
    })
    classifier = IntentClassifier(llm, _stub_domain_context())
    classifier.classify(": Identify recurring themes for past 6 months")

    sent_prompt = llm.with_structured_output.return_value.invoke.call_args[0][0]
    # The "User question:" line must carry the normalized query, not the raw
    # colon-prefixed form. (The prompt itself contains "Q:" examples, so we
    # check the specific User question line, not the whole prompt.)
    user_line = next(
        (ln for ln in sent_prompt.splitlines() if ln.startswith("User question:")),
        "",
    )
    assert user_line == "User question: Identify recurring themes for past 6 months"


def test_data_intent_routes_to_campaign_insight() -> None:
    llm = _make_llm_returning({
        "intent_type": "performance_lookup",
        "complexity": "simple",
        "channels_mentioned": ["email"],
        "metrics_mentioned": ["ctr"],
        "time_context": "last 7 days",
        "requires_audience": False,
        "requires_content": False,
    })
    classifier = IntentClassifier(llm, _stub_domain_context())
    result = classifier.classify("show top email campaigns by CTR last week")
    assert result["intent_type"] == "performance_lookup"
    assert result["target_agent"] == "campaign_insight"


def test_clarification_intent_has_no_target_agent() -> None:
    llm = _make_llm_returning({
        "intent_type": "clarification",
        "complexity": "simple",
        "channels_mentioned": [],
        "metrics_mentioned": [],
        "time_context": None,
        "requires_audience": False,
        "requires_content": False,
    })
    classifier = IntentClassifier(llm, _stub_domain_context())
    result = classifier.classify("what did you say earlier?")
    assert result["intent_type"] == "clarification"
    assert result["target_agent"] is None


def test_llm_error_fallback_routes_to_data_not_clarification() -> None:
    """A failing LLM must NOT silently route real questions to clarification."""
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("gateway timeout")
    llm.with_structured_output.return_value = structured

    classifier = IntentClassifier(llm, _stub_domain_context())
    result = classifier.classify("show me theme performance for last 6 months")

    assert result["intent_type"] == "performance_lookup"
    assert result["target_agent"] == "campaign_insight"
