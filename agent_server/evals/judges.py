"""LLM-as-judge rubrics for four CoMarketer fields.

Each judge returns a :class:`JudgeVerdict` Pydantic model so that results are
schema-validated, MLflow-logged, and comparable across runs. The rubrics are
deliberately narrow: one rubric per field, one LLM call per verdict. Prompts
are terse so latency stays low.

Fields scored:

- ``intent``              — :class:`IntentJudge`
- ``rewritten_question``  — :class:`RewriteFidelityJudge`
- ``genie_sql``           — :class:`SqlCorrectnessJudge`
- ``genie_summary``       — :class:`SummaryGroundednessJudge`
"""
from __future__ import annotations

import logging
from typing import Any, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


INTENT_TAXONOMY = (
    "greeting, clarification, out_of_scope, performance_lookup, ranking, "
    "comparison, trend_analysis, diagnostic, audience_analysis, "
    "content_analysis, recommendation"
)


class JudgeVerdict(BaseModel):
    """Uniform verdict envelope returned by every judge.

    Score scale is 0-5 where 0 = unusable and 5 = ideal. ``passed`` is a
    boolean threshold (>=3) so aggregate pass-rates are meaningful even when
    individual rubrics weight dimensions differently.
    """

    score: int = Field(ge=0, le=5)
    passed: bool
    reasoning: str
    issues: List[str] = Field(default_factory=list)


_INTENT_PROMPT = """You are auditing a supervisor intent classifier.

Taxonomy: {taxonomy}

User query:
\"\"\"{query}\"\"\"

Classifier output: intent={intent}

Decide whether the classifier's choice is defensible given the query text
and the taxonomy. Score 0-5:
  5  correct and unambiguous
  4  correct but query is genuinely ambiguous
  3  defensible but a neighbouring label fits better
  2  wrong label, same family
  1  wrong family
  0  nonsensical

Set passed=true when score >= 3. List specific issues (if any)."""


_REWRITE_PROMPT = """You judge whether a rewritten analytical question is
faithful to the user's original intent AND unambiguous enough for a SQL
agent to answer without guessing.

Original: \"\"\"{original}\"\"\"
Rewritten: \"\"\"{rewritten}\"\"\"

Rules:
- The rewrite MUST preserve the user's intent, time window, channel, and
  metric mentions. Adding explicit resolutions (e.g. default time window,
  channel filter derivable from context) is allowed.
- The rewrite MUST NOT invent filters or metrics that the user did not ask
  for (e.g. a client name the user never named).
- Channel applicability: SMS/WhatsApp campaigns do not have open_rate. If
  the rewrite introduces a metric that doesn't apply to a channel
  mentioned by the user, that is an issue.

Score 0-5. passed=true when score >= 3. Enumerate concrete issues."""


_SQL_PROMPT = """You audit Genie-produced SQL against the question it was
supposed to answer.

Rewritten question: \"\"\"{rewritten}\"\"\"
Client context: client_id={client_id}, client_name={client_name}

SQL:
```sql
{sql}
```

Checklist (each failed item is an issue):
1. SELECT list covers the metrics / columns the question asks for.
2. WHERE clause encodes the time window and channel filters (if any).
3. GROUP BY / aggregation correctly reflects the requested grouping.
4. ORDER BY + LIMIT are present when the question asks for ranking/top-N.
5. No derived rate arithmetic hand-coded in the SELECT list that should
   come from the metrics taxonomy (CTR etc. should be expressed as
   SUM(clicked)/NULLIF(SUM(delivered),0), not LLM-invented math).
6. No hard-coded cid / client_id predicate — RLS is enforced upstream.
7. SQL parses as valid Databricks SQL (syntactic sanity).

Score 0-5. passed=true when score >= 3. Return every failed check as an
issue string of the form \"rule_N: <one-line explanation>\"."""


_SUMMARY_PROMPT = """You audit an analyst-facing summary for groundedness
and format compliance. Zero tolerance for invented numbers.

Original question: \"\"\"{original}\"\"\"

Data (rendered table preview, may be truncated):
{table}

Summary:
\"\"\"{summary}\"\"\"

Checklist:
1. EVERY numeric claim in the summary must be derivable from the table.
   Any unsupported number is a critical issue (prefix with \"INVENTED:\").
2. The summary addresses the original question.
3. Counts formatted with commas (2,38,100 or 238,100), rates with 2
   decimals and a percent sign (3.20%), no currency symbols, no mention
   of \"Genie\" / \"Databricks\" / internal tool names.
4. Patterns / trends cited (\"drop\", \"spike\", \"recurring\") have data
   that plausibly supports them.

Score 0-5. passed=true only when score >= 3 AND there are no INVENTED
claims. List issues explicitly."""


class BaseJudge:
    """Shared invocation wrapper. Subclasses fill in ``_build_prompt``."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm.with_structured_output(JudgeVerdict)

    def _call(self, prompt: str) -> JudgeVerdict:
        try:
            return self._llm.invoke(prompt)  # type: ignore[return-value]
        except Exception as exc:
            logger.exception("Judge LLM call failed: %s", exc)
            return JudgeVerdict(score=0, passed=False,
                                reasoning=f"judge_error: {exc}",
                                issues=["judge_invocation_failed"])


class IntentJudge(BaseJudge):
    """Rubric judge for the supervisor's ``intent`` field."""

    def judge(self, *, query: str, intent: str) -> JudgeVerdict:
        prompt = _INTENT_PROMPT.format(taxonomy=INTENT_TAXONOMY,
                                       query=query, intent=intent or "<empty>")
        return self._call(prompt)


class RewriteFidelityJudge(BaseJudge):
    """Rubric judge for ``rewritten_question`` vs ``original_question``."""

    def judge(self, *, original: str, rewritten: str) -> JudgeVerdict:
        prompt = _REWRITE_PROMPT.format(original=original,
                                        rewritten=rewritten or "<empty>")
        return self._call(prompt)


class SqlCorrectnessJudge(BaseJudge):
    """Rubric judge for ``genie_sql`` against the rewritten question."""

    def judge(self, *, rewritten: str, sql: str, client_id: str,
              client_name: str) -> JudgeVerdict:
        prompt = _SQL_PROMPT.format(rewritten=rewritten,
                                    sql=sql or "-- no sql returned --",
                                    client_id=client_id,
                                    client_name=client_name)
        return self._call(prompt)


class SummaryGroundednessJudge(BaseJudge):
    """Rubric judge for ``genie_summary`` grounded in the returned table."""

    def judge(self, *, original: str, table: str, summary: str) -> JudgeVerdict:
        prompt = _SUMMARY_PROMPT.format(original=original,
                                        table=table or "<no table>",
                                        summary=summary or "<no summary>")
        return self._call(prompt)
