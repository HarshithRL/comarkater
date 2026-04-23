"""Eval runner — pulls the five target fields and dispatches judges.

Two modes:

- ``replay``: fields are pre-captured in a JSONL fixture (one row per case
  with keys ``intent``, ``rewritten_question``, ``genie_sql``,
  ``genie_summary``, ``genie_trace_id``, ``genie_table``). No network.
  Used to iterate on judge rubrics without re-invoking the agent.

- ``live``:   a ``live_fn`` callback is supplied by the CLI. It receives a
  case dict and must return the same five fields plus ``genie_table``.
  The callback owns auth / graph invocation / streaming teardown.

Both modes share the same judging and aggregation path.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.judges import (  # noqa: E402
    IntentJudge,
    JudgeVerdict,
    RewriteFidelityJudge,
    SqlCorrectnessJudge,
    SummaryGroundednessJudge,
)

logger = logging.getLogger(__name__)


LiveFn = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class CaseResult:
    """Per-case judging output with all audit trail fields attached."""

    case_id: str
    original_question: str
    client_id: str
    client_name: str
    intent: str
    rewritten_question: str
    genie_sql: str
    genie_summary: str
    genie_trace_id: str
    verdicts: Dict[str, JudgeVerdict] = field(default_factory=dict)

    def pass_rate(self) -> float:
        """Fraction of judges that returned ``passed=True``."""
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts.values() if v.passed) / len(self.verdicts)

    def flat_metrics(self) -> Dict[str, float]:
        """MLflow-friendly flattened metrics: ``<field>_score`` + pass rate."""
        out: Dict[str, float] = {"pass_rate": self.pass_rate()}
        for name, v in self.verdicts.items():
            out[f"{name}_score"] = float(v.score)
            out[f"{name}_passed"] = 1.0 if v.passed else 0.0
        return out


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """Load dev cases from a JSONL file, skipping blank lines."""
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def load_replay(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load captured state per case_id for replay mode."""
    records: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records[row["case_id"]] = row
    return records


def _collect_fields(case: Dict[str, Any],
                    state: Dict[str, Any]) -> CaseResult:
    """Build a :class:`CaseResult` skeleton from a case + state dict."""
    return CaseResult(
        case_id=case["case_id"],
        original_question=case["original_question"],
        client_id=case.get("client_id", ""),
        client_name=case.get("client_name", ""),
        intent=str(state.get("intent", "")),
        rewritten_question=str(state.get("rewritten_question", "")),
        genie_sql=str(state.get("genie_sql", "")),
        genie_summary=str(state.get("genie_summary", "")),
        genie_trace_id=str(state.get("genie_trace_id", "")),
    )


def _judge_case(result: CaseResult, state: Dict[str, Any], *,
                intent_j: IntentJudge,
                rewrite_j: RewriteFidelityJudge,
                sql_j: SqlCorrectnessJudge,
                summary_j: SummaryGroundednessJudge) -> None:
    """Run the four judges and attach verdicts in place."""
    result.verdicts["intent"] = intent_j.judge(
        query=result.original_question, intent=result.intent,
    )

    # Skip rewrite/sql/summary judges for non-data intents — they have no
    # rewritten question or SQL by design. Record a "not_applicable" verdict
    # so aggregate pass-rates aren't polluted with zeros.
    if result.intent in {"greeting", "clarification", "out_of_scope"}:
        na = JudgeVerdict(score=5, passed=True, reasoning="not_applicable",
                          issues=[])
        result.verdicts["rewrite"] = na
        result.verdicts["sql"] = na
        result.verdicts["summary"] = na
        return

    result.verdicts["rewrite"] = rewrite_j.judge(
        original=result.original_question,
        rewritten=result.rewritten_question,
    )
    result.verdicts["sql"] = sql_j.judge(
        rewritten=result.rewritten_question,
        sql=result.genie_sql,
        client_id=result.client_id,
        client_name=result.client_name,
    )
    result.verdicts["summary"] = summary_j.judge(
        original=result.original_question,
        table=str(state.get("genie_table", "")),
        summary=result.genie_summary,
    )


def run_eval(
    *,
    cases: Iterable[Dict[str, Any]],
    judge_llm: Any,
    mode: str,
    live_fn: Optional[LiveFn] = None,
    replay_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[CaseResult]:
    """Execute judging across all cases.

    Args:
        cases: Iterable of case dicts loaded from ``dev_cases.jsonl``.
        judge_llm: LangChain chat model exposing ``with_structured_output``.
        mode: ``"live"`` or ``"replay"``.
        live_fn: Callback for live mode; must return the five fields plus
            ``genie_table``.
        replay_records: Pre-captured state keyed by ``case_id`` for replay.

    Returns:
        One :class:`CaseResult` per case in input order.
    """
    intent_j = IntentJudge(judge_llm)
    rewrite_j = RewriteFidelityJudge(judge_llm)
    sql_j = SqlCorrectnessJudge(judge_llm)
    summary_j = SummaryGroundednessJudge(judge_llm)

    results: List[CaseResult] = []
    for case in cases:
        cid = case["case_id"]
        if mode == "live":
            if live_fn is None:
                raise ValueError("live mode requires a live_fn")
            state = live_fn(case)
        elif mode == "replay":
            if replay_records is None or cid not in replay_records:
                logger.warning("No replay record for %s; skipping", cid)
                continue
            state = replay_records[cid]
        else:
            raise ValueError(f"unknown mode: {mode}")

        result = _collect_fields(case, state)
        _judge_case(result, state, intent_j=intent_j, rewrite_j=rewrite_j,
                    sql_j=sql_j, summary_j=summary_j)
        results.append(result)

    return results


def aggregate(results: List[CaseResult]) -> Dict[str, float]:
    """Mean score and pass-rate per judge across all cases."""
    if not results:
        return {}
    keys = set()
    for r in results:
        keys.update(r.verdicts.keys())
    agg: Dict[str, float] = {}
    for k in keys:
        scores = [r.verdicts[k].score for r in results if k in r.verdicts]
        passes = [r.verdicts[k].passed for r in results if k in r.verdicts]
        if scores:
            agg[f"{k}_mean_score"] = sum(scores) / len(scores)
            agg[f"{k}_pass_rate"] = sum(1 for p in passes if p) / len(passes)
    agg["overall_pass_rate"] = sum(r.pass_rate() for r in results) / len(results)
    return agg
