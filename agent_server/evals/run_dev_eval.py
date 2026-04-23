"""CLI: run the dev evaluation and log every case to MLflow.

Usage
-----

Replay (no agent invocation — needs a fixture with captured state)::

    python -m evals.run_dev_eval \\
        --mode replay \\
        --cases agent_server/evals/datasets/dev_cases.jsonl \\
        --replay agent_server/evals/datasets/dev_state.jsonl

Live (invokes the agent via a wired-in callback)::

    python -m evals.run_dev_eval --mode live

Env vars:
    MLFLOW_EXPERIMENT_NAME   default "comarketer-dev-eval"
    EVAL_JUDGE_MODEL         default settings.LLM_ENDPOINT_NAME
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import settings  # noqa: E402
from evals.runner import (  # noqa: E402
    aggregate,
    load_cases,
    load_replay,
    run_eval,
)

logger = logging.getLogger("evals.run_dev_eval")


def _build_judge_llm() -> Any:
    """Build the chat model the judges call. Mirrors the production path."""
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    sp_token = os.environ.get("DATABRICKS_TOKEN") or os.environ.get("SP_TOKEN")
    if not sp_token:
        raise SystemExit(
            "DATABRICKS_TOKEN / SP_TOKEN must be set so the judge can call "
            "the AI Gateway."
        )
    model = os.environ.get("EVAL_JUDGE_MODEL", settings.LLM_ENDPOINT_NAME)
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(sp_token),
        base_url=settings.AI_GATEWAY_URL,
        temperature=0.0,
    )


def _live_fn(_case: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the live graph. Populate this with your preferred entry path.

    The default raises so a half-wired live run fails loudly rather than
    silently scoring empty fields.
    """
    raise NotImplementedError(
        "live mode requires wiring into CoMarketerAgent.predict(); override "
        "_live_fn or run with --mode replay."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live", "replay"], default="replay")
    parser.add_argument("--cases", type=Path,
                        default=Path(__file__).parent / "datasets" / "dev_cases.jsonl")
    parser.add_argument("--replay", type=Path,
                        default=Path(__file__).parent / "datasets" / "dev_state.jsonl")
    parser.add_argument("--experiment", default=None,
                        help="MLflow experiment name (env MLFLOW_EXPERIMENT_NAME overrides)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="skip MLflow logging (useful for local smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cases = load_cases(args.cases)
    replay_records = load_replay(args.replay) if args.mode == "replay" else None
    judge_llm = _build_judge_llm()

    results = run_eval(
        cases=cases,
        judge_llm=judge_llm,
        mode=args.mode,
        live_fn=_live_fn if args.mode == "live" else None,
        replay_records=replay_records,
    )

    agg = aggregate(results)

    # Console report
    print("\n=== Dev Eval Results ===")
    for r in results:
        print(f"\n[{r.case_id}] {r.original_question}")
        print(f"  intent={r.intent!r}  trace_id={r.genie_trace_id!r}")
        for name, v in r.verdicts.items():
            flag = "PASS" if v.passed else "FAIL"
            issues = f" issues={v.issues}" if v.issues else ""
            print(f"    {name:<8} {flag} score={v.score} — {v.reasoning[:110]}{issues}")
    print("\n=== Aggregate ===")
    for k, v in sorted(agg.items()):
        print(f"  {k:<28} {v:.3f}")

    if args.no_mlflow:
        return 0

    import mlflow
    exp = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", "comarketer-dev-eval")
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=f"dev-eval-{args.mode}"):
        mlflow.log_params({
            "mode": args.mode,
            "n_cases": len(results),
            "judge_model": os.environ.get("EVAL_JUDGE_MODEL", settings.LLM_ENDPOINT_NAME),
        })
        for k, v in agg.items():
            mlflow.log_metric(k, v)
        for r in results:
            for m, v in r.flat_metrics().items():
                mlflow.log_metric(f"case_{r.case_id}__{m}", v)
        dump = [
            {
                "case_id": r.case_id,
                "original_question": r.original_question,
                "intent": r.intent,
                "rewritten_question": r.rewritten_question,
                "genie_sql": r.genie_sql,
                "genie_summary": r.genie_summary,
                "genie_trace_id": r.genie_trace_id,
                "verdicts": {k: v.model_dump() for k, v in r.verdicts.items()},
            }
            for r in results
        ]
        artifact_path = Path("dev_eval_results.json")
        artifact_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
