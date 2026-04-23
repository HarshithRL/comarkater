"""Produce Interpretation objects from TableSummary + domain context."""
from __future__ import annotations

import json
import logging
from typing import Any, cast

import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.campaign_insight.contracts import (
    Interpretation,
    PatternMatch,
    StepResult,
    StepStatus,
)
from agents.campaign_insight.prompts.interpretation_prompt import INTERPRETATION_PROMPT

logger = logging.getLogger(__name__)


class _PatternModel(BaseModel):
    name: str = ""
    description: str = ""
    likely_causes: list[str] = Field(default_factory=list)
    severity: str = "info"


class _InterpretationModel(BaseModel):
    summary: str = ""
    insights: list[str] = Field(default_factory=list)
    patterns: list[_PatternModel] = Field(default_factory=list)
    severity: str = "info"


class Interpreter:
    """Single LLM-call interpretation with deterministic pattern augmentation."""

    def __init__(self, llm: Any, domain_knowledge: Any) -> None:
        """Initialize.

        Args:
            llm: Pre-built LLM instance with ``with_structured_output``.
            domain_knowledge: ``InsightAgentDomainKnowledge`` instance.
        """
        self.llm = llm
        self.domain_knowledge = domain_knowledge

    def interpret(
        self,
        step_results: dict[int, StepResult],
        intent: dict,
        dim_config: Any,
    ) -> Interpretation:
        """Build an Interpretation grounded in step_results and domain rules."""
        self._enforce_llm_row_budget(step_results, total_cap=50)
        _success_steps = {
            sid: sr for sid, sr in step_results.items()
            if sr.status == StepStatus.SUCCESS and sr.table_summary is not None
        }
        _rows_per_table = [
            (sr.table_summary.row_count if sr.table_summary else 0)
            for sr in _success_steps.values()
        ]
        logger.info(
            "[DEBUG][AGG_INPUT] num_tables=%s rows_per_table=%s",
            len(_success_steps),
            _rows_per_table,
        )

        step_results_block = self._summarize_step_results(step_results)
        metric_values = self._extract_metric_values(step_results)
        metric_thresholds = self._collect_metric_thresholds(metric_values, intent)
        combination_patterns = self._format_combination_patterns()
        channel_rules = self._collect_channel_rules(intent, step_results)

        detected_patterns = self._detect_deterministic_patterns(metric_values)

        _table_names = [sr.dimension for sr in _success_steps.values()]
        _schemas: dict[str, list[str]] = {}
        _samples: dict[str, list[list]] = {}
        _is_aggregated = False
        for sid, sr in _success_steps.items():
            ts = sr.table_summary
            assert ts is not None
            _schemas[str(sid)] = [
                str(c.get("name", "")) for c in (ts.schema or [])
            ]
            if ts.aggregates:
                _is_aggregated = True
            _sample_entries: list[list] = []
            if sr.display_table and sr.display_table.rows:
                for _r in sr.display_table.rows[:5]:
                    _sample_entries.append([str(v)[:80] for v in _r])
            elif ts.top_rows:
                for _r in ts.top_rows[:5]:
                    _sample_entries.append([str(v)[:80] for v in _r])
            _samples[str(sid)] = _sample_entries
        try:
            _samples_json = json.dumps(_samples, default=str)[:800]
        except Exception:
            _samples_json = "<unserializable>"
        logger.info(
            "[DEBUG][AGG_OUTPUT] tables=%s rows_total=%s rows_per_table=%s "
            "is_joined=False is_aggregated=%s table_names=%s schema=%s sample=%s",
            len(_success_steps),
            sum(_rows_per_table),
            _rows_per_table,
            _is_aggregated,
            _table_names,
            _schemas,
            _samples_json,
        )

        user_prompt = INTERPRETATION_PROMPT.format(
            query=intent.get("query", "") or intent.get("user_query", ""),
            step_results=step_results_block,
            metric_thresholds=metric_thresholds,
            combination_patterns=combination_patterns,
            channel_rules=channel_rules,
            primary_analysis=dim_config.primary_analysis,
        )

        logger.info(
            "[DEBUG][INTERPRET_INPUT] prompt_length=%s num_tables=%s total_rows=%s "
            "has_summary=%s prompt_preview=%r",
            len(user_prompt),
            len(_success_steps),
            sum(_rows_per_table),
            bool(step_results_block and step_results_block != "(no step results)"),
            user_prompt[:500],
        )

        try:
            with mlflow.start_span(name="interpreter_llm") as _span:
                _span.set_inputs({
                    "step_results_block": step_results_block[:4000],
                    "metric_thresholds": metric_thresholds[:1000],
                    "combination_patterns": combination_patterns[:1000],
                    "channel_rules": channel_rules[:1000],
                    "primary_analysis": dim_config.primary_analysis,
                    "prompt_length": len(user_prompt),
                    "num_tables": len(_success_steps),
                    "total_rows": sum(_rows_per_table),
                    "user_prompt_preview": user_prompt[:4000],
                })
                structured_llm = self.llm.with_structured_output(_InterpretationModel)
                result: _InterpretationModel = structured_llm.invoke(
                    [
                        SystemMessage(content="You are the Campaign Insight Agent interpreter."),
                        HumanMessage(content=user_prompt),
                    ]
                )
                try:
                    _raw_preview = result.model_dump_json()[:300]
                except Exception:
                    _raw_preview = str(result)[:300]
                _span.set_outputs({
                    "summary": (result.summary or "")[:1500],
                    "insights_count": len(result.insights or []),
                    "patterns_count": len(result.patterns or []),
                    "severity": result.severity or "info",
                })
            logger.info(
                "[DEBUG][INTERPRET_OUTPUT] insights_count=%s patterns_count=%s "
                "raw_output_preview=%r",
                len(result.insights or []),
                len(result.patterns or []),
                _raw_preview,
            )
            llm_patterns = [
                PatternMatch(
                    name=p.name,
                    description=p.description,
                    likely_causes=list(p.likely_causes or []),
                    severity=p.severity or "info",
                )
                for p in (result.patterns or [])
            ]
            merged = self._merge_patterns(detected_patterns, llm_patterns)
            interpretation = Interpretation(
                summary=result.summary or "",
                insights=list(result.insights or []),
                patterns=merged,
                severity=result.severity or self._roll_up_severity(merged),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Interpretation LLM call failed: %s", exc)
            interpretation = Interpretation(
                summary="Interpretation could not be generated.",
                insights=[],
                patterns=detected_patterns,
                severity=self._roll_up_severity(detected_patterns),
            )
        try:
            from dataclasses import asdict as _asdict

            from langgraph.config import get_stream_writer
            writer = get_stream_writer()
            writer({
                "event_type": "phase_progress",
                "phase": "interpret",
                "insights": len(interpretation.insights),
            })
            writer({
                "event_type": "analysis_ready",
                "item_id": "analysis",
                "summary": interpretation.summary,
                "insights": list(interpretation.insights),
                "patterns": [_asdict(p) for p in interpretation.patterns],
                "severity": interpretation.severity or "info",
            })
        except Exception:
            pass
        return interpretation

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _enforce_llm_row_budget(
        step_results: dict[int, StepResult], total_cap: int = 50
    ) -> None:
        """Trim ``table_summary.full_data`` across steps so total rows <= cap.

        Mutates the TableSummary objects in place. Distributes the cap evenly
        across successful steps (minimum 1 row per step). Does not touch
        ``top_rows``, ``bottom_rows``, aggregates, or statistical summary —
        only the raw ``full_data`` that would otherwise bloat the prompt.
        """
        successes = [
            sr for sr in step_results.values()
            if sr.status == StepStatus.SUCCESS
            and sr.table_summary is not None
            and sr.table_summary.full_data
        ]
        if not successes:
            return
        total = sum(
            len((sr.table_summary.full_data if sr.table_summary else None) or [])
            for sr in successes
        )
        if total <= total_cap:
            return
        per_step = max(1, total_cap // len(successes))
        remaining = total_cap
        for sr in successes:
            ts = sr.table_summary
            if ts is None or ts.full_data is None:
                continue
            take = min(per_step, remaining)
            if take <= 0:
                ts.full_data = []
            else:
                ts.full_data = ts.full_data[:take]
            remaining -= len(ts.full_data)
        capped = sum(
            len((sr.table_summary.full_data if sr.table_summary else None) or [])
            for sr in successes
        )
        logger.info(
            "[DEBUG][INTERPRET_TRIM] original_rows=%s capped_rows=%s steps=%s",
            total,
            capped,
            len(successes),
        )

    def _summarize_step_results(self, step_results: dict[int, StepResult]) -> str:
        lines: list[str] = []
        for sid, sr in sorted(step_results.items()):
            if sr.status != StepStatus.SUCCESS or sr.table_summary is None:
                lines.append(
                    f"- step {sid} [{sr.dimension}] status={sr.status.value} "
                    f"error={sr.error_message or ''}"
                )
                continue
            ts = sr.table_summary
            aggregates = ts.aggregates
            statistical_summary = ts.statistical_summary
            top_rows = ts.top_rows
            bottom_rows = ts.bottom_rows
            if ts.mode == "full" and ts.full_data and not ts.aggregates:
                fb = self._fallback_from_full_data(ts.full_data, ts.schema)
                if fb is not None:
                    aggregates, statistical_summary, top_rows, bottom_rows = fb
            lines.append(
                f"- step {sid} [{sr.dimension}] rows={ts.row_count} "
                f"aggregates={aggregates} "
                f"stats={statistical_summary} "
                f"top={top_rows[:3] if top_rows else []} "
                f"bottom={bottom_rows[:3] if bottom_rows else []}"
            )
        return "\n".join(lines) or "(no step results)"

    @staticmethod
    def _fallback_from_full_data(
        full_data: list[list], schema: list[dict]
    ) -> tuple[dict, dict, list[list], list[list]] | None:
        if not full_data:
            return None
        col_names = [
            str(c.get("name", f"col_{i}")) for i, c in enumerate(schema or [])
        ]
        if not col_names:
            return None
        df = pd.DataFrame(full_data, columns=cast(Any, col_names))

        numeric_cols: list[str] = []
        coerced_cols: dict[str, pd.Series] = {}
        for c in col_names:
            cname = c.lower()
            rate_hint = (
                cname.endswith("_rate")
                or cname.endswith("_pct")
                or "rate" in cname
                or "ctr" in cname
                or "cvr" in cname
            )
            series = df[c]
            if rate_hint:
                series = series.apply(
                    lambda v: v[:-1] if isinstance(v, str) and v.endswith("%") else v
                )
            coerced = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
            notna_mask = cast(pd.Series, coerced.notna())
            if int(notna_mask.sum()) > 0 and float(notna_mask.mean()) >= 0.5:
                numeric_cols.append(c)
                coerced_cols[c] = coerced

        logger.info(
            "INTERPRETER.fallback: computed from full_data | rows=%d | numerics=%d",
            len(full_data), len(numeric_cols),
        )

        if not numeric_cols:
            return {}, {}, [], []

        aggregates: dict[str, dict[str, float]] = {}
        statistical_summary: dict[str, dict[str, float]] = {}
        for c in numeric_cols:
            s = coerced_cols[c].dropna()
            if s.empty:
                continue
            aggregates[c] = {
                "sum": round(float(s.sum()), 4),
                "mean": round(float(s.mean()), 4),
            }
            statistical_summary[c] = {
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std(ddof=0)), 4),
            }

        primary = numeric_cols[0]
        try:
            sorted_idx = (
                coerced_cols[primary]
                .sort_values(ascending=False, na_position="last")
                .index.tolist()
            )
            # Edge case: single row → top and bottom reference that same row.
            top_rows = [full_data[i] for i in sorted_idx[:3]]
            bottom_rows = [full_data[i] for i in sorted_idx[-3:]]
        except Exception:  # noqa: BLE001
            top_rows = list(full_data[:3])
            bottom_rows = list(full_data[-3:])

        return aggregates, statistical_summary, top_rows, bottom_rows

    def _extract_metric_values(
        self, step_results: dict[int, StepResult]
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for sr in step_results.values():
            if sr.table_summary is None:
                continue
            for key, val in (sr.table_summary.aggregates or {}).items():
                if isinstance(val, (int, float)):
                    values[key] = float(val)
            for key, stats in (sr.table_summary.statistical_summary or {}).items():
                if isinstance(stats, dict):
                    mean = stats.get("mean")
                    if isinstance(mean, (int, float)):
                        values.setdefault(key, float(mean))
        return values

    def _collect_metric_thresholds(
        self, metric_values: dict[str, float], intent: dict
    ) -> str:
        names = set(metric_values.keys())
        for m in intent.get("metrics", []) or []:
            if isinstance(m, str):
                names.add(m)
        out: list[str] = []
        for name in sorted(names):
            try:
                th = self.domain_knowledge.get_metric_thresholds(name)
                if th:
                    out.append(f"- {name}: {th}")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(out) or "(no threshold data)"

    def _format_combination_patterns(self) -> str:
        try:
            patterns = self.domain_knowledge.detect_combination_patterns({})
            if patterns:
                return "\n".join(f"- {p.name}: {p.description}" for p in patterns)
        except Exception:  # noqa: BLE001
            pass
        try:
            formatted = self.domain_knowledge.format_for_subagent()
            return formatted[:2000]
        except Exception:  # noqa: BLE001
            return "(combination patterns unavailable)"

    def _collect_channel_rules(
        self, intent: dict, step_results: dict[int, StepResult]
    ) -> str:
        channels = set()
        for ch in intent.get("channels", []) or []:
            if isinstance(ch, str):
                channels.add(ch)
        for sr in step_results.values():
            if sr.table_summary is None:
                continue
            for key, dist in (sr.table_summary.categorical_distribution or {}).items():
                if "channel" in key.lower() and isinstance(dist, dict):
                    for ch in dist:
                        if isinstance(ch, str):
                            channels.add(ch)
        out: list[str] = []
        for ch in sorted(channels):
            try:
                rules = self.domain_knowledge.get_channel_constraints(ch)
                if rules:
                    out.append(f"- {ch}: {rules}")
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(out) or "(no channel rules)"

    def _detect_deterministic_patterns(
        self, metric_values: dict[str, float]
    ) -> list[PatternMatch]:
        if not metric_values:
            return []
        try:
            detected = self.domain_knowledge.detect_combination_patterns(metric_values)
        except Exception as exc:  # noqa: BLE001
            logger.debug("detect_combination_patterns failed: %s", exc)
            return []
        out: list[PatternMatch] = []
        for p in detected or []:
            if isinstance(p, PatternMatch):
                out.append(p)
            elif isinstance(p, dict):
                out.append(
                    PatternMatch(
                        name=p.get("name", ""),
                        description=p.get("description", ""),
                        likely_causes=list(p.get("likely_causes", []) or []),
                        severity=p.get("severity", "info"),
                    )
                )
        return out

    def _merge_patterns(
        self,
        deterministic: list[PatternMatch],
        from_llm: list[PatternMatch],
    ) -> list[PatternMatch]:
        by_name: dict[str, PatternMatch] = {}
        for p in deterministic + from_llm:
            key = (p.name or "").strip().lower()
            if not key:
                continue
            if key not in by_name:
                by_name[key] = p
        return list(by_name.values())

    def _roll_up_severity(self, patterns: list[PatternMatch]) -> str:
        order = {"info": 0, "warning": 1, "critical": 2}
        worst = 0
        for p in patterns:
            worst = max(worst, order.get((p.severity or "info").lower(), 0))
        for name, rank in order.items():
            if rank == worst:
                return name
        return "info"
